# -*- coding: utf-8 -*-
"""流式（实时）zipformer2 transducer：音频块 -> 增量文本。

把三段拼成一条实时链路：

    StreamingLogFBank（streaming_features.py）  ->  音频块变特征帧
    OnnxModel（asr_onnx_decode.py）             ->  encoder/decoder/joiner 会话
    StreamingAsr（本文件）                      ->  增量解码循环 + 文本还原

============================================================
一、增量解码与批量解码必须严格等价
============================================================

批量版（asr_onnx_decode.OnnxModel.decode）是：

    start = 0
    while start + T < n_frames:
        encoder_out = run_encoder(features[start:start+T])   # 状态跨轮累积
        start += decode_chunk_len
        逐帧过 joiner，非 blank 就接进假设

流式版拿到的是不断到来的特征帧，不知道最终的 n_frames，所以只能一边攒一边
喂。关键是**什么时候可以喂**：

    批量第 k 次喂的条件是  48k + 61 < L          （L = 总帧数）
    等价于                  L - 48k > 61
    而流式喂完第 k 次后剩余  L - 48k 帧

所以流式的判据必须是 **len(buf) > T（严格大于）**。写成 >= 就会在
L 恰好等于 48k + 61 时多喂一次 encoder，尾部多吐出几个 token，与批量不一致。

注意这个「严格大于」在麦克风场景下意味着多攒 1 帧（10ms）才喂，可以接受。

============================================================
二、状态怎么跨块存活
============================================================

三样东西必须一直活着，reset() 之外绝不清零：

    1. encoder 的 116 个状态张量  —— 存在 OnnxModel.states 里，每次
       run_encoder 返回新值就地覆盖（这是流式编码器「记住前文」的全部机制）；
    2. 假设序列 hyp              —— 已经确定的 token，decoder 要从它取最近
       context_size 个当输入；
    3. decoder_out               —— 最近一次 decoder 的输出，每帧过 joiner 都要用。
       没吐新 token 之前不重算，省掉大量重复前向。

============================================================
三、用法
============================================================

离线验证（按 chunk 样本一块一块喂整条 wav，并与批量结果对拍）::

    python -I src/streaming_asr.py --model-dir <权重目录> --batch-check

实时麦克风（结束按 Ctrl+C）::

    python -I src/streaming_asr.py --model-dir <权重目录> --mic

    --device 不给就用系统默认输入设备；--list-devices 先看看有哪些设备。
"""

import argparse
import collections
import glob
import os
import queue
import sys
import threading
import time

import numpy as np

# -I 隔离模式下脚本目录不进 sys.path，这里显式加回才能 import 同目录模块
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from asr_onnx_decode import (OnnxModel, detokenize, find_blank, find_onnx,
                             load_audio, load_tokens, setup_stdout_utf8)
from streaming_features import StreamingLogFBank
from traditional_features import TraditionalFeatureExtractor


# ------------------------------------------------------------
# 权重定位：把「用哪份权重」集中在一个地方，两个 GUI 脚本共用
# ------------------------------------------------------------
DEFAULT_WEIGHT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weight")


def find_model_dir(root=DEFAULT_WEIGHT_ROOT):
    """在权重根目录下自动找出权重目录（含 tokens.txt 且有 encoder*.onnx 的那个）。

    不把目录名写死：这批权重是从 ModelScope 下的，带日期后缀，换个版本名字就变了。
    """
    if not os.path.isdir(root):
        raise FileNotFoundError("权重根目录不存在：%s" % root)
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        if (os.path.isfile(os.path.join(d, "tokens.txt"))
                and glob.glob(os.path.join(d, "encoder*.onnx"))):
            return d
    raise FileNotFoundError(
        "%s 下没找到权重目录（需要有 tokens.txt 和 encoder*.onnx）" % root)


# ============================================================
# 一、流式解码器
# ============================================================
class StreamingAsr:
    """增量解码：喂任意长度的音频块，随时能问到当前识别文本。

    参数:
        model_dir       —— 权重目录（三段 onnx + tokens.txt）
        sample_rate     —— 期望采样率，默认 16000
        tail_pad_frames —— 收尾时补几帧特征域 0，默认 100（与官方一致）
    """

    def __init__(self, model_dir, sample_rate=16000, tail_pad_frames=100):
        model_dir = os.path.abspath(model_dir)
        tok_path = os.path.join(model_dir, "tokens.txt")
        if not os.path.isfile(tok_path):
            raise FileNotFoundError("找不到词表：%s" % tok_path)

        self.onnx = OnnxModel(find_onnx(model_dir, "encoder"),
                              find_onnx(model_dir, "decoder"),
                              find_onnx(model_dir, "joiner"))
        self.id2token = load_tokens(tok_path)
        self.blank_id = find_blank(self.id2token)

        self.sample_rate = int(sample_rate)
        self.ext = TraditionalFeatureExtractor(sample_rate=self.sample_rate)
        self.fbank = StreamingLogFBank(extractor=self.ext,
                                       tail_pad_frames=tail_pad_frames)

        # 尺寸参数全部从模型签名里读，不硬编码
        self.T = self.onnx.T
        self.shift = self.onnx.decode_chunk_len
        self.context_size = self.onnx.context_size
        self.n_mels = self.ext.n_mels

        self.reset()

    # --------------------------------------------------------
    def reset(self):
        """回到「刚开始听」，清掉全部跨块状态。"""
        self.onnx.reset()
        self.fbank.reset()
        self.hyp = [self.blank_id] * self.context_size
        self.decoder_out = self.onnx.run_decoder(self.hyp)
        self._feat_buf = np.zeros((0, self.n_mels), dtype=np.float32)

        self.n_samples = 0
        self.n_chunks = 0        # 喂过几次 encoder
        self.n_enc_frames = 0    # encoder 一共吐了多少帧
        self._text = ""

    # --------------------------------------------------------
    # 对外接口
    # --------------------------------------------------------
    @property
    def text(self):
        """当前识别文本（每次有新 token 就整体重新还原一遍）。"""
        return self._text

    @property
    def token_ids(self):
        """已确定的 token id（不含开头的 context_size 个 blank）。"""
        return self.hyp[self.context_size:]

    def accept_audio(self, samples):
        """喂一块原始音频，返回当前完整识别文本。

        参数:
            samples —— 一维波形（[-1,1] 量纲，任意长度；推荐给 160 的整数倍）
        """
        samples = np.asarray(samples, dtype=np.float64).ravel()
        self.n_samples += samples.size
        self._decode_frames(self.fbank.accept_waveform(samples))
        return self._text

    def input_finished(self):
        """宣告音频结束，补完尾帧，返回最终文本。"""
        self._decode_frames(self.fbank.input_finished())
        return self._text

    # --------------------------------------------------------
    # 内部：增量解码循环
    # --------------------------------------------------------
    def _decode_frames(self, feats):
        """把新到的特征接进缓冲，能喂 encoder 就喂，并累积新出现的 token。"""
        if feats.shape[0]:
            self._feat_buf = (np.concatenate((self._feat_buf, feats))
                              if self._feat_buf.shape[0] else feats)

        n_new = 0
        # 判据必须是 > T 而不是 >= T，理由见模块 docstring 第一节
        while self._feat_buf.shape[0] > self.T:
            x = np.ascontiguousarray(self._feat_buf[:self.T][None])
            self._feat_buf = np.ascontiguousarray(self._feat_buf[self.shift:])
            self.n_chunks += 1

            enc_out = self.onnx.run_encoder(x)      # 状态在 onnx.states 里累积
            nf = enc_out.shape[1]
            self.n_enc_frames += nf
            for k in range(nf):
                cur = enc_out[0, k:k + 1]
                tid = int(self.onnx.run_joiner(cur, self.decoder_out).argmax())
                if tid != self.blank_id:
                    self.hyp.append(tid)
                    n_new += 1
                    # 只有真出了新 token 才重算 decoder，否则每帧都白算一次
                    self.decoder_out = self.onnx.run_decoder(
                        self.hyp[-self.context_size:])

        if n_new:
            self._text = detokenize([self.id2token.get(i, "?")
                                     for i in self.token_ids])
        return n_new

    # --------------------------------------------------------
    # 对照用：同一套权重跑一遍批量解码
    # --------------------------------------------------------
    def batch_reference(self, samples):
        """用同一个 onnx 会话跑批量解码，返回 (token id 列表, 文本)。

        **会打乱流式状态**（OnnxModel.decode 内部 reset 了 encoder 状态），
        调用后必须 self.reset() 才能继续流式。只用于验证对拍。
        """
        feats = self.ext.frontend_for_onnx(np.asarray(samples, dtype=np.float64),
                                           tail_pad_frames=self.fbank.tail_pad_frames)
        ids, _ = self.onnx.decode(feats, blank_id=self.blank_id)
        text = detokenize([self.id2token.get(i, "?") for i in ids])
        return ids, text


# ============================================================
# 二、GUI 用的后台识别线程
# ============================================================
class AsrWorker(threading.Thread):
    """在后台线程里跑流式识别，主线程只管塞音频块、读结果文本。

    为什么非得开线程：喂一次 encoder 最坏要约 40ms，而实时监控的画面每 25ms
    就要推一帧，放主线程会让滚动明显一顿一顿。实测过对照（offscreen 下同一段
    音频，开识别 vs 纯可视化）：帧间隔中位数 31.3ms vs 31.2ms，最坏 47.2ms vs
    47.1ms —— 差在噪声级别，等于没有影响。

    数据怎么走：
        主线程  worker.feed(块)  ->  q（有界队列）
        子线程  q.get() -> asr.accept_audio()  ->  self.text
        主线程  读 worker.text 刷新界面

    self.text 每次都是一整个字符串的赋值，CPython 的 GIL 保证主线程读到的
    要么是旧串要么是新串，不会读到半截，所以这里不需要额外加锁。

    离线回放（test_spectrogram.py）和麦克风（realtime_mic_monitor.py）共用这一份。
    """

    def __init__(self, asr, queue_cap=256):
        super().__init__(daemon=True)   # daemon：主线程一退出就跟着结束，不留悬挂线程
        self.asr = asr
        self.q = queue.Queue(maxsize=queue_cap)
        self.text = ""                  # 主线程只读
        self.error = None               # 子线程里抛出的异常，存下来给主线程看
        self._finishing = False

    def feed(self, block):
        """主线程调用：把一块音频交给识别线程；队列满了就丢掉这一块。

        为什么不阻塞等待：识别真跟不上的时候，界面绝不能跟着卡住。丢块会让
        识别漏掉一小段音频，但总比整个界面僵死要好。
        """
        try:
            self.q.put_nowait(np.asarray(block, dtype=np.float32))
        except queue.Full:
            pass

    def finish(self):
        """主线程调用：告知音频结束，让识别线程补尾帧、吐出最终文本。

        用哨兵 None 而不是直接 join：队列里的块会先被消费完，最后才轮到哨兵，
        所以识别一定是从头顺到尾处理完的。
        """
        if self._finishing:
            return
        self._finishing = True
        while True:
            try:
                self.q.put_nowait(None)
                return
            except queue.Full:
                # 队满说明积压了，丢最旧的一块给哨兵腾位置（宁可漏也不要卡）
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass

    def drain(self, timeout=10.0):
        """收尾并等待：先送哨兵（补尾帧、吐最终文本），再等线程结束。

        返回是否正常结束。False 表示超时——注意这时 input_finished() 可能还没
        执行完，self.text 是**缺失尾帧**的结果。

        内部会自己调 finish()（重复调无害），调用方不用先送哨兵，
        否则线程会一直阻塞在 q.get() 上，这里只能等到超时。
        """
        self.finish()
        self.join(timeout)
        return not self.is_alive()

    def run(self):
        try:
            while True:
                blk = self.q.get()
                if blk is None:
                    self.text = self.asr.input_finished()
                    return
                self.text = self.asr.accept_audio(blk)
        except Exception as exc:
            # 子线程里的异常不会自己冒到主线程，存起来让主线程发现并报出来
            self.error = exc


# ============================================================
# 三、离线验证：按块喂 wav，与批量对拍
# ============================================================
def run_wav_streaming(asr, samples, chunk):
    """把整条信号按 chunk 样本一块一块喂进去。

    返回 (最终文本, 每块耗时, 文本变化时间线)，时间线是 (音频秒数, 文本) 列表。
    """
    asr.reset()
    costs, timeline = [], []
    prev = ""
    for i in range(0, len(samples), chunk):
        blk = samples[i:i + chunk]
        t0 = time.perf_counter()
        asr.accept_audio(blk)
        costs.append((time.perf_counter() - t0, blk.size / asr.sample_rate))
        if asr.text != prev:
            prev = asr.text
            timeline.append(((i + blk.size) / asr.sample_rate, prev))
    t0 = time.perf_counter()
    asr.input_finished()
    costs.append((time.perf_counter() - t0, 0.0))
    if asr.text != prev:
        timeline.append((len(samples) / asr.sample_rate, asr.text))
    return asr.text, costs, timeline


def check_wav(asr, wav, chunks=(160,)):
    name = os.path.basename(wav)
    samples, sr = load_audio(wav)
    dur = len(samples) / sr
    print("\n" + "=" * 74)
    print("[%s]  sr=%d  样本=%d  时长=%.3fs" % (name, sr, len(samples), dur))

    ref_ids, ref_text = asr.batch_reference(samples)
    print("  批量：%d token  %s" % (len(ref_ids), ref_text))

    # 实时性的分母不是「单块时长」：绝大多数块根本不喂 encoder，真正费时的
    #   只有喂的那几次，而两次喂之间隔着 decode_chunk_len 帧的音频。
    gap_sec = asr.shift * asr.ext.frame_step / asr.sample_rate
    for chunk in chunks:
        text, costs, timeline = run_wav_streaming(asr, samples, chunk)
        ids = asr.token_ids
        same = ids == ref_ids
        total = sum(c for c, _ in costs)
        worst = max((c for c, d in costs if d > 0), default=0.0)
        print("  流式(chunk=%-4d)：%d token  喂了 %d 次 encoder  耗时 %.2fs  "
              "RTF=%.3f" % (chunk, len(ids), asr.n_chunks, total, total / dur))
        print("      单块最坏耗时 %.2fms，而两次喂之间隔着 %.0fms 音频 -> "
              "峰值负载 %.1f%%  %s"
              % (worst * 1e3, gap_sec * 1e3, worst / gap_sec * 100,
                 "跟得上实时" if worst < gap_sec else "跟不上！"))
        print("      与批量 token 序列：%s" % ("完全一致" if same else "不一致"))
        if not same:
            print("        批量：%s" % ref_ids[:40])
            print("        流式：%s" % ids[:40])
        # 文本是边听边长出来的，不是等音频结束才一次吐出：看它变化了多少次
        if timeline:
            print("      文本随音频增长：%d 次刷新，%.2fs 时首次出字（音频长 %.2fs）"
                  % (len(timeline), timeline[0][0], dur))
            print("        %.2fs: %s" % timeline[0])
            print("        %.2fs: %s" % timeline[-1])
        print("      最终文本：%s" % text)
    return


# ============================================================
# 四、实时麦克风
# ============================================================
def list_devices():
    import sounddevice as sd
    print(sd.query_devices())
    print("\n默认输入设备：%s" % (sd.query_devices(kind="input"),))


def run_mic(asr, device=None, block=160, channels=1, max_blocks=64, seconds=0):
    """实时麦克风识别。seconds=0 表示不限时（Ctrl+C 结束）。

    线程与缓冲的两条铁律（realtime_mic_monitor.py 里踩过）：
        1. PortAudio 回调线程里**只能**做一次 .copy() 再塞进缓冲，
           绝不碰模型、不碰 Qt；indata 的底层缓冲会被 PortAudio 复用，
           不 copy 的话读到的可能是已经被覆盖的脏数据。
        2. 消费端跟不上时丢**最旧**的块（用 deque(maxlen)，append 自动淘汰）。
           实时识别要的是「现在」，堆积只会让延迟越来越大。
    """
    import sounddevice as sd

    q = collections.deque(maxlen=max_blocks)
    dropped = [0]
    peak = [0.0]

    def callback(indata, frames, time_info, status):
        if status:
            print("\n[音频状态] %s" % status, file=sys.stderr)
        if len(q) == q.maxlen:
            dropped[0] += 1
        blk = indata[:, 0].copy()          # 必须 copy，理由见 docstring
        peak[0] = max(peak[0], float(np.abs(blk).max()))
        q.append(blk)

    host_sr = asr.sample_rate
    print("正在打开麦克风（device=%s, %d Hz, %d 声道, 每块 %d 样本=%.1fms）..."
          % (device if device is not None else "系统默认", host_sr, channels,
             block, block / host_sr * 1e3))
    if seconds:
        print("限时 %g 秒。\n" % seconds)
    else:
        print("说话即可看到文字；Ctrl+C 结束。\n")

    asr.reset()
    t_start = time.perf_counter()
    n_blocks = 0
    busy = 0.0
    worst = 0.0
    shown = None
    try:
        with sd.InputStream(samplerate=host_sr, channels=channels,
                            dtype="float32", blocksize=block,
                            device=device, callback=callback):
            while True:
                if seconds and time.perf_counter() - t_start >= seconds:
                    break
                if q:
                    blk = q.popleft()
                    n_blocks += 1
                    t0 = time.perf_counter()
                    asr.accept_audio(blk)
                    dt = time.perf_counter() - t0
                    busy += dt
                    worst = max(worst, dt)
                    # 只在文字真变了才重画：\r 原地刷新，但不该白刷一屏
                    if asr.text != shown:
                        shown = asr.text
                        sys.stdout.write("\r%s" % shown.ljust(70))
                        sys.stdout.flush()
                else:
                    time.sleep(0.002)
    except KeyboardInterrupt:
        pass

    dur = time.perf_counter() - t_start
    text = asr.input_finished()
    print("\n")
    print("=" * 74)
    print("音频 %.1fs，收了 %d 块，丢弃 %d 块，峰值幅度 %.3e"
          % (dur, n_blocks, dropped[0], peak[0]))
    print("喂了 %d 次 encoder，共 %d 编码帧，%d token"
          % (asr.n_chunks, asr.n_enc_frames, len(asr.token_ids)))
    print("处理耗时占比 = %.1f%%（单块最坏 %.2fms / 块时长 %.2fms）"
          % (busy / dur * 100 if dur else 0.0,
             worst * 1e3, block / host_sr * 1e3))
    print("识别结果：%s" % text)


# ============================================================
# 五、命令行
# ============================================================
def main():
    setup_stdout_utf8()
    ap = argparse.ArgumentParser(
        description="流式 zipformer2 transducer：离线按块对拍 / 实时麦克风",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="权重目录")
    ap.add_argument("--wav", nargs="*", default=None,
                    help="离线验证用的 wav；不给则跑 <model-dir>/test_wavs/*.wav")
    ap.add_argument("--chunk", type=int, nargs="*", default=[160],
                    help="离线验证的块长（样本数），可给多个")
    ap.add_argument("--batch-check", action="store_true",
                    help="离线按块喂并与批量解码对拍 token 序列")
    ap.add_argument("--mic", action="store_true", help="实时麦克风识别")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="麦克风模式限时多少秒后自动结束；0 = 不限时")
    ap.add_argument("--device", default=None,
                    help="输入设备号；不给用系统默认")
    ap.add_argument("--list-devices", action="store_true", help="列出音频设备后退出")
    args = ap.parse_args()

    if args.list_devices:
        list_devices()
        return

    print("正在加载权重 ...")
    t0 = time.time()
    asr = StreamingAsr(args.model_dir)
    print("就绪，耗时 %.2fs；T=%d  decode_chunk_len=%d  context_size=%d  "
          "blank_id=%d  词表=%d"
          % (time.time() - t0, asr.T, asr.shift, asr.context_size,
             asr.blank_id, len(asr.id2token)))

    if args.mic:
        run_mic(asr, device=args.device, seconds=args.seconds)
        return

    if not args.batch_check:
        print("\n没指定 --batch-check 或 --mic，什么都没做。")
        return

    import glob
    wavs = args.wav
    if not wavs:
        wavs = sorted(glob.glob(os.path.join(os.path.abspath(args.model_dir),
                                             "test_wavs", "*.wav")))
    if not wavs:
        sys.exit("没有可解码的 wav：请用 --wav 指定")

    for wav in wavs:
        check_wav(asr, wav, chunks=tuple(args.chunk))


if __name__ == "__main__":
    main()
