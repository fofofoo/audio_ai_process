# -*- coding: utf-8 -*-
"""端到端流式 ASR 解码：自研前端 + 官方 ONNX 权重（纯 onnxruntime）。

功能上与权重包自带的 test_onnx.py 等价，但有四处刻意不同：

    1. **前端可切换**。`--frontend ours`（默认）走 traditional_features.py 的
       frontend_for_onnx()，也就是本项目的目标路线；`knf` 切回
       kaldi_native_fbank 作数值标尺；`both` 两条都跑，逐点比特征 + 比识别
       结果，用来证明自研链路相对参考实现没有回归。

    2. **不依赖权重包目录里的任何东西**。脚本、bpe.model 一概不碰，只读三个
       onnx 与 tokens.txt，路径全部由命令行参数传入。权重包是外部下载产物，
       按不可信数据处理：绝不在其目录内执行解释器，否则同目录下被投毒的
       json.py 之流会被 import 钩子带进来。

    3. blank_id 从 tokens.txt 里查（<blk> / <blank>），不硬编码 0。

    4. 额外报告每条的编码耗时与 RTF（实时率），为后面的实时麦克风 ASR 与
       基准测试铺路。

用法：
    python -I src/asr_onnx_decode.py --model-dir <权重目录>
                                 [--frontend ours|knf|both]
                                 [--wav a.wav b.wav ...]
                                 [--out result.txt]

    不指定 --wav 时，自动跑 <权重目录>/test_wavs/*.wav。
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf

# -I 隔离模式下脚本目录不进 sys.path，这里显式加回，才能 import 同目录模块
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from console_utf8 import setup_stdout_utf8
from traditional_features import TraditionalFeatureExtractor

try:
    import kaldi_native_fbank as knf
except ImportError:      # 只在 --frontend knf/both 时才真的需要它
    knf = None

SR = 16000
N_MELS = 80
TAIL_PAD_FRAMES = 100    # 官方 test_onnx.py 也补 100 帧（约 1 秒），照抄


# ============================================================
# 一、基础工具
# ============================================================
def load_audio(path):
    """读 wav 为单通道 float32。非 16k 用 librosa 重采样（与官方一致）。"""
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    data = np.ascontiguousarray(data[:, 0])
    if sr != SR:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=SR).astype(np.float32)
        data = np.ascontiguousarray(data)
        sr = SR
    return data, sr


def load_tokens(path):
    """读 tokens.txt。每行是 `token id`（token 在前、id 在后）。

    用 rpartition 而不是 split()：BPE token 用 ▁ 表示词首空格，本身不含空格，
    但 rpartition 对这种格式更稳，遇到多余空白的行也不会崩。
    """
    id2token = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            token, sep, idx = line.rpartition(" ")
            if not sep:
                continue
            id2token[int(idx)] = token
    return id2token


def find_blank(id2token):
    """从词表里定位 blank 的 id；找不到就退回 0 并提示。"""
    for i, t in id2token.items():
        if t in ("<blk>", "<blank>"):
            return i
    print("警告：tokens.txt 里没找到 <blk>/<blank>，退回 blank_id=0")
    return 0


# ------------------------------------------------------------
# 文本后处理
# ------------------------------------------------------------
_CJK = r"一-鿿㐀-䶿"        # 统一表意文字 + 扩展 A
_FULLWIDTH = r"　-〿＀-￯"  # CJK 标点 + 全角形


def detokenize(tokens):
    """把 BPE token 序列拼成可读文本。

    不能只做 "".join()，原因有两层：

        1. ▁ 是 BPE 的词首标记，得还原成空格，否则得到「▁昨▁天▁是」。
        2. 本模型的词表把中文按单字切（▁礼▁拜▁二），单把 ▁ 换成空格会散成
           「礼 拜 二」。所以还得去空格，但只能去特定位置：
              - CJK 之间          去（「昨 天」→「昨天」）
              - 全角标点两侧      去（「Monday ，today」→「Monday，today」）
              - CJK 与 ASCII 之间 保留（「是 Monday」不能粘成「是Monday」）
    """
    text = "".join(tokens).replace("▁", " ")
    text = re.sub(r"(?<=[%s])\s+(?=[%s])" % (_CJK, _CJK), "", text)
    text = re.sub(r"\s+(?=[%s])" % _FULLWIDTH, "", text)
    text = re.sub(r"(?<=[%s])\s+" % _FULLWIDTH, "", text)
    return text.strip()


def pad_tail(feats):
    """在尾部补 TAIL_PAD_FRAMES 帧 0。

    补的 0 是**特征域**的 0（log 域对应能量 1），不是静音 —— 这是照抄官方
    test_onnx.py 的做法。流式解码循环是 `while start + T < n_frames`，尾部
    不补的话，最后一段凑不满一个 chunk 会被整段丢掉，尾字识别不出来。
    """
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    if TAIL_PAD_FRAMES > 0:
        feats = np.pad(feats, ((0, TAIL_PAD_FRAMES), (0, 0)),
                       mode="constant", constant_values=0.0)
        feats = np.ascontiguousarray(feats)
    return feats


# ============================================================
# 二、两条特征前端
# ============================================================
def feats_knf(samples, sr):
    """kaldi_native_fbank 参考前端（与官方 test_onnx.py 逐参数一致）。"""
    if knf is None:
        raise RuntimeError("kaldi_native_fbank 未安装，无法使用 --frontend knf")
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.window_type = "povey"
    opts.frame_opts.samp_freq = sr
    opts.mel_opts.num_bins = N_MELS
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(sr, samples.tolist())
    fbank.input_finished()
    feats = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
    return pad_tail(feats)


def feats_ours(samples, sr):
    """自研前端：traditional_features.py（默认参数已对齐 kaldi 口径）。"""
    ext = TraditionalFeatureExtractor(sample_rate=sr)
    return ext.frontend_for_onnx(samples, tail_pad_frames=TAIL_PAD_FRAMES)


def compare_feats(ref, ours, tag):
    """比 knf 与自研前端的特征（含尾部 pad 区），打印差异统计。"""
    if ref.shape != ours.shape:
        print("  [%s] 特征形状不一致：knf=%s ours=%s" % (tag, ref.shape, ours.shape))
        return False
    d = np.abs(ref.astype(np.float64) - ours.astype(np.float64))
    fi, bi = np.unravel_index(int(np.argmax(d)), d.shape)
    print("  [%s] 特征 knf vs ours：%s  max|Δ|=%.3e @(帧%d,维%d)  mean|Δ|=%.3e"
          % (tag, ref.shape, d.max(), fi, bi, d.mean()))
    return True


# ============================================================
# 三、ONNX 模型与会话
# ============================================================
class OnnxModel:
    """encoder / decoder / joiner 三个会话 + 流式贪心解码。

    116 个状态张量的形状与 dtype 直接从 encoder 的输入签名推出来（签名里的
    字符串维度，如 'N'，按 batch=1 取 1），解码过程中由每次 run_encoder 的
    返回就地更新，不用手写任何硬编码形状。
    """

    _DTYPES = {"tensor(float)": np.float32,
               "tensor(int64)": np.int64,
               "tensor(int32)": np.int32}

    def __init__(self, encoder_path, decoder_path, joiner_path):
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        so.log_severity_level = 3

        def _sess(p):
            return ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])

        self.encoder = _sess(encoder_path)
        self.decoder = _sess(decoder_path)
        self.joiner = _sess(joiner_path)

        meta = self.encoder.get_modelmeta().custom_metadata_map
        model_type = meta.get("model_type", "")
        if model_type != "zipformer2":
            raise RuntimeError("encoder 的 model_type=%r，本解码循环只支持 "
                               "zipformer2" % model_type)
        self.T = int(meta["T"])
        self.decode_chunk_len = int(meta["decode_chunk_len"])

        dmeta = self.decoder.get_modelmeta().custom_metadata_map
        self.context_size = int(dmeta["context_size"])
        self.vocab_size = int(dmeta["vocab_size"])

        # 名字先缓存，免得解码循环里对每帧都做一次模型反射查询
        self._enc_in = [n.name for n in self.encoder.get_inputs()]
        self._enc_out = [n.name for n in self.encoder.get_outputs()]
        self._dec_in = self.decoder.get_inputs()[0]
        self._dec_out = self.decoder.get_outputs()[0].name
        self._joi_in = [n.name for n in self.joiner.get_inputs()]
        self._joi_out = self.joiner.get_outputs()[0].name

        self.states = None
        self.reset()

    def reset(self):
        """把全部状态张量清零，回到「刚开始解码」的初始态。"""
        states = []
        for n in self.encoder.get_inputs()[1:]:
            if n.type not in self._DTYPES:
                raise RuntimeError("状态张量 %s 的类型 %s 未支持" % (n.name, n.type))
            shape = [1 if isinstance(s, str) else int(s) for s in n.shape]
            states.append(np.zeros(shape, dtype=self._DTYPES[n.type]))
        self.states = states

    def run_encoder(self, x):
        feed = {self._enc_in[0]: x}
        for i, name in enumerate(self._enc_in[1:]):
            feed[name] = self.states[i]
        out = self.encoder.run(self._enc_out, feed)
        self.states = out[1:]
        return out[0]

    def run_decoder(self, hyp):
        """hyp 是长度 context_size 的 token id 列表。"""
        dtype = (np.int64 if self._dec_in.type == "tensor(int64)" else np.int32)
        y = np.array([hyp], dtype=dtype)
        return self.decoder.run([self._dec_out], {self._dec_in.name: y})[0]

    def run_joiner(self, encoder_out, decoder_out):
        return self.joiner.run([self._joi_out],
                               {self._joi_in[0]: encoder_out,
                                self._joi_in[1]: decoder_out})[0]

    def decode(self, features, blank_id=0):
        """流式贪心解码。

        每次取 T 帧喂 encoder、前移 decode_chunk_len 帧；encoder 输出的每一帧
        都过一次 joiner，非 blank 就把 token 接到假设尾部、并用最近
        context_size 个 token 刷新 decoder 状态。
        终止条件与官方一致：start + T 已经越过特征尾部就停。

        返回 (新生成的 token id 列表, 统计字典)。
        """
        self.reset()

        hyp = [blank_id] * self.context_size
        decoder_out = self.run_decoder(hyp)

        T = self.T
        shift = self.decode_chunk_len
        n_total = features.shape[0]
        n_chunks = 0
        n_enc_frames = 0

        start = 0
        while start + T < n_total:
            x = np.ascontiguousarray(features[start:start + T][None])
            start += shift
            n_chunks += 1

            encoder_out = self.run_encoder(x)
            nf = encoder_out.shape[1]
            n_enc_frames += nf
            for k in range(nf):
                cur = encoder_out[0, k:k + 1]
                token_id = int(self.run_joiner(cur, decoder_out).argmax())
                if token_id != blank_id:
                    hyp.append(token_id)
                    decoder_out = self.run_decoder(hyp[-self.context_size:])

        stats = {"n_feat_frames": int(n_total),
                 "n_chunks": n_chunks,
                 "n_enc_frames": n_enc_frames}
        return hyp[self.context_size:], stats


def find_onnx(model_dir, stem):
    """优先 int8，再退回 fp32。"""
    for name in ("%s.int8.onnx" % stem, "%s.onnx" % stem):
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("在 %s 下找不到 %s.int8.onnx 或 %s.onnx"
                            % (model_dir, stem, stem))


# ============================================================
# 四、主流程
# ============================================================
def run_one(model, wav, frontend, id2token, blank_id):
    """解码单个 wav，返回要写进结果文件的文本行。"""
    name = os.path.basename(wav)
    print("\n" + "=" * 72)
    print("[%s]" % name)

    samples, sr = load_audio(wav)
    dur = len(samples) / sr
    print("  音频：sr=%d  样本=%d  时长=%.3f s" % (sr, len(samples), dur))

    outs = []
    if frontend in ("ours", "both"):
        outs.append(("ours", feats_ours(samples, sr)))
    if frontend in ("knf", "both"):
        outs.append(("knf", feats_knf(samples, sr)))

    if frontend == "both":
        d = dict(outs)
        compare_feats(d["knf"], d["ours"], name)

    lines = ["%s  时长=%.3fs" % (name, dur)]
    results = {}
    for tag, feats in outs:
        t0 = time.time()
        ids, stats = model.decode(feats, blank_id=blank_id)
        dt = time.time() - t0
        tokens = [id2token.get(i, "?") for i in ids]
        text = detokenize(tokens)
        results[tag] = (ids, text)
        print("  [%-4s] 特征%s  编码 %d chunk / %d 帧  耗时 %.2fs  RTF=%.3f"
              % (tag, feats.shape, stats["n_chunks"], stats["n_enc_frames"],
                 dt, dt / dur))
        print("  [%-4s] 文本：%s" % (tag, text))
        print("  [%-4s] token：%s" % (tag, "".join(tokens)))
        lines.append("  [%s] RTF=%.3f  %s" % (tag, dt / dur, text))

    if len(results) == 2:
        same = results["ours"][0] == results["knf"][0]
        print("  两条前端识别结果：%s" % ("一致 ✓" if same else "不一致 ✗"))
        lines.append("  两条前端识别结果：%s" % ("一致" if same else "不一致"))
        if not same:
            print("    ours token：%s" % results["ours"][0])
            print("    knf  token：%s" % results["knf"][0])

    lines.append("")
    return lines


def main():
    setup_stdout_utf8()

    ap = argparse.ArgumentParser(
        description="流式 zipformer2 transducer 端到端解码（自研前端 + onnxruntime）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model-dir", required=True,
                    help="权重目录，内含 encoder/decoder/joiner 与 tokens.txt")
    ap.add_argument("--frontend", choices=("ours", "knf", "both"), default="both",
                    help="特征前端：ours=自研模块，knf=kaldi_native_fbank 标尺，"
                         "both=两条都跑并互相比对")
    ap.add_argument("--wav", nargs="*", default=None,
                    help="要解码的 wav；不给则跑 <model-dir>/test_wavs/*.wav")
    ap.add_argument("--out", default=None,
                    help="把识别结果额外写一份 UTF-8 文本到这里（可选）")
    args = ap.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    enc = find_onnx(model_dir, "encoder")
    dec = find_onnx(model_dir, "decoder")
    joi = find_onnx(model_dir, "joiner")
    tok_path = os.path.join(model_dir, "tokens.txt")

    wavs = args.wav
    if not wavs:
        wavs = sorted(glob.glob(os.path.join(model_dir, "test_wavs", "*.wav")))
    if not wavs:
        sys.exit("没有可解码的 wav：请用 --wav 指定")
    if not os.path.isfile(tok_path):
        sys.exit("找不到词表：%s" % tok_path)

    print("encoder : %s" % os.path.basename(enc))
    print("decoder : %s" % os.path.basename(dec))
    print("joiner  : %s" % os.path.basename(joi))
    print("tokens  : %s" % os.path.basename(tok_path))
    print("frontend: %s" % args.frontend)
    print("wavs    : %d 个" % len(wavs))

    id2token = load_tokens(tok_path)
    blank_id = find_blank(id2token)
    print("blank_id: %d (%s)   词表大小=%d"
          % (blank_id, id2token.get(blank_id, "?"), len(id2token)))

    print("正在加载 onnx 会话 ...")
    t0 = time.time()
    model = OnnxModel(enc, dec, joi)
    print("会话就绪，耗时 %.2f s；T=%d  decode_chunk_len=%d  context_size=%d  "
          "vocab_size=%d  状态张量=%d 个"
          % (time.time() - t0, model.T, model.decode_chunk_len,
             model.context_size, model.vocab_size, len(model.states)))

    lines = []
    for wav in wavs:
        lines.extend(run_one(model, wav, args.frontend, id2token, blank_id))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("\n结果已写入：%s" % args.out)


if __name__ == "__main__":
    main()
