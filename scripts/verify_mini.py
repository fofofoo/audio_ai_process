# -*- coding: utf-8 -*-
"""mini 版与 src 版逐点对拍 —— 把 mini 当第三把数值标尺来验。

============================================================
这份脚本验什么
============================================================
mini/ 下的实现是**独立重写**的（不 import src/ 任何东西），所以它天然可以
拿来跟 src/ 互相印证：两边算的是同一件事，若逐点一致，说明两份实现都没写错。

    A 部分：前端对拍
        src.traditional_features.TraditionalFeatureExtractor.frontend_for_onnx
        vs mini.audio_preprocess.AudioPreprocess
        同一段 wav，mini 侧按多种块长流式喂（160/257/400/1000/一次全喂），
        逐点比 max|Δ|。另测极短信号打「edge 退化分支」。

    B 部分：解码端到端对拍
        同一段 wav，src 走「批量前端 + OnnxModel.decode」，
        mini 走「流式前端 + AsrDecoder.accept_features」，比 **token 序列**
        是否完全一致。这一条同时验 mini 的解码不变量（> T 判据、尾部补 100 帧、
        blank 不刷新 decoder、状态跨块存活）。

为什么期望「几乎逐点相同」而不是「大致相同」：
    src 的 mel 滤波器组与窗函数都是 float64、公式与 mini 逐字一致；两边唯一的
    差异是求和的次序（src 用 power @ filters.T 走 BLAS，mini 用显式稀疏循环）
    与 |z|² 的实现（np.abs vs real²+imag²）。这些在 float64 下只差最后几个 bit，
    而两边最终都 cast 到 float32，所以结果应当逐点相同，或最多差 1 个 ulp。

跑法（按 CLAUDE.md：一律 python -I，且脚本自己插 sys.path）：
    python -I scripts/verify_mini.py --part a          # 只跑前端对拍（快）
    python -I scripts/verify_mini.py --part b          # 只跑解码对拍（要加载 onnx）
    python -I scripts/verify_mini.py                   # 全跑
    python -I scripts/verify_mini.py --part b --wav <某.wav> --frame-chunks 1 37
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

# -I 隔离模式下脚本目录与 PYTHONPATH 都不进 sys.path，三个目录都得手动插：
#   项目根（只为保持与其它脚本一致）、src/（被对拍方）、mini/（待验方）
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "mini")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from console_utf8 import setup_stdout_utf8
from traditional_features import TraditionalFeatureExtractor, read_wav
from audio_preprocess import AudioPreprocess

# 尾部补帧数。与 src/asr_onnx_decode.py 的 TAIL_PAD_FRAMES 是同一个值，
# 这里单独写死是为了让 A 部分不必 import asr_onnx_decode（那会拖进
# onnxruntime 与 soundfile）。B 部分会断言两边确实相等。
TAIL_PAD_FRAMES = 100

WEIGHT_ROOT = os.path.join(_ROOT, "weight")


# ============================================================
# 工具
# ============================================================
def find_model_dir(root=WEIGHT_ROOT):
    """在 weight/ 下找权重目录：同时有 tokens.txt 与 encoder*.onnx。"""
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if (os.path.isdir(d)
                and os.path.isfile(os.path.join(d, "tokens.txt"))
                and glob.glob(os.path.join(d, "encoder*.onnx"))):
            return d
    raise FileNotFoundError("weight/ 下没找到权重目录")


def mini_frontend_stream(samples, chunk, n_mels=80, tail_pad=TAIL_PAD_FRAMES):
    """把整段波形按 chunk 长切块，流式喂 mini 前端，返回拼好的特征矩阵。

    chunk <= 0 表示一次全喂（走批量路径，用于确认流式与批量等价）。
    """
    front = AudioPreprocess(n_mels=n_mels, tail_pad_frames=tail_pad)
    step = len(samples) if chunk <= 0 else int(chunk)
    parts = []
    for i in range(0, len(samples), max(1, step)):
        f = front.accept_waveform(samples[i:i + max(1, step)])
        if f.shape[0]:
            parts.append(f)
    f = front.input_finished()
    if f.shape[0]:
        parts.append(f)
    if not parts:
        return np.zeros((0, n_mels), dtype=np.float32)
    return np.concatenate(parts)


def cmp_feats(ref, got, tag):
    """比两份特征，打印形状 / max|Δ| / mean|Δ|。返回是否通过。"""
    if ref.shape != got.shape:
        print("  %-34s ✗ 形状不一致  src=%s  mini=%s"
              % (tag, ref.shape, got.shape))
        return False
    if ref.size == 0:
        print("  %-34s ✓ 两边都是空 %s" % (tag, ref.shape))
        return True
    d = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    bi = int(d.argmax())
    fi, mi = divmod(bi, d.shape[1])
    maxd, meand = float(d.max()), float(d.mean())
    # float32 特征的值域大约 [-16, 10]，1e-5 已经是「明显不同」的量级；
    # 逐点相同或差 1 个 ulp 时 max|Δ| 会在 1e-6 以下。
    ok = maxd < 1e-5
    print("  %-34s %s %s  max|Δ|=%.3e @(帧%d,维%d)  mean|Δ|=%.3e"
          % (tag, "✓" if ok else "✗", ref.shape, maxd, fi, mi, meand))
    return ok


# ============================================================
# A 部分：前端对拍
# ============================================================
def part_a(wavs, chunk_sizes):
    print("\n" + "=" * 78)
    print("A. 前端对拍：src.TraditionalFeatureExtractor  vs  mini.AudioPreprocess")
    print("=" * 78)
    all_ok = True

    for wav in wavs:
        samples, sr = read_wav(wav)
        ext = TraditionalFeatureExtractor(sample_rate=sr)
        ref = ext.frontend_for_onnx(samples, tail_pad_frames=TAIL_PAD_FRAMES)
        print("\n[%s]  %d 样本 (%.2f s)  nf=%d + %d 帧补零 -> %s"
              % (os.path.basename(wav), len(samples), len(samples) / sr,
                 ref.shape[0] - TAIL_PAD_FRAMES, TAIL_PAD_FRAMES, ref.shape))
        for chunk in chunk_sizes:
            tag = ("一次全喂" if chunk <= 0 else "块长 %d 样本" % chunk)
            got = mini_frontend_stream(samples, chunk)
            all_ok &= cmp_feats(ref, got, tag)

    # ---- 极短信号：打 edge 退化分支（numpy 镜像补齐宽度不得 >= 原数组长）----
    print("\n" + "-" * 78)
    print("极短信号（n <= 140 触发 edge 退化补齐；n = 60 连一帧都凑不出）")
    print("  n = 80 是分界：n < 80 时 round(n/S) = 0，两边都只有 100 帧补零")
    print("-" * 78)
    base, _ = read_wav(wavs[0])
    for n in (0, 60, 79, 80, 120, 139, 140, 141, 280, 281):
        seg = np.ascontiguousarray(base[:n], dtype=np.float64)
        ref = TraditionalFeatureExtractor(sample_rate=16000) \
            .frontend_for_onnx(seg, tail_pad_frames=TAIL_PAD_FRAMES)
        for chunk in (64, 160):
            got = mini_frontend_stream(seg, chunk)
            all_ok &= cmp_feats(ref, got, "n=%d  块长 %d" % (n, chunk))
    return all_ok


# ============================================================
# B 部分：解码端到端对拍
# ============================================================
def mini_decode_stream(dec, samples, frame_chunk):
    """mini 全链路：波形 -> 流式前端 -> 分块喂解码器。返回 (token ids, 统计)。

    frame_chunk 是每次 accept_features 喂多少**帧**（不是样本），
    用来检验解码缓冲的拼接与 > T 判据在任何切法下都等价。
    """
    dec.reset()
    front = AudioPreprocess(n_mels=dec.n_mels, tail_pad_frames=TAIL_PAD_FRAMES)
    step = len(samples) if frame_chunk <= 0 else None
    if step is not None:
        dec.accept_features(front.accept_waveform(samples))
    else:
        # 按 frame_chunk 帧切：先攒够 frame_chunk 帧再喂一次
        buf = []
        n_in_buf = 0
        pos = 0
        SR_CHUNK = 160                      # 每次先喂 10ms 样本给前端
        while pos < len(samples):
            f = front.accept_waveform(samples[pos:pos + SR_CHUNK])
            pos += SR_CHUNK
            if f.shape[0]:
                buf.append(f)
                n_in_buf += f.shape[0]
            while n_in_buf >= frame_chunk:
                cat = np.concatenate(buf)
                dec.accept_features(cat[:frame_chunk])
                rest = cat[frame_chunk:]
                buf = [rest] if rest.shape[0] else []
                n_in_buf = rest.shape[0]
        if n_in_buf:
            dec.accept_features(np.concatenate(buf))
    dec.accept_features(front.input_finished())
    return list(dec.token_ids), {"n_chunks": dec.n_chunks,
                                 "n_enc_frames": dec.n_enc_frames,
                                 "n_feat_frames": front.n_frames}


def part_b(wavs, model_dir, frame_chunks):
    # 延迟到这里才 import：会加载 onnxruntime / soundfile，A 部分用不到
    from asr_onnx_decode import (OnnxModel, find_onnx, load_tokens,
                                 find_blank, detokenize)
    from asr_onnx_decode import TAIL_PAD_FRAMES as SRC_TAIL_PAD
    # mini 侧的文本还原是 AsrDecoder 的实例方法，走已建好的 mini_model 调用
    from asr_decode import AsrDecoder

    assert SRC_TAIL_PAD == TAIL_PAD_FRAMES, \
        "尾部补帧数两边不一致：src=%d 本脚本=%d" % (SRC_TAIL_PAD, TAIL_PAD_FRAMES)

    print("\n" + "=" * 78)
    print("B. 解码端到端对拍：src 批量  vs  mini 流式（比 token 序列）")
    print("=" * 78)

    id2token = load_tokens(os.path.join(model_dir, "tokens.txt"))
    blank_id = find_blank(id2token)

    print("正在加载 src 的 OnnxModel ...")
    t0 = time.time()
    src_model = OnnxModel(find_onnx(model_dir, "encoder"),
                          find_onnx(model_dir, "decoder"),
                          find_onnx(model_dir, "joiner"))
    print("  就绪 %.2fs  T=%d  decode_chunk_len=%d  context_size=%d  状态=%d"
          % (time.time() - t0, src_model.T, src_model.decode_chunk_len,
             src_model.context_size, len(src_model.states)))

    print("正在加载 mini 的 AsrDecoder ...")
    t0 = time.time()
    # mini 的 AsrDecoder 现在只接收现成的 session（权重定位与执行器选择都
    # 归入口脚本管），所以这里自己把三个建好。固定走 CPU：src 侧就是 CPU，
    # 两边同条件才对拍得公平。
    import onnxruntime as ort
    _so = ort.SessionOptions()
    _so.inter_op_num_threads = 1
    _so.intra_op_num_threads = 1
    _so.log_severity_level = 3

    def _mini_sess(stem):
        return ort.InferenceSession(find_onnx(model_dir, stem), _so,
                                    providers=["CPUExecutionProvider"])

    mini_model = AsrDecoder(_mini_sess("encoder"), _mini_sess("decoder"),
                            _mini_sess("joiner"),
                            os.path.join(model_dir, "tokens.txt"))
    info = mini_model.info()
    print("  就绪 %.2fs  %s" % (time.time() - t0, info))

    # 两个解码器的静态配置必须一致
    assert src_model.T == mini_model.input_frame_num, "T 不一致"
    assert src_model.decode_chunk_len == mini_model.T_shift, "decode_chunk_len 不一致"
    assert src_model.context_size == mini_model.context_size, "context_size 不一致"
    assert blank_id == mini_model.blank_id, "blank_id 不一致"
    print("  静态配置一致 ✓  (T / decode_chunk_len / context_size / blank_id)")

    all_ok = True
    for wav in wavs:
        samples, sr = read_wav(wav)
        ext = TraditionalFeatureExtractor(sample_rate=sr)
        ref_feats = ext.frontend_for_onnx(samples, tail_pad_frames=TAIL_PAD_FRAMES)

        t0 = time.time()
        ids_src, st_src = src_model.decode(ref_feats, blank_id=blank_id)
        txt_src = detokenize([id2token.get(i, "?") for i in ids_src])
        t_src = time.time() - t0
        print("\n[%s]  %.2f s  特征%s" % (os.path.basename(wav),
                                          len(samples) / sr, ref_feats.shape))
        print("  src   批量 : chunk=%d  enc帧=%d  %.2fs  RTF=%.3f"
              % (st_src["n_chunks"], st_src["n_enc_frames"], t_src,
                 t_src / (len(samples) / sr)))
        print("  src   文本 : %s" % txt_src)

        for fc in frame_chunks:
            t0 = time.time()
            ids_mini, st_mini = mini_decode_stream(mini_model, samples, fc)
            txt_mini = mini_model.detokenize(
                [id2token.get(i, "?") for i in ids_mini])
            t_mini = time.time() - t0
            same = (list(ids_src) == ids_mini)
            all_ok &= same
            tag = ("一次全喂" if fc <= 0 else "每次 %d 帧" % fc)
            print("  mini  %-8s: chunk=%d  enc帧=%d  %.2fs  RTF=%.3f  token=%d %s"
                  % (tag, st_mini["n_chunks"], st_mini["n_enc_frames"], t_mini,
                     t_mini / (len(samples) / sr), len(ids_mini),
                     "一致 ✓" if same else "不一致 ✗"))
            if not same:
                print("    src  token：%s" % list(ids_src))
                print("    mini token：%s" % ids_mini)
                # 找出第一处分歧
                for j, (a, b) in enumerate(zip(ids_src, ids_mini)):
                    if a != b:
                        print("    首个分歧 @%d：src=%d(%s)  mini=%d(%s)"
                              % (j, a, id2token.get(a, "?"), b,
                                 id2token.get(b, "?")))
                        break
                print("    src  文本：%s" % txt_src)
                print("    mini 文本：%s" % txt_mini)
            # 额外核对：mini 的 chunk 数 / encoder 帧数应与 src 完全相同
            if st_mini["n_chunks"] != st_src["n_chunks"] or \
                    st_mini["n_enc_frames"] != st_src["n_enc_frames"]:
                print("    ✗ 切块统计不同：src chunk/帧=%d/%d  mini=%d/%d"
                      % (st_src["n_chunks"], st_src["n_enc_frames"],
                         st_mini["n_chunks"], st_mini["n_enc_frames"]))
                all_ok = False
    return all_ok


def main():
    setup_stdout_utf8()
    ap = argparse.ArgumentParser(
        description="mini 版与 src 版逐点对拍",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--part", choices=("a", "b", "ab"), default="ab")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--wav", nargs="*", default=None)
    ap.add_argument("--chunks", nargs="*", type=int,
                    default=[160, 257, 400, 1000, 0],
                    help="A 部分的音频块长（样本）；0 = 一次全喂")
    ap.add_argument("--frame-chunks", nargs="*", type=int,
                    default=[1, 7, 37, 0],
                    help="B 部分每次喂多少帧特征；0 = 一次全喂")
    args = ap.parse_args()

    model_dir = os.path.abspath(args.model_dir) if args.model_dir \
        else find_model_dir()
    print("权重目录：%s" % model_dir)

    wavs = args.wav or sorted(glob.glob(
        os.path.join(model_dir, "test_wavs", "*.wav")))
    if not wavs:
        sys.exit("没有可用的 wav")
    print("音频：%d 个" % len(wavs))

    ok = True
    if args.part in ("a", "ab"):
        ok &= part_a(wavs, args.chunks)
    if args.part in ("b", "ab"):
        ok &= part_b(wavs, model_dir, args.frame_chunks)

    print("\n" + "=" * 78)
    print("总体结论：%s" % ("全部一致 ✓" if ok else "存在不一致 ✗，见上方 ✗ 行"))
    print("=" * 78)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
