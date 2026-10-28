# -*- coding: utf-8 -*-
"""流式特征对齐验证（真实音频）。

用真实 wav 证明 StreamingFramer / StreamingLogFBank 与批量路径**逐位一致**，
并确认流式结果同样继承了批量路径与 kaldi_native_fbank 的对齐。

为什么还要单独验一遍：streaming_features.py 的自测用的是合成正弦，太平稳，
掩盖不了静音段、突发、边界处的差异。真实语音才有这些东西。

控制台编码由 console_utf8 统一设置（Windows 控制台默认 GBK，不转就是乱码）。

用法：
    python -I scripts/probe_streaming_features.py <wav> [<wav> ...]
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 核心库现在在 qt/ 下：根目录用来 import 入口脚本，qt/ 用来 import 核心库
for _p in (_ROOT, os.path.join(_ROOT, "qt")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import soundfile as sf

from console_utf8 import setup_stdout_utf8
from traditional_features import TraditionalFeatureExtractor
from streaming_features import StreamingFramer, StreamingLogFBank

try:
    import kaldi_native_fbank as knf
except ImportError:
    knf = None

SR = 16000
N_MELS = 80
HOPS = (160, 100, 400)      # 160 是麦克风 hop，另两个试探非整除分块


def load(path):
    """读 wav 为单通道 float64。"""
    data, sr = sf.read(path, always_2d=True, dtype="float64")
    return np.ascontiguousarray(data[:, 0]), sr


def stream_all(signal, ext, hop):
    """按 hop 分块喂完整条信号，返回 (原始帧, 特征)。"""
    fr = StreamingFramer(ext)
    fb = StreamingLogFBank(extractor=ext)
    raws, feats = [], []
    for i in range(0, len(signal), hop):
        blk = signal[i:i + hop]
        a = fr.accept_waveform(blk)
        if a.shape[0]:
            raws.append(a)
        b = fb.accept_waveform(blk)
        if b.shape[0]:
            feats.append(b)
    a = fr.input_finished()
    if a.shape[0]:
        raws.append(a)
    b = fb.input_finished()
    if b.shape[0]:
        feats.append(b)
    return (np.concatenate(raws, axis=0) if raws else np.zeros((0, ext.frame_size)),
            np.concatenate(feats, axis=0) if feats else np.zeros((0, N_MELS)))


def knf_ref(signal, sr):
    """kaldi_native_fbank 参考特征（与官方 test_onnx.py 同配置）。"""
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.window_type = "povey"
    opts.frame_opts.samp_freq = sr
    opts.mel_opts.num_bins = N_MELS
    f = knf.OnlineFbank(opts)
    f.accept_waveform(sr, signal.tolist())
    f.input_finished()
    return np.stack([f.get_frame(i) for i in range(f.num_frames_ready)])


def main():
    setup_stdout_utf8()   # Windows 控制台默认 GBK，不转一下中文输出就是乱码
    wavs = sys.argv[1:]
    if not wavs:
        sys.exit("用法：python -I scripts/probe_streaming_features.py <wav> [<wav> ...]")

    bad = 0
    for wav in wavs:
        signal, sr = load(wav)
        ext = TraditionalFeatureExtractor(sample_rate=sr)
        print("=" * 74)
        print("%s   sr=%d   n=%d   dur=%.3fs"
              % (os.path.basename(wav), sr, len(signal), len(signal) / sr))

        ref_frames = np.asarray(ext.frame_signal(signal), dtype=np.float64)
        ref_feats = np.asarray(ext.logfbank(signal), dtype=np.float64)
        print("  批量：帧数=%d  帧长=%d  n_mels=%d"
              % (ref_feats.shape[0], ext.frame_size, ext.n_mels))

        for hop in HOPS:
            got_frames, got_feats = stream_all(signal, ext, hop)

            # 分帧层：要求逐位相同
            n = min(ref_frames.shape[0], got_frames.shape[0])
            d_frame = (np.abs(ref_frames[:n] - got_frames[:n]).max()
                       if n else 0.0)
            same_shape = ref_frames.shape == got_frames.shape

            # 特征层：两边都收敛到 float32 再比（比的是代码，不是 dtype）
            m = min(ref_feats.shape[0], got_feats.shape[0])
            d_feat = (np.abs(ref_feats[:m].astype(np.float32).astype(np.float64)
                             - got_feats[:m].astype(np.float64)).max()
                      if m else 0.0)

            ok = same_shape and d_frame == 0.0 and d_feat == 0.0
            bad += 0 if ok else 1
            print("  hop=%-4d 分帧 %d/%d 帧 max|d|=%.3e | 特征 %d/%d 帧 "
                  "max|d|=%.3e  %s"
                  % (hop, got_frames.shape[0], ref_frames.shape[0], d_frame,
                     got_feats.shape[0], ref_feats.shape[0], d_feat,
                     "BIT-EXACT OK" if ok else "FAIL"))

        # 流式结果是否仍与 kaldi 对齐。
        #   判据不能是「绝对残差小于某个数」：批量路径对 kaldi 本身就有
        #   float32 量级的残差（实测 mean≈5e-6、max≈5e-4，因为 kaldi 全程
        #   float32 而本模块用 float64）。既然流式与批量逐位相同，正确的判据
        #   是「流式对 kaldi 的残差 == 批量对 kaldi 的残差」，一个 bit 都不差。
        if knf is not None:
            ref_knf = knf_ref(signal, sr).astype(np.float64)
            _, got_feats = stream_all(signal, ext, HOPS[0])
            m = min(ref_knf.shape[0], ref_feats.shape[0], got_feats.shape[0])
            # 两边都收敛到 float32 再比，否则量到的是 dtype 而不是代码
            ref_batch32 = ref_feats[:m].astype(np.float32).astype(np.float64)
            d_batch = np.abs(ref_knf[:m] - ref_batch32)
            d_stream = np.abs(ref_knf[:m] - got_feats[:m].astype(np.float64))
            identical = np.array_equal(d_batch, d_stream)
            print("  对 kaldi（hop=%d，%d 帧）：批量 vs knf  max|d|=%.3e "
                  "mean|d|=%.3e" % (HOPS[0], m, d_batch.max(), d_batch.mean()))
            print("                        流式 vs knf  max|d|=%.3e "
                  "mean|d|=%.3e" % (d_stream.max(), d_stream.mean()))
            print("                        两者逐位相同：%s  "
                  "（流式对 kaldi 没有引入任何额外偏差）" % identical)
            bad += 0 if identical else 1

    print()
    print("总判定：%s" % ("全部通过" if bad == 0 else "%d 项不通过" % bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
