# -*- coding: utf-8 -*-
"""探针：读取权重包元数据，并把 kaldi_native_fbank 的参考特征与
traditional_features.py 的输出做对比。

与权重包分开放（本文件在 scripts/ 下）；权重路径全部由命令行参数传入。
用 -I 隔离模式运行，避免解释器从权重包目录加载到外部代码。

用法：
    python -I scripts/probe_knf_reference.py <encoder> <decoder> <joiner> <wav>

注意：控制台编码由 console_utf8 统一设置。
"""
import os
import sys

# -I 模式下脚本目录不进 sys.path，这里显式加回，才能 import 到项目根下的模块
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 核心库现在在 qt/ 下：根目录用来 import 入口脚本，qt/ 用来 import 核心库
for _p in (_ROOT, os.path.join(_ROOT, "qt")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import onnxruntime as ort
import kaldi_native_fbank as knf
import soundfile as sf

from console_utf8 import setup_stdout_utf8
from traditional_features import TraditionalFeatureExtractor


def open_session(path, tag, max_dump=6):
    """打开 onnx 会话并打印元数据 / 输入输出签名（超长的只列前几个）。"""
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.log_severity_level = 3
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

    print("=" * 72)
    print("[%s] %s" % (tag, os.path.basename(path)))
    meta = sess.get_modelmeta().custom_metadata_map
    if meta:
        print("  metadata:")
        for k in sorted(meta):
            print("    %-24s = %s" % (k, meta[k]))
    for label, items in (("inputs", sess.get_inputs()),
                         ("outputs", sess.get_outputs())):
        print("  %s (%d):" % (label, len(items)))
        for it in items[:max_dump]:
            print("    %-24s %-20s %s" % (it.name, str(it.shape), it.type))
        if len(items) > max_dump:
            print("    ... (%d more)" % (len(items) - max_dump))
    return sess


def knf_features(samples, sample_rate, num_bins=80):
    """严格照 test_onnx.py 的配置提取 kaldi fbank 特征。"""
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.window_type = "povey"
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = num_bins
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(sample_rate, samples.tolist())
    fbank.input_finished()
    n = fbank.num_frames_ready
    feats = np.stack([fbank.get_frame(i) for i in range(n)])
    return feats, opts


def main():
    setup_stdout_utf8()   # Windows 控制台默认 GBK，不转一下中文输出就是乱码
    enc_path, dec_path, joi_path, wav_path = sys.argv[1:5]

    open_session(enc_path, "encoder")
    open_session(dec_path, "decoder")
    open_session(joi_path, "joiner")

    # ---- 读音频（与 test_onnx.py 一致：float32、取第 0 通道）----
    samples, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    samples = np.ascontiguousarray(samples[:, 0])
    print("=" * 72)
    print("wav: %s" % os.path.basename(wav_path))
    print("  sr=%d  n=%d  dur=%.3f s  range=[%.6f, %.6f]" %
          (sr, len(samples), len(samples) / sr, samples.min(), samples.max()))

    # ---- 参考：kaldi_native_fbank ----
    feats, opts = knf_features(samples, sr)
    fo = opts.frame_opts
    mo = opts.mel_opts
    win_size = int(round(fo.frame_length_ms * sr / 1000.0))
    win_shift = int(round(fo.frame_shift_ms * sr / 1000.0))

    print("=" * 72)
    print("[knf reference] shape=%s" % (feats.shape,))
    print("  frame_opts: snip_edges=%s window=%s dither=%s" %
          (fo.snip_edges, fo.window_type, fo.dither))
    print("  frame_opts: win_size=%d win_shift=%d preemph=%.4f remove_dc=%s" %
          (win_size, win_shift, fo.preemph_coeff, fo.remove_dc_offset))
    print("  frame_opts: round_to_power_of_two=%s" % fo.round_to_power_of_two)
    print("  mel_opts  : num_bins=%d low_freq=%d high_freq=%d" %
          (mo.num_bins, mo.low_freq, mo.high_freq))
    print("  value range: [%.4f, %.4f]  mean=%.4f" %
          (feats.min(), feats.max(), feats.mean()))
    print("  frame count: round(N/S)=%d  actual=%d" %
          (int(round(len(samples) / win_shift)), feats.shape[0]))

    # ---- 我方模块：当前实现，两种窗各跑一次 ----
    print("=" * 72)
    for win in ("hamming", "povey"):
        ext = TraditionalFeatureExtractor(sample_rate=sr, window=win)
        ours = ext.logfbank(samples)
        print("[ours/%-7s] shape=%s  range=[%.4f, %.4f]  mean=%.4f" %
              (win, ours.shape, ours.min(), ours.max(), ours.mean()))
        n = min(feats.shape[0], ours.shape[0])
        if n > 0:
            d = np.abs(feats[:n].astype(np.float64) - ours[:n])
            print("             first %d frames |delta|: max=%.4f mean=%.4f" %
                  (n, d.max(), d.mean()))


if __name__ == "__main__":
    main()
