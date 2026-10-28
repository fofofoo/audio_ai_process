# -*- coding: utf-8 -*-
"""逐组件比对：把 kaldi_native_fbank 暴露出来的窗函数 / Mel 滤波器组
与 traditional_features.py 的内部实现逐点对比，定位差异来源。

只依赖 knf + traditional_features，不加载 onnx，跑得很快。
控制台编码由 console_utf8 统一设置。

用法：
    python -I scripts/probe_knf_components.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 核心库现在在 qt/ 下：根目录用来 import 入口脚本，qt/ 用来 import 核心库
for _p in (_ROOT, os.path.join(_ROOT, "qt")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import kaldi_native_fbank as knf

from console_utf8 import setup_stdout_utf8
from traditional_features import TraditionalFeatureExtractor

SR = 16000
FRAME_LEN_MS = 25.0
FRAME_SHIFT_MS = 10.0
N_FFT = 512
N_MELS = 80


def base_frame_opts(**kw):
    """一份与 test_onnx.py 等价的 FrameExtractionOptions。"""
    o = knf.FrameExtractionOptions()
    o.samp_freq = SR
    o.frame_length_ms = FRAME_LEN_MS
    o.frame_shift_ms = FRAME_SHIFT_MS
    o.dither = 0.0
    o.snip_edges = False
    o.window_type = "povey"
    o.remove_dc_offset = True
    o.preemph_coeff = 0.97
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def cmp_arr(label, a, b):
    """打印两个数组的差异统计；形状不同则只报形状。"""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        print("  %-16s SHAPE MISMATCH  kaldi=%s  ours=%s" %
              (label, a.shape, b.shape))
        return
    d = np.abs(a - b)
    flag = "OK " if d.max() < 1e-6 else ">>>"
    print("  %-16s %s n=%-6d max|d|=%.6e  mean|d|=%.6e" %
          (label, flag, a.size, d.max(), d.mean()))


def section(title):
    print("=" * 72)
    print(title)


def main():
    setup_stdout_utf8()   # Windows 控制台默认 GBK，不转一下中文输出就是乱码
    ext = TraditionalFeatureExtractor(sample_rate=SR, n_fft=N_FFT, n_mels=N_MELS)

    # ----------------------------------------------------------------
    # A. 窗函数
    # ----------------------------------------------------------------
    section("A. window function")
    for wtype in ("povey", "hamming", "hann", "hanning", "sine", "blackman",
                  "rectangular"):
        try:
            kwin = np.array(knf.FeatureWindowFunction(
                base_frame_opts(window_type=wtype)).window, dtype=np.float64)
        except Exception as e:
            print("  %-16s kaldi build FAILED: %s" % (wtype, e))
            continue
        try:
            mine = TraditionalFeatureExtractor(sample_rate=SR, window=wtype)._win
        except Exception as e:
            print("  %-16s ours  build FAILED: %s" % (wtype, e))
            continue
        cmp_arr(wtype, kwin, mine)
        if wtype in ("povey", "hamming"):
            n = len(kwin)
            for denom in (n - 1, n):
                idx = np.arange(n)
                if wtype == "povey":
                    cand = (0.5 - 0.5 * np.cos(2.0 * np.pi * idx / denom)) ** 0.85
                else:
                    cand = 0.54 - 0.46 * np.cos(2.0 * np.pi * idx / denom)
                print("         %s with denom=%-4d max|d| vs kaldi = %.6e" %
                      (wtype, denom, np.abs(cand - kwin).max()))

    # ----------------------------------------------------------------
    # B. Mel 滤波器组
    # ----------------------------------------------------------------
    section("B. mel filterbank")
    mo = knf.MelBanksOptions()
    mo.num_bins = N_MELS
    mo.low_freq = 20
    mo.high_freq = 0
    mb = knf.MelBanks(mo, base_frame_opts(), 1.0)
    km = np.array(mb.get_matrix(), dtype=np.float64)
    print("  kaldi dim=%s  raw shape=%s  size=%d" % (mb.dim, km.shape, km.size))

    if km.ndim == 1 and km.size % int(mb.dim) == 0:
        km = km.reshape(int(mb.dim), km.size // int(mb.dim))
    print("  kaldi matrix shape = %s" % (km.shape,))
    print("  ours  matrix shape = %s" % (ext.mel_filters.shape,))

    if km.shape == ext.mel_filters.shape:
        cmp_arr("mel matrix", km, ext.mel_filters)
        d = np.abs(km - ext.mel_filters)
        per_bin = d.max(axis=1)
        worst = np.argsort(per_bin)[::-1][:5]
        print("  worst bins: %s" % [(int(i), round(float(per_bin[i]), 6))
                                    for i in worst])
        print("  kaldi nnz/bin[:10] = %s" % np.array2string(
            (km > 0).sum(axis=1)[:10], separator=","))
        print("  ours  nnz/bin[:10] = %s" % np.array2string(
            (ext.mel_filters > 0).sum(axis=1)[:10], separator=","))
        print("  kaldi rowsum[:5] = %s" % np.array2string(
            km.sum(axis=1)[:5], precision=6))
        print("  ours  rowsum[:5] = %s" % np.array2string(
            ext.mel_filters.sum(axis=1)[:5], precision=6))
        # 打印第 0 个 bin 的非零段，肉眼比对三角形状
        row = km[0]
        nz = np.nonzero(row)[0]
        if len(nz):
            print("  kaldi bin0 nonzero fft idx [%d..%d]: %s" %
                  (nz[0], nz[-1], np.array2string(row[nz[0]:nz[-1] + 1],
                                                  precision=5, max_line_width=200)))
        row = ext.mel_filters[0]
        nz = np.nonzero(row)[0]
        if len(nz):
            print("  ours  bin0 nonzero fft idx [%d..%d]: %s" %
                  (nz[0], nz[-1], np.array2string(row[nz[0]:nz[-1] + 1],
                                                  precision=5, max_line_width=200)))
    else:
        print("  SHAPE MISMATCH - cannot compare pointwise")

    # ----------------------------------------------------------------
    # C. mel 刻度
    # ----------------------------------------------------------------
    section("C. mel scale")
    for f in (0.0, 100.0, 1000.0, 4000.0, 8000.0):
        k = mb.mel_scale(f)
        ref = 1127.0 * np.log(1.0 + f / 700.0)
        print("  f=%-7.1f kaldi=%.6f  ref=%.6f  diff=%.3e" % (f, k, ref, abs(k - ref)))


if __name__ == "__main__":
    main()
