# -*- coding: utf-8 -*-
"""用 numpy 复现 kaldi MelBanks::InitKaldiMelBanks 的算法，并与
knf.MelBanks.get_matrix() 逐点比对，迭代到一致。

用法：
    python -I scripts/probe_mel_iterate.py
"""
import os
import sys

# 核心库都在 qt/ 下：根目录用来 import 入口脚本，qt/ 用来 import 核心库
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "qt")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import kaldi_native_fbank as knf

from console_utf8 import setup_stdout_utf8

SR = 16000
N_FFT = 512          # PaddedWindowSize
N_MELS = 80


def mel_scale(f, dtype=np.float64):
    """kaldi 的 Hz->Mel：1127 * ln(1 + f/700)（等价于 2595*log10(...)）。"""
    return 1127.0 * np.log(1.0 + np.asarray(f, dtype=dtype) / 700.0)


def kaldi_mel_banks(num_fft_bins, low_freq=20.0, high_freq=0.0):
    """照抄 kaldi InitKaldiMelBanks 的实现。"""
    nyquist = 0.5 * SR
    high = high_freq if high_freq > 0.0 else nyquist + high_freq
    fft_bin_width = SR / float(N_FFT)

    mel_low = mel_scale(low_freq)
    mel_high = mel_scale(high)
    delta = (mel_high - mel_low) / (N_MELS + 1)

    M = np.zeros((N_MELS, num_fft_bins), dtype=np.float64)
    for b in range(N_MELS):
        left = mel_low + b * delta
        center = mel_low + (b + 1) * delta
        right = mel_low + (b + 2) * delta
        for i in range(num_fft_bins):
            mel = mel_scale(fft_bin_width * i)
            if left < mel < right:
                if mel <= center:
                    M[b, i] = (mel - left) / (center - left)
                else:
                    M[b, i] = (right - mel) / (right - center)
    return M


def main():
    setup_stdout_utf8()   # Windows 控制台默认 GBK，不转一下中文输出就是乱码
    fo = knf.FrameExtractionOptions()
    fo.samp_freq = SR
    fo.frame_length_ms = 25.0
    fo.frame_shift_ms = 10.0

    for low, high in ((20.0, 0.0), (20.0, -400.0)):
        mo = knf.MelBanksOptions()
        mo.num_bins = N_MELS
        mo.low_freq = low
        mo.high_freq = high
        mb = knf.MelBanks(mo, fo, 1.0)
        km = np.array(mb.get_matrix(), dtype=np.float64).reshape(int(mb.dim), -1)
        print("=" * 72)
        print("kaldi ref: low_freq=%g high_freq=%g  shape=%s" % (low, high, km.shape))

        for nfb in (N_FFT // 2, N_FFT // 2 + 1):
            M = kaldi_mel_banks(nfb, low, high)
            if M.shape == km.shape:
                d = np.abs(M - km)
                print("  numpy num_fft_bins=%-4d shape OK   max|d|=%.6e  mean|d|=%.6e"
                      % (nfb, d.max(), d.mean()))
            else:
                print("  numpy num_fft_bins=%-4d shape %s != %s" % (nfb, M.shape, km.shape))

        # 看看 kaldi 尾列是否恒为 0（判断 257 列里最后一列是否有内容）
        print("  kaldi last col sum = %.6e   col 256 sum = %.6e"
              % (km[:, -1].sum(), km[:, -1].sum()))
        nz = np.nonzero(km[0])[0]
        print("  kaldi bin0 idx %s -> %s" % (nz.tolist()[:6],
                                             np.array2string(km[0][nz][:6], precision=6)))


if __name__ == "__main__":
    main()
