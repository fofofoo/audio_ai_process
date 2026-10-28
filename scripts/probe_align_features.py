# -*- coding: utf-8 -*-
"""特征对齐验证：把 traditional_features.py 的输出与 kaldi_native_fbank
逐点比对，确认改完之后是否真的等价。

不加载 onnx，跑得很快。控制台编码由 console_utf8 统一设置。

用法：
    python -I scripts/probe_align_features.py <wav> [<wav> ...]
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
import soundfile as sf

from console_utf8 import setup_stdout_utf8
from traditional_features import TraditionalFeatureExtractor

SR = 16000
N_MELS = 80


def knf_ref(samples, sr, low=20.0, high=0.0):
    """kaldi_native_fbank 参考特征（与 test_onnx.py 同配置）。"""
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.window_type = "povey"
    opts.frame_opts.samp_freq = sr
    opts.mel_opts.num_bins = N_MELS
    opts.mel_opts.low_freq = low
    opts.mel_opts.high_freq = high
    f = knf.OnlineFbank(opts)
    f.accept_waveform(sr, samples.tolist())
    f.input_finished()
    return np.stack([f.get_frame(i) for i in range(f.num_frames_ready)])


def report(tag, ref, ours):
    if ref.shape != ours.shape:
        print("  %-24s SHAPE MISMATCH kaldi=%s ours=%s" %
              (tag, ref.shape, ours.shape))
        return
    d = np.abs(ref - ours)
    fi, bi = np.unravel_index(int(np.argmax(d)), d.shape)
    print("  %-24s max|d|=%.6e @(frame %d/%d, bin %d)  p99|d|=%.6e  mean|d|=%.6e" %
          (tag, d.max(), fi, ref.shape[0], bi,
           np.percentile(d, 99), d.mean()))


def main():
    setup_stdout_utf8()   # Windows 控制台默认 GBK，不转一下中文输出就是乱码
    wavs = sys.argv[1:]
    for wav in wavs:
        samples, sr = sf.read(wav, dtype="float32", always_2d=True)
        samples = np.ascontiguousarray(samples[:, 0])
        print("=" * 76)
        print("%s  sr=%d n=%d dur=%.3fs" %
              (os.path.basename(wav), sr, len(samples), len(samples) / sr))

        for low, high in ((20.0, 0.0), (20.0, -400.0)):
            ref = knf_ref(samples, sr, low, high)
            print("  --- low_freq=%g high_freq=%g  kaldi shape=%s  range=[%.4f, %.4f]" %
                  (low, high, ref.shape, ref.min(), ref.max()))

            ext = TraditionalFeatureExtractor(sample_rate=sr, low_freq=low,
                                              high_freq=high)
            print("      ours frame_size=%d step=%d  n_frames=%d" %
                  (ext.frame_size, ext.frame_step, ext.num_frames(len(samples))))

            # 1) 走完整工作流
            report("logfbank() [reflect]", ref, ext.logfbank(samples))

            # 2) 手动指定边界模式（frame_signal 的 boundary 参数）
            for boundary in ("reflect", "symmetric"):
                frames = ext.process_window(ext.frame_signal(samples, boundary=boundary))
                ours = ext.log_mel_spectrogram(ext.power_spectrum(frames))
                report("manual [%s]" % boundary, ref, ours)

            # 3) 关掉逐帧预加重的教科书写法，看差多少
            ext2 = TraditionalFeatureExtractor(sample_rate=sr, low_freq=low,
                                               high_freq=high,
                                               frame_preemphasis=False)
            report("textbook preemph", ref, ext2.logfbank(samples))

            # 4) 只切帧不加任何处理，隔离「分帧+反射」这一层
            opts2 = knf.FbankOptions()
            opts2.frame_opts.dither = 0
            opts2.frame_opts.snip_edges = False
            opts2.frame_opts.window_type = "rectangular"
            opts2.frame_opts.remove_dc_offset = False
            opts2.frame_opts.preemph_coeff = 0.0
            opts2.frame_opts.samp_freq = sr
            opts2.mel_opts.num_bins = N_MELS
            opts2.mel_opts.low_freq = low
            opts2.mel_opts.high_freq = high
            opts2.use_log_fbank = False
            f2 = knf.OnlineFbank(opts2)
            f2.accept_waveform(sr, samples.tolist())
            f2.input_finished()
            raw_ref = np.stack([f2.get_frame(i) for i in range(f2.num_frames_ready)])

            ext3 = TraditionalFeatureExtractor(sample_rate=sr, low_freq=low,
                                               high_freq=high,
                                               window="rectangular",
                                               remove_dc_offset=False,
                                               preemph=0.0,
                                               log_eps=None)
            for boundary in ("reflect", "symmetric"):
                frames = ext3.frame_signal(samples, boundary=boundary)
                # 不取对数：直接线性 mel 能量
                mel = ext3.power_spectrum(frames) @ ext3.mel_filters.T
                report("raw mel [%s]" % boundary, raw_ref, mel)


if __name__ == "__main__":
    main()
