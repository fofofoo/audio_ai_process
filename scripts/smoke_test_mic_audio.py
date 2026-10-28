# -*- coding: utf-8 -*-
"""麦克风链路的端到端验证：把一条真实 wav 伪装成麦克风数据喂进去。

为什么需要这个：真开麦克风只能证明「链路通」，证明不了「识别对」——机器没法
对着麦克风念稿。这里把 sd.InputStream 换成一个按实时节奏回放 wav 的替身，
构造参数、回调签名、数据形状、送达节奏都与真麦克风一致，于是整条

    PortAudio 回调 -> audio_q -> on_tick -> push_hop / worker.feed -> 识别

都能被端到端跑一遍，而且结果可以与离线路径逐字对照。

关键点：替身伪造的是**最外层**（PortAudio 那一层），realtime_mic_monitor.py
自己的代码一行都不改、一行都不跳过。

不弹窗：QT_QPA_PLATFORM=offscreen。
用法：
    python -I scripts/smoke_test_mic_audio.py [样本序号]      # 默认 16
"""
import os
import sys
import threading
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"   # 必须在 import pyqtgraph 之前

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 核心库现在在 qt/ 下：根目录用来 import 入口脚本，qt/ 用来 import 核心库
for _p in (_ROOT, os.path.join(_ROOT, "qt")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

import realtime_mic_monitor as mic
from test_spectrogram import load_audio, load_split


class _FakeInputStream:
    """假的 sd.InputStream：按实时节奏把一段 wav 当麦克风数据喂给回调。

    只实现 main() 用到的那部分接口。回调的送达节奏（每 blocksize 样本一块、
    按 1 倍速实时推进）和数组形状都与真实 PortAudio 一致。
    """

    def __init__(self, wav, sr, **kwargs):
        self.wav = np.asarray(wav, dtype=np.float32)
        self.sr = int(kwargs.get("samplerate") or sr)
        self.blocksize = int(kwargs.get("blocksize") or 400)
        self.callback = kwargs["callback"]
        self._stop = threading.Event()
        self._th = None

    def __enter__(self):
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=3)
        return False

    def _run(self):
        # 先等主线程进事件循环再开喂：audio_q 只有 10 块容量，
        #   启动瞬间冲进去会把开头丢掉。真麦克风也有这个窗口，只是没这么尖。
        time.sleep(0.5)
        t0 = time.perf_counter()
        for k in range(0, len(self.wav), self.blocksize):
            if self._stop.is_set():
                return
            delay = (t0 + k / self.sr) - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            blk = self.wav[k:k + self.blocksize]
            # 与真麦克风同样的形状：(帧数, 声道数)，让回调里的 indata[:, 0] 成立
            self.callback(blk.reshape(-1, 1), len(blk), None, None)


def main():
    mic.setup_stdout_utf8()

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    df = load_split("test")
    row = df.iloc[idx]
    wav, sr = load_audio(row["audio_path"])
    text = row["text"]
    dur = len(wav) / sr
    print(f"[假麦克风] 音频 {dur:.2f}s  sr={sr}  参考: {text}")

    mic.LIST_DEVICES_ON_START = False
    # 只换掉最外层：真实的 PortAudio 流换成按 1 倍速回放 wav 的替身
    mic.sd.InputStream = lambda **kw: _FakeInputStream(wav, sr, **kw)

    app = pg.mkQApp()
    QtCore.QTimer.singleShot(int((dur + 4) * 1000), app.quit)

    mic.main()
    print("[假麦克风] 结束")
    return 0


if __name__ == "__main__":
    sys.exit(main())
