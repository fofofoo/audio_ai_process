# -*- coding: utf-8 -*-
"""test_spectrogram.py 的 offscreen 冒烟测试：静音跑一遍真实的播放路径。

验证四件事：
    1. 四块图 + 底部文字标签能搭起来（addLabel 的 row / colspan 不会把布局搞崩）
    2. 后台识别线程与主线程的队列协作正常（不出错、能正常收尾）
    3. 整条音频喂完后，识别文本与标注对得上（标点单独判）
    4. 开着识别时画面刷新的帧间隔，和不识别的纯可视化基本一样
       —— 这是「识别放后台线程」这个设计到底有没有用的唯一硬证据

不碰声卡：把 sd.play 换成空操作，其余流程一行不改。
不弹窗：QT_QPA_PLATFORM=offscreen，在 import pyqtgraph 之前设好。

用法（放项目根目录）：
    python -I scripts/smoke_test_spectrogram.py            # 带识别
    SMOKE_NO_ASR=1 python -I scripts/smoke_test_spectrogram.py   # 纯可视化，做对照
    SMOKE_WAV=BAC009S0770W0265 python -I scripts/smoke_test_spectrogram.py   # 挑指定样本
"""
import os
import sys
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

import test_spectrogram as ts


def main():
    ts.setup_stdout_utf8()

    # 静音：真正的播放被替换掉，之后走到的地方全是真实代码
    ts.sd.play = lambda *a, **k: None

    df = ts.load_split("test")
    # 默认用配置里的那一条；给 SMOKE_WAV 就按文件名挑（用来试带标点的样本）
    want = os.environ.get("SMOKE_WAV")
    if want:
        hit = df[df["audio_path"].str.contains(want, regex=False)]
        if hit.empty:
            sys.exit(f"test split 里没有匹配 {want!r} 的样本")
        row = hit.iloc[0]
    else:
        row = df.iloc[ts.SAMPLE_INDEX]
    wav, sr = ts.load_audio(row["audio_path"])
    text = row["text"]
    dur = len(wav) / sr
    print(f"[冒烟] 音频 {dur:.2f}s  sr={sr}  参考: {text}")

    # 给每次画面刷新打时间戳：update_plot 每轮开头都会调 view.push_audio，
    #   所以这串时间戳就是回调的真实节奏。
    stamps = []
    orig_push = ts.LiveAudioView.push_audio

    def timed_push(self, *args, **kwargs):
        stamps.append(time.perf_counter())
        return orig_push(self, *args, **kwargs)

    ts.LiveAudioView.push_audio = timed_push

    no_asr = os.environ.get("SMOKE_NO_ASR") == "1"
    asr = None
    if not no_asr:
        model_dir = ts.find_model_dir()
        asr = ts.StreamingAsr(model_dir)
        print(f"[冒烟] 权重就绪: {os.path.basename(model_dir)}")
    else:
        print("[冒烟] 纯可视化对照（不给 asr）")

    app = pg.mkQApp()
    # 音频在 offscreen 下仍按实时速度喂，多给 4 秒余量后强制退出
    QtCore.QTimer.singleShot(int((dur + 4) * 1000), app.quit)

    ts.play_with_monitor(wav, sr, text, asr=asr)
    print("[冒烟] 窗口已退出，布局与识别链路都没崩")

    # ---- 帧间隔统计：这是判断「线程有没有白开」的依据 ----
    hop = int(sr * ts.HOP_MS / 1000)
    expect = int(np.ceil(len(wav) / hop))
    print(f"[帧间隔] 共刷 {len(stamps)} 帧（音频 {dur:.1f}s ÷ {ts.HOP_MS}ms 应需约 {expect} 帧）")
    if len(stamps) > 2:
        gaps_ms = np.diff(np.asarray(stamps)) * 1e3
        # 只留刷帧的那一段：音频放完到兜底退出之间是个大空档，会带偏统计
        gaps_ms = gaps_ms[gaps_ms < 500]
        print(f"[帧间隔] 中位 {np.median(gaps_ms):.1f}ms  "
              f"P95 {np.percentile(gaps_ms, 95):.1f}ms  "
              f"最坏 {gaps_ms.max():.1f}ms  (目标 {ts.HOP_MS}ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
