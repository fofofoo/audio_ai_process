# -*- coding: utf-8 -*-
"""aishell1_subset 单条语音样本：播放 + 实时监控 + 流式识别（全部在 Qt 界面内完成）

流程：
    1. load_split()          从本地 CSV 读出样本清单（音频路径 + 文本）
    2. load_audio()          读出一条样本的 wav 波形
    3. play_with_monitor()   出声播放，同时用 LiveAudioView 实时显示
                             「波形 / RMS / 实时语谱图」三块图，
                             并把同一份音频边播边喂给流式 ASR，底部实时显示识别文字；
                             播放结束后窗口保持打开，关闭窗口即退出程序

三块图的绘制逻辑都在 src/live_audio_view.py 里，与麦克风脚本 realtime_mic_monitor.py 共用同一份代码。
识别逻辑全部来自 src/streaming_asr.StreamingAsr，与麦克风版是同一个类。

为什么识别要放单独线程：喂一次 encoder 最坏要约 40ms，而画面每 25ms 就要推一帧，
放在主线程会让滚动卡顿。所以主线程只负责「把当前 hop 塞进队列」和「读一下最新文本」，
真正的解码在后台线程里做。

运行环境：py310-gpu
    python test_spectrogram.py

依赖：numpy / pandas / scipy / sounddevice / pyqtgraph(PyQt5) / onnxruntime / soundfile
"""

# ============================================================
# 一、导入
# ============================================================
from pathlib import Path
import html
import re
import sys
import time
import wave

import numpy as np
import pandas as pd
import sounddevice as sd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

# 核心库都在 src/ 下。正常 `python test_spectrogram.py` 时脚本目录本来就在
#   sys.path 里，找不到 src/ 的子模块；加 -I 时连脚本目录都不在。两种情况下
#   都得手动把 src/ 塞进去。
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from live_audio_view import LiveAudioView, estimate_levels_from_wav
from streaming_asr import AsrWorker, StreamingAsr, find_model_dir, setup_stdout_utf8

# ============================================================
# 二、全局配置
# ============================================================
# --- 数据路径 ---
DATA_ROOT = Path(__file__).resolve().parent / "data"
CSV_DIR = DATA_ROOT / "aishell1_subset"

# --- 播放参数 ---
PLAY_GAIN = 3.0          # 播放增益：只放大送给声卡的那份数据，不改动原始波形

# --- 视图参数（三块图的时间口径统一由 HOP_MS 推导）---
HOP_MS = 25.0            # 帧步进：25ms -> 40fps；语谱图相邻列重叠 112 点（约 21.9%）
FFT_SIZE = 512           # FFT 窗长：512 @16kHz ≈ 32ms，rfft 输出 257 个频率 bin
CANVAS_COLS = 200        # 语谱图列数 -> 200 * 25ms = 5 秒滚动窗口
WAVE_SEC = 0.1           # 波形图一屏显示多长时间（秒）
WAVE_DISPLAY_GAIN = 5    # 波形图显示放大倍数（注意：算 RMS 用的是同一份数据，
                         #   所以 RMS 也会等比放大，等效于把阈值降到 1/5）
RMS_THRESHOLD = 0.015    # 判定「有声音」的 RMS 阈值，超过则波形变红
RMS_YMAX = 0.2           # RMS 趋势图 Y 轴上界

# --- 流式识别参数 ---
ENABLE_ASR = True        # 设 False 就退化成纯可视化（调界面时省掉加载权重那 1~2 秒）
ASR_QUEUE_CAP = 200      # 识别线程积压上限：200 块 × 25ms = 5 秒音频，
                         #   再积压就没意义了（实时识别要的是「现在」）
SAMPLE_INDEX = 16        # 用 test split 的第几条样本（0 基），换样本只需要改这个数


# ============================================================
# 三、数据加载层
# ============================================================
def load_split(name):
    """读取指定 split 的样本清单。

    参数:
        name: "train" 或 "test"
    返回:
        DataFrame，两列
            audio_path —— wav 文件的绝对路径
            text       —— 对应的文本标注
    """
    csv_path = CSV_DIR / f"aishell1_{name}_subset.csv"
    # 本地这份 CSV 是 UTF-8；ModelScope 仓库里那份是 GBK，直接用会报 UnicodeDecodeError
    df = pd.read_csv(csv_path, encoding="utf-8")
    # 原始列名是 "Audio:FILE" / "Text:LABEL"，改成好用的名字
    df = df.rename(columns={"Audio:FILE": "audio_path", "Text:LABEL": "text"})
    # CSV 里的路径是相对 data/ 的，转成绝对路径后面才能直接打开
    df["audio_path"] = df["audio_path"].map(lambda p: str(DATA_ROOT / p))
    return df[["audio_path", "text"]]


def load_audio(path):
    """读取 16bit PCM wav。

    参数:
        path: wav 文件路径
    返回:
        wav —— float32 numpy 数组，取值范围 [-1, 1]
        sr  —— 采样率（Hz）
    """
    with wave.open(str(path), "rb") as f:
        sr = f.getframerate()
        sampwidth = f.getsampwidth()
        data = f.readframes(f.getnframes())
    if sampwidth != 2:
        raise ValueError(f"仅支持 16bit PCM，实际 sampwidth={sampwidth}: {path}")
    # int16 原始字节 -> float32，并归一化到 [-1, 1]
    wav = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return wav, sr


# ============================================================
# 四、播放 + 实时监控
# ============================================================
def _strip_punct(s):
    """去掉标点和空白，只留字词（\\w 覆盖汉字、字母、数字）。"""
    return re.sub(r"[\W_]+", "", s)


def _verdict(hyp, ref):
    """给出「识别 vs 标注」的判定语。

    不能只比字面：这个权重是 punct 版，会自己补「，。？」，而 aishell1 的标注是
    纯汉字、一个标点都没有。所以「忽略标点后一致」才是有意义的判据。
    """
    if hyp == ref:
        return "与参考：逐字完全一致"
    if _strip_punct(hyp) == _strip_punct(ref):
        return "与参考：忽略标点后一致（标点是模型自己加的，标注里没有）"
    return "与参考：有出入（忽略标点后仍不一致）"


def _info_html(ref, hyp=""):
    """底部标签的内容：上一行参考文本，下一行实时识别结果。

    返回的是 HTML 片段（pyqtgraph 的 LabelItem 走 QLabel 富文本）。
    """
    shown = (html.escape(hyp) if hyp
             else "<span style='color:#aaa'>（等待出字…）</span>")
    return ("<div style='font-size:11pt; line-height:160%%'>"
            "<span style='color:#888'>参考</span>　%s<br>"
            "<span style='color:#888'>识别</span>　"
            "<span style='color:#0a6'>%s</span>"
            "</div>") % (html.escape(ref), shown)


def _finish_asr(worker, info, ref_text):
    """播放结束后收尾：让识别线程补完尾帧，再把最终文本打印出来对照。"""
    if worker is None:
        return
    # drain 会先送哨兵（补尾帧）、再等线程结束。队列里的块消化极快
    #   （整条链 RTF 约 0.07），这里几乎不会真等到超时。
    if not worker.drain(timeout=10.0):
        print("⚠️ 识别线程还没跑完，最终文本可能不完整")
    if worker.error is not None:          # 补尾帧时也可能出错，这里再兜一次
        print(f"❌ 识别线程出错：{worker.error!r}")
        return
    info.setText(_info_html(ref_text, worker.text))
    print(f"📝 参考: {ref_text}")
    print(f"🎤 识别: {worker.text}")
    print(f"📊 {_verdict(worker.text, ref_text)}")
    print(f"   （识别共喂 {worker.asr.n_chunks} 次 encoder、"
          f"{worker.asr.n_enc_frames} 帧、{len(worker.asr.token_ids)} 个 token）")


def play_with_monitor(wav, sr, text, asr=None):
    """出声播放音频，同时实时滚动显示三块图，并流式识别、底部显示文字。

    参数:
        wav  —— 波形数组（原始，不会被改动）
        sr   —— 采样率
        text —— 文本标注，显示在窗口标题和底部标签上
        asr  —— StreamingAsr 实例；给 None 则只做可视化、不做识别
    """
    # ---- 1) 建窗口 ----
    app = pg.mkQApp("本地音频实时回放可视化")
    view = LiveAudioView(
        sr=sr,
        fft_size=FFT_SIZE,
        cols=CANVAS_COLS,
        hop_ms=HOP_MS,
        wave_sec=WAVE_SEC,
        wave_display_gain=WAVE_DISPLAY_GAIN,
        rms_threshold=RMS_THRESHOLD,
        rms_ymax=RMS_YMAX,
        # 离线数据一次性全部可得：直接用整段音频的百分位把语谱图色阶一次定死，
        #   这样画面上不同时刻的颜色可以直接横向比较。
        spec_level=estimate_levels_from_wav(wav, sr, fft_size=FFT_SIZE),
        title=f"实时回放监控: {text}",
    )
    view.resize(900, 800)
    view.show()

    # 底部加一行文字，挂在视图 layout 的第 4 行（前三行是波形 / RMS / 语谱图）
    info = view.ci.addLabel("", row=3, col=0, colspan=4, justify="left")
    view.ci.layout.setRowStretchFactor(3, 0)   # 不给额外高度，让标签按文字本身撑
    info.setText(_info_html(text))

    # ---- 2) 起识别线程 ----
    worker = AsrWorker(asr, queue_cap=ASR_QUEUE_CAP) if asr is not None else None
    if worker is not None:
        worker.start()                    # 线程起来就开始等队列，之后只管 feed
        print("🧠 流式识别已就绪（界面上的字会比声音晚约 0.5 秒，属正常）")

    # ---- 3) 起播 ----
    print("🔊 正在播放，请观察实时滚动效果...")
    # 固定增益放大音量，再 clip 到 ±1：超过量程会被声卡硬截断，产生削顶失真
    play_wav = np.clip(wav * PLAY_GAIN, -1.0, 1.0)
    clipped = float(np.mean(np.abs(wav) * PLAY_GAIN > 1.0)) * 100
    if clipped > 0:
        print(f"⚠️ 约 {clipped:.2f}% 的样本被削顶，若听感发破可调小 PLAY_GAIN")
    sd.play(play_wav, sr)                # 非阻塞播放，声音在后台持续

    # ---- 4) 定时器逐帧推进 ----
    state = {"idx": 0}   # 当前播放到的样本索引（用 dict 装，方便闭包里修改）

    def update_plot():
        """定时器回调：切片 -> 推给视图 / 喂给识别 -> 推进播放位置。

        每帧推进 hop 个样本、耗时 1/fps 秒，二者相等，所以画面与声音 1:1 同步。
        """
        i = state["idx"]

        # 已经放到末尾 —— 停定时器；窗口保持打开，关掉窗口即退出程序
        if i >= len(wav):
            timer.stop()
            print("\n✅ 播放结束")
            _finish_asr(worker, info, text)
            print("（界面保持显示，关闭窗口退出）")
            return

        # 波形取最近一屏，频谱取最近一个 FFT 窗；
        #   末段不足时由视图内部补零，这里不用自己处理。
        wave_seg = wav[i:i + view.wave_points]
        spec_seg = wav[i:i + view.fft_size]
        view.push_audio(wave_seg, spec_seg)

        if worker is not None:
            # 识别吃的是「这一步新到的 hop」，不是上面两个窗口：
            #   一个 hop 一块顺序推进，正好等于批量解码的分块口径。
            #   解码在后台线程做，这里只入队，不等它。
            worker.feed(wav[i:i + view.hop])

            err = worker.error
            if err is not None:
                timer.stop()
                print(f"\n❌ 识别线程出错：{err!r}")
                return

            # 只在文字真变了才重设标签：LabelItem.setText 会重建富文本文档，
            #   40fps 无脑刷是白费性能。
            if worker.text != state.get("shown"):
                state["shown"] = worker.text
                info.setText(_info_html(text, worker.text))

        state["idx"] = i + view.hop

    timer = QtCore.QTimer()
    timer.timeout.connect(update_plot)
    timer.start(int(1000 / view.fps))
    app.exec()                           # 阻塞在这里，直到用户关闭窗口


# ============================================================
# 五、主流程
# ============================================================
if __name__ == "__main__":
    setup_stdout_utf8()   # Windows 控制台默认 GBK，不转一下 emoji 会直接报错

    # ---- 1) 读取数据 ----
    df = load_split("test")
    row = df.iloc[SAMPLE_INDEX]
    wav, sr = load_audio(row["audio_path"])
    text = row["text"]

    print("=" * 50)
    print(f"🎵 音频: {Path(row['audio_path']).name}")
    print(f"📝 文本: {text}")
    print(f"📊 采样率: {sr} Hz, 总时长: {len(wav) / sr:.2f} 秒")
    print("=" * 50)

    # ---- 2) 加载权重：耗时 1~2 秒，放在起播前一次性做完 ----
    asr = None
    if ENABLE_ASR:
        model_dir = find_model_dir()
        print(f"⏳ 正在加载权重: {Path(model_dir).name}")
        t0 = time.perf_counter()
        asr = StreamingAsr(model_dir)
        print(f"🧠 就绪，耗时 {time.perf_counter() - t0:.2f}s  "
              f"(T={asr.T}, 每段 {asr.shift} 帧, context={asr.context_size})")
    else:
        print("⚠️ ENABLE_ASR=False，本次只做可视化、不识别")

    # ---- 3) 播放 + 实时监控 + 流式识别，全部在一个 Qt 窗口里 ----
    play_with_monitor(wav, sr, text, asr=asr)

    # 播放结束后窗口保持打开，手动关闭窗口即退出
