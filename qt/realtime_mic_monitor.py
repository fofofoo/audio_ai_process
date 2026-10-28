# -*- coding: utf-8 -*-
"""麦克风实时监控：波形 / RMS / 滚动语谱图 三块图 + 流式识别文字。

规格：16kHz 采样，512 点 FFT（= 32ms 窗），帧步进 25ms（400 点）→ 40fps，
      rfft 输出 257 个频率 bin，覆盖 0 ~ 8000 Hz。

线程模型（重点，一共三条线程）：

    PortAudio 回调线程 —— 只做「取第 0 通道 + copy + 塞队列」，绝不碰 Qt、绝不碰模型
    AsrWorker 识别线程 —— 从自己的队列取音频块喂 onnx，把结果写进 worker.text
    Qt 主线程          —— 40fps 定时取数据、绘图，并把同一块音频转交给识别线程

    回调 -> 主线程 走 audio_q（queue.Queue，自带锁）
    主线程 -> 识别 走 worker.feed()（内部也是 queue.Queue）
    识别 -> 主线程 走 worker.text（整串赋值，GIL 保证读不到半截）

    识别为什么要单独开线程：喂一次 encoder 最坏约 40ms，而画面每 25ms 就要推一帧，
    放主线程会让滚动明显一顿一顿。实测过对照（offscreen 下同一段音频），开着识别
    与纯可视化的帧间隔中位数 31.3ms vs 31.2ms、最坏 47.2ms vs 47.1ms —— 噪声级别。

运行环境：conda py310-gpu
    python -I qt/realtime_mic_monitor.py

关窗口即退出；退出时会把整段识别结果打印到终端。
"""

# ============================================================
# 一、导入
# ============================================================
import html
import queue
import sys
import time
from pathlib import Path

import pyqtgraph as pg
import sounddevice as sd
from pyqtgraph.Qt import QtCore

# 核心库与入口脚本都在 qt/ 下。正常 `python qt/realtime_mic_monitor.py` 时
#   脚本目录本来就在 sys.path 里；加 -I 时连脚本目录都不在。两种情况下都手动
#   把 qt/（即本文件所在目录）塞进去。
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from live_audio_view import LiveAudioView
from streaming_asr import AsrWorker, StreamingAsr, find_model_dir, setup_stdout_utf8

# ============================================================
# 二、设备与采集配置（设备不对就改这里）
# ============================================================
LIST_DEVICES_ON_START = True   # 启动时打印所有输入设备及其编号
DEVICE_ID = None               # 输入设备编号；None = 用系统默认设备
CHANNELS = 1                   # 麦克风阵列若是 2 in，可改成 2（代码只取第 0 通道）
SAMPLE_RATE = 16000
DTYPE = "float32"
BLOCK_SIZE = 400               # 每块样本数 = 一个 hop（25ms），保证 1 块 = 语谱图 1 列
QUEUE_CAP = 10                 # 队列最多缓存 10 块（约 250ms），超了丢最旧的

# ============================================================
# 三、视图参数
# ============================================================
FPS = 40                       # = 1000 / 25ms，与 BLOCK_SIZE 匹配
CANVAS_COLS = 200              # 语谱图列数 -> 200 * 25ms = 5 秒滚动窗口
WAVE_DISPLAY_GAIN = 1.0        # 波形显示增益：麦克风信号通常偏小，嫌波形太扁就调大
RMS_THRESHOLD = 0.015          # 判定「有声音」的 RMS 阈值，超过则波形变红
RMS_YMAX = 0.2                 # RMS 趋势图 Y 轴上界
CALIBRATE_SEC = 2.0            # 启动后先收集约 2 秒，再锁定语谱图色阶

# ============================================================
# 四、流式识别配置
# ============================================================
ENABLE_ASR = True              # 设 False 就退回纯三图监控
ASR_QUEUE_CAP = 256            # 识别线程积压上限（256 块 × 25ms ≈ 6.4 秒音频）
TAIL_CHARS = 40                # 界面上只滚动显示最近多少个字（全文退出时打到终端）


# ============================================================
# 五、工具函数
# ============================================================
def print_input_devices():
    """打印所有输入设备及编号，方便挑 DEVICE_ID。"""
    print("-" * 60)
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"[{i}] {dev['name']} | 输入通道: {dev['max_input_channels']}")
    print("-" * 60)


def _info_html(text):
    """底部标签内容：滚动显示最近若干个字 + 累计字数。

    麦克风是无限流，跑久了全文会很长，标签上只显示最近一段；全文在关窗口时
    一次性打到终端。注意标签里显示的「字」是模型输出的字符数，标点也算。
    """
    if not text:
        return "<div style='font-size:12pt; color:#aaa'>🎤 等待识别…</div>"
    trimmed = len(text) > TAIL_CHARS
    shown = text[-TAIL_CHARS:] if trimmed else text
    # 注意 CSS 里的 % 要写成 %%：这个字符串后面走 % 格式化，单个 % 会被吃掉
    return ("<div style='font-size:12pt; line-height:150%%; white-space:pre-wrap'>"
            "<span style='color:#888'>🎤 识别</span>　"
            "<span style='color:#0a6'>%s%s</span><br>"
            "<span style='color:#bbb; font-size:10pt'>"
            "累计 %d 字%s</span></div>"
            ) % ("… " if trimmed else "", html.escape(shown),
                 len(text), "（只显示最近 %d 字）" % TAIL_CHARS if trimmed else "")


# ============================================================
# 六、主流程
# ============================================================
def main():
    setup_stdout_utf8()
    if LIST_DEVICES_ON_START:
        print_input_devices()

    # ---- 1) 建窗口：三块图（波形 / RMS / 语谱图），底部留一行给识别文字 ----
    app = pg.mkQApp("麦克风实时监控")
    view = LiveAudioView(
        sr=SAMPLE_RATE,
        fft_size=512,
        cols=CANVAS_COLS,
        hop_ms=25.0,
        wave_sec=0.1,
        wave_display_gain=WAVE_DISPLAY_GAIN,
        rms_threshold=RMS_THRESHOLD,
        rms_ymax=RMS_YMAX,
        spec_calibrate_frames=int(CALIBRATE_SEC * FPS),   # 启动标定约 80 列
        title="麦克风实时监控 (16kHz / 512 点 FFT / 25ms 步进)",
    )
    view.resize(900, 800)
    view.show()

    info = view.ci.addLabel("", row=3, col=0, colspan=4, justify="left")
    view.ci.layout.setRowStretchFactor(3, 0)   # 不给额外高度，让标签按文字本身撑
    info.setText(_info_html(""))

    # ---- 2) 起识别线程 ----
    asr = None
    worker = None
    if ENABLE_ASR:
        model_dir = find_model_dir()
        print(f"⏳ 正在加载权重: {Path(model_dir).name}")
        t0 = time.perf_counter()
        asr = StreamingAsr(model_dir)
        print(f"🧠 就绪，耗时 {time.perf_counter() - t0:.2f}s  "
              f"(T={asr.T}, 每段 {asr.shift} 帧, context={asr.context_size})")
        worker = AsrWorker(asr, queue_cap=ASR_QUEUE_CAP)
        worker.start()          # 线程起来就开始等队列，之后只管 feed
    else:
        print("⚠️ ENABLE_ASR=False，本次只做三图监控、不识别")

    # ---- 3) 跨线程通道：回调线程写，UI 线程读 ----
    audio_q = queue.Queue(maxsize=QUEUE_CAP)

    def audio_callback(indata, frames, time_info, status):
        """PortAudio 回调线程：只取数据 + 塞队列，绝不碰 Qt，也绝不碰模型。"""
        if status:
            print(f"音频流状态警告: {status}")
        # 必须 copy：indata 的缓冲区由 PortAudio 复用，不复制会读到被覆盖的数据。
        # 只取第 0 通道，兼容单/双通道麦克风阵列。
        hop = indata[:, 0].copy()
        try:
            audio_q.put_nowait(hop)
        except queue.Full:
            # 队列满了说明 UI 跟不上：丢掉最旧的一块，保证低延迟、不越积越久
            try:
                audio_q.get_nowait()
            except queue.Empty:
                pass
            audio_q.put_nowait(hop)

    shown = {"text": None}     # 上一次已画到界面上的文本（用 dict 装，方便闭包里改）

    def on_tick():
        """UI 线程定时器：把本 tick 到达的所有块按顺序消费掉。"""
        while True:
            try:
                hop = audio_q.get_nowait()
            except queue.Empty:
                break
            # 每块各推一列：即使定时器抖动导致某 tick 到了 2 块，
            #   也会补成 2 列，时间轴不会被压缩、也不会丢列。
            view.push_hop(hop)
            # 同一块音频顺手转交给识别线程 —— 识别吃的和画面看的是同一份数据。
            #   feed 只是入队，不等它，所以这里不会卡住画面。
            if worker is not None:
                worker.feed(hop)

        if worker is not None:
            err = worker.error
            if err is not None:
                timer.stop()
                print(f"\n❌ 识别线程出错：{err!r}")
                return
            # 只在文字真变了才重设标签：LabelItem.setText 会重建富文本文档，
            #   40fps 无脑刷是白费性能。
            if worker.text != shown["text"]:
                shown["text"] = worker.text
                info.setText(_info_html(worker.text))

    timer = QtCore.QTimer()
    timer.timeout.connect(on_tick)
    timer.start(int(1000 / FPS))

    # ---- 4) 启动音频流，进入 Qt 事件循环 ----
    print("🎤 正在启动麦克风... 请说话或拍手！")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE,
                            channels=CHANNELS,
                            dtype=DTYPE,
                            blocksize=BLOCK_SIZE,
                            device=DEVICE_ID,
                            callback=audio_callback):
            app.exec()
    except Exception as e:
        print(f"\n[错误] 启动音频流失败: {e}")
        print("排查建议：")
        print("1. 检查 Win11 设置 -> 隐私和安全性 -> 麦克风 权限是否开启。")
        print("2. 检查是否有其他软件（如腾讯会议）独占了麦克风。")
        print("3. 核对上面打印的设备编号，修改 DEVICE_ID / CHANNELS。")

    # ---- 5) 收尾：音频已停，让识别补完尾帧再报最终结果 ----
    #   放在 with 之外，此时音频流已经关掉，不会再有新块进来。
    if worker is not None:
        print("\n正在补完尾帧...")
        if not worker.drain(timeout=10.0):
            print("⚠️ 识别线程还没跑完，下面的结果可能不完整")
        print("=" * 60)
        print(f"🎤 最终识别: {worker.text}")
        print(f"   共喂 {asr.n_chunks} 次 encoder、{asr.n_enc_frames} 帧、"
              f"{len(asr.token_ids)} 个 token；"
              f"音频 {asr.n_samples / asr.sample_rate:.1f}s")
        print("=" * 60)


if __name__ == "__main__":
    main()
