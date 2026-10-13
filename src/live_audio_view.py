# -*- coding: utf-8 -*-
"""实时音频可视化组件：波形 / RMS / 滚动语谱图。

把各块图的绘制逻辑封装成可复用的类，供两种数据源共用：
    - 离线回放（整段 wav，按时间轴主动推进）  —— 见 test_spectrogram.py
    - 实时采集（麦克风流，被动等数据到达）    —— 见 realtime_mic_monitor.py

设计要点
--------
1. 三个组件类（WaveformView / RmsView / SpectrogramView）都是
   「无源」的：只提供 push_* 这类被动方法，自己不持有数据源，
   谁有数据谁负责驱动。
2. 容器类 LiveAudioView 继承 pg.GraphicsLayoutWidget，负责装配布局、
   统一时间基准、横轴联动，并对外提供两个驱动入口：
       push_audio(wave_seg, spec_seg) —— 离线：一次给一整窗
       push_hop(hop)                  —— 实时：一次给一个小块，内部滚动拼接

依赖：numpy / scipy / pyqtgraph(PyQt5)，无新增第三方依赖。
运行环境：conda py310-gpu

典型参数：采样率 16000 Hz，FFT 窗长 512 点（= 32 ms），
    rfft 输出 257 个频率 bin（0 ~ 8000 Hz），
    帧步进 hop = 400 点（= 25 ms）→ 刷新率 40 fps。
    注意区分：512 点是【窗长】，25 ms 是【相邻两帧的间隔】，两者不冲突。
"""

# ============================================================
# 一、导入
# ============================================================
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from scipy.signal import stft, get_window

# ============================================================
# 二、pyqtgraph 全局显示选项
# ============================================================
# 白底黑字；setImage 的数组按 (行=频率, 列=时间) 理解
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")
pg.setConfigOption("imageAxisOrder", "row-major")


# ============================================================
# 三、工具函数
# ============================================================
def estimate_levels_from_wav(wav, sr, fft_size=512, p_lo=5.0, p_hi=99.5):
    """用整段音频估一组固定的语谱图色阶 (db_lo, db_hi)。

    离线回放时数据一次性全部可得，用整段的百分位定色阶，
    这样画面上不同时刻的颜色可以直接横向比较（不会「呼吸」）。

    参数:
        wav      —— 1D 波形数组
        sr       —— 采样率
        fft_size —— FFT 窗长，与实时语谱图保持一致
        p_lo/p_hi—— 取百分位的位置
    返回:
        (db_lo, db_hi)
    """
    win = get_window("hann", fft_size)
    _, _, Z = stft(wav, fs=sr, nperseg=fft_size, window=win)
    db = 20.0 * np.log10(np.abs(Z) + 1e-10)
    return float(np.percentile(db, p_lo)), float(np.percentile(db, p_hi))


# ============================================================
# 四、波形图组件
# ============================================================
class WaveformView:
    """实时滚动的波形图（局部放大）。

    职责很窄：把一段样本按显示增益画出来，以及按需切换「告警」配色。
    不参与 RMS 计算、不做阈值判定 —— 那是 RmsView 的事，避免出现两份口径。
    """

    def __init__(self,
                 title="实时波形 (局部放大)",
                 y_range=(-1.0, 1.0),
                 display_gain=1.0,
                 pen_normal=None,
                 pen_alert=None):
        self.display_gain = float(display_gain)
        # 用 dict 展开成关键字调 mkPen(**...)。
        #   注意不能写成 pg.mkPen(color, width)：这个版本的 mkPen 在收到多个
        #   位置参数时会把它们整体当作「一个颜色」传给 mkColor，而 mkColor 对
        #   长度 2 的元组会走 intColor(*args)，最后 int("b") 直接报 ValueError。
        self._pen_normal = pen_normal or {"color": "b", "width": 1.0}
        self._pen_alert = pen_alert or {"color": "r", "width": 1.5}

        self.plot = pg.PlotItem(title=title)
        self.plot.setYRange(*y_range)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot(pen=pg.mkPen(**self._pen_normal))
        self._alert_on = False

    def push(self, seg):
        """推一帧波形。seg 是原始样本（不含显示增益，这里内部乘）。"""
        seg = np.asarray(seg, dtype=np.float32) * self.display_gain
        # 不传 x：pyqtgraph 默认按样本序号 0,1,2,... 当横轴
        self.curve.setData(seg)

    def set_alert(self, on):
        """按 RMS 是否超阈值切换配色（超了变红）。

        状态没变就直接返回 —— 否则每帧都 setPen 会白白触发一次重绘。
        """
        on = bool(on)
        if on == self._alert_on:
            return
        self._alert_on = on
        pen = self._pen_alert if on else self._pen_normal
        self.curve.setPen(pg.mkPen(**pen))


# ============================================================
# 五、RMS 趋势图组件
# ============================================================
class RmsView:
    """实时滚动的 RMS 曲线（音量趋势）。

    横轴用「秒」，跨度与语谱图一致；由容器负责把 RMS 与语谱图
    两块的横轴联动起来，这样上下两块图能直接对照同一时刻。
    """

    def __init__(self,
                 history_len,
                 span_sec,
                 y_range=(0.0, 0.2),
                 threshold=None,
                 title="实时振幅 RMS (音量趋势)"):
        self.history = np.zeros(int(history_len), dtype=np.float32)
        # 每一格对应的横轴时刻（秒）：两端与语谱图画布严格对齐
        self.x = np.linspace(0.0, float(span_sec), len(self.history))

        self.plot = pg.PlotItem(title=title)
        self.plot.setLabel("left", "RMS")
        self.plot.setLabel("bottom", "Time [s] (最近)")
        # 横轴用秒：0 ~ span_sec。padding=0 让范围精确，才能和语谱图逐像素对齐
        self.plot.setXRange(0.0, float(span_sec), padding=0)
        self.plot.setYRange(*y_range)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        if threshold is not None:
            # 红色虚线：标出「有声音」的判定阈值
            self.plot.addItem(pg.InfiniteLine(
                pos=threshold, angle=0,
                pen=pg.mkPen("r", style=QtCore.Qt.DashLine, width=2)))
        self.curve = self.plot.plot(pen=pg.mkPen("g", width=2))

    def push(self, value):
        """推一个 RMS 值：队列整体左移，最新值放到队尾。"""
        self.history[:-1] = self.history[1:]
        self.history[-1] = value
        self.curve.setData(self.x, self.history)


# ============================================================
# 六、滚动语谱图组件
# ============================================================
class SpectrogramView:
    """固定画布的滚动语谱图。

    思路：维护一张 (频率 × 时间) 的 numpy 画布，每来一列就把画布整体
    左移一格、新列贴到右端，显示交给 pyqtgraph 的 ImageItem。

    色阶（颜色映射的 dB 上下界）支持两种来源：
        - 外部一次算好传进来     —— 离线：用整段音频的百分位（见 estimate_levels_from_wav）
        - 启动后收集若干列再锁定 —— 实时麦克风：没有「整段」可用
    """

    def __init__(self,
                 n_freq,
                 n_cols,
                 span_sec,
                 freq_max,
                 cmap="viridis",
                 level_lo=-100.0,
                 level_hi=-20.0,
                 calibrate_frames=0,
                 calibrate_range_db=40.0,
                 title="实时语谱图 (滚动)"):
        self.n_freq = int(n_freq)
        self.n_cols = int(n_cols)
        self.level_lo = float(level_lo)
        self.level_hi = float(level_hi)
        # 标定期内最响也只有这么点动态范围，就认定环境太安静、标定结果不可用
        self.calibrate_range_db = float(calibrate_range_db)

        # 固定画布：先整体填成静音底色
        self.canvas = np.full((self.n_freq, self.n_cols),
                              self.level_lo, dtype=np.float32)

        self.plot = pg.PlotItem(title=title)
        self.plot.setLabel("left", "Frequency [Hz]")
        self.plot.setLabel("bottom", "Time [s] (最近)")
        self.plot.setYRange(0, freq_max)
        # padding=0：让画布正好顶满左右边缘，范围精确等于 0 ~ span_sec
        self.plot.setXRange(0, span_sec, padding=0)
        self.plot.showGrid(x=True, y=True, alpha=0.3)

        self.image = pg.ImageItem()
        self.image.setColorMap(pg.colormap.get(cmap))
        self.image.setLevels([self.level_lo, self.level_hi])
        self.plot.addItem(self.image)

        # 顺序不能反：setRect 内部用 self.width()/height() 算缩放比，
        #   图像还没 setImage 时取不到尺寸（None），会算出一个巨大缩放把
        #   图像放到视野之外，表现就是整块纯色、「没数据」。
        self.image.setImage(self.canvas, autoLevels=False)
        # 把画布的像素坐标映射到 (时间秒, 频率 Hz)：列 -> x，频率 bin -> y
        self.image.setRect(QtCore.QRectF(0, 0, span_sec, freq_max))

        # 标定状态
        self._calibrate_left = int(calibrate_frames)
        self._calib_cols = []

    @property
    def calibrating(self):
        """是否还处在「收集样本、尚未锁定色阶」的阶段。"""
        return self._calibrate_left > 0

    def push_col(self, db_col):
        """推一列频谱（dB 值）：画布左移一格，新列贴到右端。"""
        db_col = np.asarray(db_col, dtype=np.float32)

        # 标定阶段：顺手把这一列收起来，收满再一次性算色阶
        if self._calibrate_left > 0:
            self._calib_cols.append(db_col.copy())
            self._calibrate_left -= 1
            if self._calibrate_left == 0:
                self._apply_calibration()

        self.canvas[:, :-1] = self.canvas[:, 1:]   # 画布整体左移一格
        self.canvas[:, -1] = db_col                # 最新一列贴在右端
        self.image.setImage(self.canvas, autoLevels=False)

    def _apply_calibration(self):
        """收集结束：用百分位锁定一组固定色阶（之后就固定不变）。"""
        db = np.concatenate(self._calib_cols)
        self._calib_cols = []
        lo, hi = np.percentile(db, [5.0, 99.5])
        if hi - lo < self.calibrate_range_db:
            # 标定期几乎没声音，色阶会窄得没法用，退回构造时给的默认值
            # 注意：库模块里的提示不要用 emoji —— 终端编码不是 UTF-8 时会直接
            #   抛 UnicodeEncodeError，把一次「标定失败」升级成整个程序崩溃。
            print(f"[标定] 声音过小（动态范围仅 {hi - lo:.1f} dB），"
                  f"沿用默认色阶 [{self.level_lo:.0f}, {self.level_hi:.0f}] dB")
            return
        self.set_levels(lo, hi)
        print(f"[标定] 语谱图色阶已锁定: [{lo:.1f}, {hi:.1f}] dB")

    def set_levels(self, lo, hi):
        """手动覆盖色阶（逃生口：环境换了、标定不合适时可以调）。"""
        self.level_lo, self.level_hi = float(lo), float(hi)
        self.image.setLevels([self.level_lo, self.level_hi])


# ============================================================
# 七、总容器：三块图 + 统一的驱动入口
# ============================================================
class LiveAudioView(pg.GraphicsLayoutWidget):
    """波形 / RMS / 滚动语谱图的监控窗口。

    时间基准只有一条：col_sec = hop_ms / 1000（每列代表多长时间）。
    span_sec = cols * col_sec 是参与联动的各块的横轴跨度，各块联动：
    任一块平移或缩放，其余各块立刻跟到完全相同的范围。

    两个驱动入口（数据源与组件解耦的关键）：
        push_audio(wave_seg, spec_seg) —— 离线回放：一次喂一整窗
        push_hop(hop)                  —— 实时采集：一次喂一个小块
    """

    def __init__(self,
                 sr=16000,
                 fft_size=512,
                 cols=200,
                 hop_ms=25.0,
                 wave_sec=0.1,
                 wave_display_gain=1.0,
                 rms_threshold=0.015,
                 rms_ymax=0.2,
                 spec_level=None,
                 spec_calibrate_frames=0,
                 cmap="viridis",
                 window="hann",
                 title="实时音频监控"):
        super().__init__(show=False, title=title)

        # ---- 派生量：所有时间/尺寸都从这几个数推出来 ----
        self.sr = int(sr)
        self.fft_size = int(fft_size)
        self.n_freq = self.fft_size // 2 + 1                # rfft 输出 bin 数
        self.hop = int(round(self.sr * hop_ms / 1000.0))    # 每帧推进的样本数
        self.fps = 1000.0 / float(hop_ms)                   # 界面刷新率
        self.col_sec = float(hop_ms) / 1000.0               # 每列代表的时间（秒）
        self.cols = int(cols)
        self.span_sec = self.cols * self.col_sec            # 滚动窗口时长（秒）
        self.wave_points = int(self.sr * wave_sec)          # 波形一屏显示的点数
        self.wave_display_gain = float(wave_display_gain)
        self.rms_threshold = float(rms_threshold)
        self.freq_max = self.sr / 2.0                       # Nyquist 频率

        # 统一用一份窗，保证波形与语谱图的频谱口径一致
        self.window = get_window(window, self.fft_size)
        self.win_sum = float(self.window.sum())

        # ---- 组装各块图 ----
        self.wave = WaveformView(display_gain=self.wave_display_gain,
                                 title="实时波形 (局部放大)")
        self.rms = RmsView(history_len=self.cols,
                           span_sec=self.span_sec,
                           y_range=(0.0, rms_ymax),
                           threshold=self.rms_threshold)
        lo, hi = spec_level if spec_level is not None else (-100.0, -20.0)
        self.spec = SpectrogramView(n_freq=self.n_freq,
                                    n_cols=self.cols,
                                    span_sec=self.span_sec,
                                    freq_max=self.freq_max,
                                    cmap=cmap,
                                    level_lo=lo,
                                    level_hi=hi,
                                    calibrate_frames=spec_calibrate_frames)

        # 自上而下依次排：波形 / RMS / 语谱图。
        #   用列表收集再按索引排行号，以后加块不用动前面的号。
        rows = [(self.wave.plot, 1), (self.rms.plot, 1), (self.spec.plot, 2)]
        for row, (plot, factor) in enumerate(rows):
            self.addItem(plot, row=row, col=0)
            # 高度比：语谱图给高一点，它才是要看结构的那块
            self.ci.layout.setRowStretchFactor(row, factor)

        # 参与联动的图共用同一条时间轴。
        #   这里不用 setXLink：它同步范围时会把值再套一层 ViewBox 的默认 padding，
        #   两侧会差出千分之几秒。手动同步直接传 padding=0，两边严格相等。
        #   值相同时 pyqtgraph 不发射信号，所以回调不会来回震荡。
        #   以语谱图为枢纽：其余各块的变化先汇到它，再由它广播出去，
        #   避免多块两两相连带来的回调爆炸。lambda 用默认参数绑定目标 ViewBox，
        #   否则闭包会捕获循环变量、全部指向最后一块。
        spec_vb = self.spec.plot.getViewBox()
        followers = [self.rms.plot.getViewBox()]
        for vb in followers:
            spec_vb.sigXRangeChanged.connect(
                lambda _vb, rng, d=vb: d.setXRange(rng[0], rng[1], padding=0))
            vb.sigXRangeChanged.connect(
                lambda _vb, rng, d=spec_vb: d.setXRange(rng[0], rng[1], padding=0))

        # 波形图刻意【不】联动：它显示的是最近 wave_sec 的局部放大，
        #   横轴是样本序号，和下面几块表示「滚动时间」的横轴不是一个语义。

        # ---- 内部滚动样本缓冲（实时模式用）----
        # 长度取波形窗与 FFT 窗的较大者，保证两个窗口都能从中截出来
        self._buf_len = max(self.wave_points, self.fft_size)
        self._buf = np.zeros(self._buf_len, dtype=np.float32)

    # --------------------------------------------------------
    # 驱动入口 1：离线回放，一次喂一整窗
    # --------------------------------------------------------
    def push_audio(self, wave_seg, spec_seg=None):
        """离线回放用：wave_seg 是当前一屏波形切片，spec_seg 是做 FFT 的切片。

        spec_seg 省略时，默认取 wave_seg 的末尾 fft_size 个点。
        """
        if spec_seg is None:
            spec_seg = np.asarray(wave_seg)[-self.fft_size:]
        self._push_wave_and_rms(wave_seg)
        self.spec.push_col(self._spectrum_db(spec_seg))

    # --------------------------------------------------------
    # 驱动入口 2：实时采集，一次喂一个小块
    # --------------------------------------------------------
    def push_hop(self, hop):
        """实时采集用：hop 是刚到的这一小块样本。

        小块被拼进内部滚动缓冲，波形与频谱都从缓冲里取窗口，
        所以驱动方（音频回调）不需要自己维护窗口。
        """
        hop = np.asarray(hop, dtype=np.float32).reshape(-1)
        n = min(len(hop), self._buf_len)
        if n == self._buf_len:
            self._buf[:] = hop[-n:]                 # 一块就填满整个缓冲
        elif n > 0:
            self._buf[:-n] = self._buf[n:]          # 缓冲整体左移，腾出尾部
            self._buf[-n:] = hop[-n:]               # 新数据接到尾部

        self._push_wave_and_rms(self._buf[-self.wave_points:])
        self.spec.push_col(self._spectrum_db(self._buf))

    # --------------------------------------------------------
    # 对其他组件的薄封装
    # --------------------------------------------------------
    def set_spec_levels(self, lo, hi):
        """手动覆盖语谱图色阶。"""
        self.spec.set_levels(lo, hi)

    # --------------------------------------------------------
    # 内部实现
    # --------------------------------------------------------
    def _push_wave_and_rms(self, wave_seg):
        """刷新波形与 RMS，并让波形颜色跟随 RMS 是否超阈值。"""
        seg = self._fit(np.asarray(wave_seg, dtype=np.float32), self.wave_points)
        self.wave.push(seg)

        # 注意：RMS 和波形用的是同一份数据，只是这里显式再乘一次显示增益。
        #   也就是说放大显示倍数会等比放大 RMS，等效于把阈值降低。
        rms = float(np.sqrt(np.mean((seg * self.wave_display_gain) ** 2)))
        self.rms.push(rms)
        self.wave.set_alert(rms > self.rms_threshold)
        return rms

    def _spectrum_db(self, seg):
        """对一段样本做加窗 FFT，返回 dB 频谱（长度 n_freq）。

        除以窗和，与 scipy.stft 的 scaling='spectrum' 口径一致。
        """
        seg = self._fit(np.asarray(seg, dtype=np.float32), self.fft_size)
        spec = np.abs(np.fft.rfft(seg * self.window)) / self.win_sum
        return 20.0 * np.log10(spec + 1e-10)

    @staticmethod
    def _fit(seg, n):
        """把 seg 调整到正好 n 个点：短了末尾补零，长了取末尾 n 个。"""
        if len(seg) < n:
            return np.pad(seg, (0, n - len(seg)), "constant")
        if len(seg) > n:
            return seg[-n:]
        return seg
