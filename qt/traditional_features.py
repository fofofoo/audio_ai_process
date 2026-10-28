# -*- coding: utf-8 -*-
"""传统语音前端特征提取（Frontend Feature Extraction）。

把经典的声学特征流水线封装成一个类，两条流程共用同一套步骤：

    LogFBank：预加重 -> 分帧 -> 加窗 -> FFT -> 功率谱 -> Mel 滤波 -> 对数压缩
    MFCC：    预加重 -> 分帧 -> 加窗 -> FFT -> 功率谱 -> Mel 滤波 -> 对数压缩
              -> DCT(取前 K 个系数) -> (可选) 动态特征

两者前 7 步完全相同，区别只在 LogFBank 停在取对数、MFCC 再多做一步 DCT。

============================================================
一、默认口径：对齐 kaldi_native_fbank（sherpa-onnx / icefall 前端）
============================================================

默认参数直接对齐 kaldi 的语音前端，目的是把本模块产出的特征喂给 sherpa-onnx
系模型（如流式 zipformer2 transducer）时能与 kaldi_native_fbank 逐点吻合：

    snip_edges=False        帧数 = round(N/S)，帧中心钉在 t·S 上
    boundary="symmetric"    帧两端越界处镜像补齐，且重复边缘样本
    window="povey"          (0.5 - 0.5·cos(2πn/(N-1)))^0.85
    frame_preemphasis=True  预加重在分帧之后逐帧做，帧首样本乘 (1-α)
    remove_dc_offset=True   每帧先减去自己的均值
    dither=0                推理不加抖（kaldi 默认 1.0，随机、不可复现）
    low_freq / high_freq    20Hz ~ Nyquist 的 Mel 滤波器组
    log_eps=None            log 前把能量下限抬到 float32 epsilon

实测（对比 kaldi_native_fbank.OnlineFbank，4 条 16kHz 测试音频）：
    平均绝对偏差约 5e-6，即相对误差约 5e-7 —— 残差已是 float32 精度量级。
    反例：边界用 "reflect" 会差到 1e-4，整段预加重会差到 1e-2。

============================================================
二、方法一览
============================================================

教科书写法（步骤逐个对应，步骤之间互不依赖，可单独调用）：
    pre_emphasis        —— 预加重（对整段波形做一次）
    frame_and_window    —— 分帧 + 加窗（snip_edges=True 语义）
    power_spectrum      —— FFT + 取模平方（短时功率谱 / 周期图）
    log_mel_spectrogram —— Mel 滤波 + 对数压缩（输出即 LogFBank）
    dct                 —— DCT + 倒谱提升（输出 MFCC 静态系数）
    dynamic_features    —— 一阶差分 Δ + 二阶差分 ΔΔ
    normalize           —— 逐维全局归一化（CMVN）

kaldi 口径的等价方法（默认工作流走的是这一套，与上面几个并存）：
    num_frames          —— kaldi NumFrames：一共能出多少帧
    frame_signal        —— kaldi FirstSampleOfFrame：怎么切、越界怎么补
    process_window      —— kaldi ProcessWindow：逐帧 加抖/去直流/预加重/加窗
    frontend_for_onnx   —— 直通 onnx：输出 float32 并做尾部静音补齐

默认参数按 16kHz 语音：窗长 25ms(400 点)、帧移 10ms(160 点)、
    FFT 512 点 -> 257 个频率 bin（0 ~ 8000Hz，分辨率 31.25 Hz/bin）、
    Mel 滤波器 80 个（深度学习前端主流；传统 GMM-HMM 常用 40）。

依赖：numpy / scipy，无新增第三方依赖。
运行环境：conda py310-gpu
"""

# ============================================================
# 一、导入
# ============================================================
import wave

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from scipy.fft import dct as _dct_transform
from scipy.signal import get_window


# ============================================================
# 二、wav 读取工具
# ============================================================
def read_wav(path):
    """读取 16bit PCM wav，返回 (float64 波形[-1,1], 采样率)。

    与 test_spectrogram.py 的读法保持一致：
    用标准库 wave，避免引入 soundfile / librosa / torchcodec 依赖。
    只支持 16bit PCM；多声道会被混成单声道。
    """
    with wave.open(str(path), "rb") as f:
        sr = f.getframerate()
        sampwidth = f.getsampwidth()
        n_channels = f.getnchannels()
        data = f.readframes(f.getnframes())
    if sampwidth != 2:
        raise ValueError(f"仅支持 16bit PCM，实际 sampwidth={sampwidth}: {path}")
    # int16 原始字节 -> float64，并归一化到 [-1, 1]
    wav = np.frombuffer(data, dtype=np.int16).astype(np.float64) / 32768.0
    if n_channels > 1:
        # 多声道是交错排列的；不reshape 的话长度会翻倍、内容全错。
        #   这里直接取各声道平均，混成单声道。
        wav = wav.reshape(-1, n_channels).mean(axis=1)
    return wav, sr


# kaldi 在给 mel 能量取对数前，会先把它的下限抬到 float32 的 epsilon：
#   1.1920929e-07 -> log = -15.9424。这个下限直接决定静音帧的取值地板。
#   注意和「加 eps 再取对数」是两回事：加 1e-10 会得到 log = -23.03。
FLOAT32_EPS = float(np.finfo(np.float32).eps)


# ============================================================
# 三、Mel 刻度换算（模块级纯函数，滤波器组与可视化都用得上）
# ============================================================
def hz_to_mel(hz):
    """赫兹 -> Mel 刻度。低频变化快、高频变慢，模拟人耳的非线性感知。"""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel):
    """Mel 刻度 -> 赫兹。"""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def hz_to_mel_kaldi(hz):
    """赫兹 -> Mel 刻度（kaldi / HTK 写法）：1127·ln(1 + f/700)。

    与上面的 hz_to_mel 数学上完全等价（2595 / ln(10) = 1127.01…）。之所以
    还留一份，是因为 kaldi 用的是 1127 这个取整常数、且全程 float32 运算，
    要想逐点对齐 kaldi 的滤波器组，就得照抄它的写法而不是等价的另一种。
    """
    return 1127.0 * np.log(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz_kaldi(mel):
    """Mel 刻度 -> 赫兹（kaldi / HTK 写法）：700·(exp(m/1127) - 1)。"""
    return 700.0 * (np.exp(np.asarray(mel, dtype=np.float64) / 1127.0) - 1.0)


# ============================================================
# 四、传统特征提取器
# ============================================================
class TraditionalFeatureExtractor:
    """传统语音前端特征提取器：从原始波形算出 LogFBank 或 MFCC。

    用法::

        ext = TraditionalFeatureExtractor(sample_rate=16000)
        fb = ext.logfbank(wav)          # (n_frames, 80)
        mc = ext.mfcc(wav)              # (n_frames, 39) = 13 静态 + 13Δ + 13ΔΔ

    也可以只调中间某一步（各步骤的输入输出都是 numpy 数组，互不依赖）::

        x = ext.pre_emphasis(wav)       # 预加重后的波形
        frames = ext.frame_and_window(x)  # 加窗后的帧矩阵

    所有步骤方法都接收/返回 float64 数组，不修改传入的数组。
    """

    def __init__(self,
                 sample_rate=16000,
                 frame_len=0.025,
                 frame_shift=0.01,
                 n_fft=512,
                 n_mels=80,
                 n_mfcc=13,
                 preemph=0.97,
                 window="povey",
                 lifter=22,
                 log_eps=None,
                 low_freq=20.0,
                 high_freq=0.0,
                 snip_edges=False,
                 remove_dc_offset=True,
                 dither=0.0,
                 frame_preemphasis=True,
                 boundary="symmetric"):
        """
        参数:
            sample_rate —— 采样率（Hz）
            frame_len   —— 帧长（秒）。25ms 是业界甜点值：够长能拿到可靠的
                           能谱估计，够短不会把 10~30ms 的爆破音糊掉
            frame_shift —— 帧移（秒）。10ms 保证时间轴上的平滑连续
            n_fft       —— FFT 点数，取 >= 帧长的 2 的幂（512 @16kHz -> 257 bin）
            n_mels      —— Mel 滤波器个数，也就是 LogFBank 的维度
            n_mfcc      —— 保留的 DCT 系数个数，也就是 MFCC 静态维度
            preemph     —— 预加重系数，通常 0.95 ~ 0.97
            window      —— 窗类型，见 _build_window 支持的列表
            lifter      —— 倒谱提升参数；None 表示不做提升
            log_eps     —— 取对数前的能量下限。
                           None（默认）走 kaldi 口径：log(max(energy, float32_eps))；
                           给数值则走旧口径 log(energy + log_eps)。
            low_freq    —— Mel 滤波器组下限（Hz），kaldi 默认 20
            high_freq   —— Mel 滤波器组上限；0 = Nyquist，负数 = Nyquist + 该值
                           （kaldi 的 -400 即 7600Hz）
            snip_edges  —— False（默认，kaldi 口径）：帧数 = round(N/S)，
                           帧中心落在 t·S 上，两端用镜像反射补齐；
                           True：帧数 = (N-W)//S+1，只取完整落在波形内的帧
            remove_dc_offset —— 是否逐帧减去均值（kaldi 默认开）
            dither      —— 加抖幅度（高斯噪声标准差，直接作用于输入量纲）。
                           kaldi_native_fbank 的默认值是 2.9999999e-05
                           （float32 的 3e-5，该值假设输入是 16-bit 整数域）；
                           模型推理一律用 0，非 0 时结果不可复现（随机数）
            frame_preemphasis —— True（默认，kaldi 口径）：预加重在「分帧之后
                           逐帧」做，帧首样本按 (1-α) 缩放；
                           False：先对整段波形做一次预加重再分帧（教科书写法）
            boundary    —— snip_edges=False 时帧越界的补齐方式。
                           "symmetric"（默认，kaldi 口径）：镜像时**重复**边缘样本；
                           "reflect"：镜像时不重复边缘样本。
                           实测 kaldi 用 symmetric（reflect 会差到 6e-1）

        默认值一律对齐 kaldi_native_fbank（sherpa-onnx / icefall 的语音前端口径）。
        """
        self.sample_rate = int(sample_rate)
        self.frame_len = float(frame_len)
        self.frame_shift = float(frame_shift)
        self.n_fft = int(n_fft)
        self.n_mels = int(n_mels)
        self.n_mfcc = int(n_mfcc)
        self.preemph = float(preemph)
        self.window = window
        self.lifter = lifter
        self.log_eps = None if log_eps is None else float(log_eps)
        self.low_freq = float(low_freq)
        self.high_freq = float(high_freq)
        self.snip_edges = bool(snip_edges)
        self.remove_dc_offset = bool(remove_dc_offset)
        self.dither = float(dither)
        self.frame_preemphasis = bool(frame_preemphasis)
        self.boundary = boundary

        # ---- 一次性算好、之后反复复用的量 ----
        # kaldi FrameExtractionOptions::WindowSize() 是「截断」而非四舍五入
        self.frame_size = int(self.sample_rate * 0.001 * self.frame_len * 1000.0)
        self.frame_step = int(self.sample_rate * 0.001 * self.frame_shift * 1000.0)
        self.n_freq = self.n_fft // 2 + 1                                  # rfft 输出 bin 数
        self._win = self._build_window(self.frame_size)
        self.mel_filters = self._build_mel_filterbank()

        # 动态特征的分母：2 * Σ_{k=1..K} k²，与 K 有关，用到时再算

    # --------------------------------------------------------
    # 步骤 1：预加重
    # --------------------------------------------------------
    def pre_emphasis(self, signal, alpha=None):
        """预加重：y[n] = x[n] - α·x[n-1]，一个一阶高通滤波器。

        作用：
            - 语音的频谱总体按 -6dB/倍频程衰减，预加重把高频抬回来，
              让各频段的幅度量级更均衡；
            - 补偿声带和嘴唇辐射对高频的抑制，突出高频共振峰；
            - 改善信噪比，也避免 FFT 时因动态范围过大产生数值问题。

        参数:
            signal —— 原始波形 (n_samples,)
            alpha  —— 预加重系数；None 则用构造时的 self.preemph
        返回:
            预加重后的波形，长度与输入相同（首点直接复制，不引入直流跳变）
        """
        signal = np.asarray(signal, dtype=np.float64)
        # 如果 alpha 是 None，就用构造函数里设定的默认值 self.preemph；否则把 alpha 转成 float
        alpha = self.preemph if alpha is None else float(alpha)
        # 首点没有前一点可用，直接照抄
        return np.concatenate((signal[:1], signal[1:] - alpha * signal[:-1]))

    # --------------------------------------------------------
    # 步骤 2：分帧 + 加窗
    # --------------------------------------------------------
    def frame_and_window(self, signal):
        """分帧并加窗，返回 (n_frames, frame_size) 的帧矩阵。

        为什么要分帧：语音是短时平稳信号 —— 整体非平稳，但在 20~40ms
        的短窗口内频谱近似不变。直接在整段上做 FFT 会丢掉频率随时间的
        变化，所以切成短帧、逐帧做 FFT，再把相邻帧接起来近似频率轮廓。

        为什么要加窗：FFT 隐含假设输入是首尾相接的周期信号。直接对截断的
        帧做 FFT，两端的不连续会造成频谱泄漏（主频两侧拖出虚假谱线）。
        乘窗函数让两端平滑归零，泄漏显著减小。

        实现上用 sliding_window_view 取滑动窗口视图，不真正复制数据，
        比 np.tile 拼索引矩阵省一大块内存。

        参数:
            signal —— 波形 (n_samples,)
        返回:
            加窗后的帧矩阵 (n_frames, frame_size)
        """
        signal = np.asarray(signal, dtype=np.float64)

        # 太短的信号先补零，保证至少能出一帧
        if len(signal) < self.frame_size:
            signal = np.pad(signal, (0, self.frame_size - len(signal)))

        # 帧数 = (n_samples - frame_size) // frame_step + 1，与下式等价
        # sliding_window_view造出所有滑动窗口 → [::self.frame_step]按帧移抽样 →  * self._win加窗。
        frames = sliding_window_view(signal, self.frame_size)[::self.frame_step]
        return frames * self._win

    # --------------------------------------------------------
    # 步骤 2b：kaldi 语义的分帧与逐帧处理（对齐 kaldi_native_fbank）
    #          frame_and_window 是「教科书写法」，下面三个是 kaldi 写法
    # --------------------------------------------------------
    def num_frames(self, n_samples):
        """按 kaldi NumFrames(flush=True) 算帧数。

        snip_edges=True  —— 1 + (N - W)//S，只数完整落在波形里的帧
        snip_edges=False —— round(N/S)，帧中心固定钉在 t·S 上，
                            两端缺的部分靠镜像反射补出来
        """
        if self.snip_edges:
            if n_samples < self.frame_size:
                return 0
            return 1 + (n_samples - self.frame_size) // self.frame_step
        # 等价于 (N + S/2) // S，即四舍五入
        return (n_samples + self.frame_step // 2) // self.frame_step

    def frame_signal(self, signal, boundary=None):
        """按 kaldi FirstSampleOfFrame 切帧，返回 (n_frames, frame_size)。

        与 frame_and_window 的区别：
            - snip_edges=False 时帧起点 = f·S + S//2 - W//2，**可以是负数**，
              也可能越过波形末尾；越界部分按 kaldi 的做法用镜像反射补齐；
            - 只负责切帧，不加窗、不预加重（那是 process_window 的事）。

        参数:
            signal   —— 波形 (n_samples,)
            boundary —— 越界补齐方式；None 则用构造时的 self.boundary。
                        实测 kaldi 用的是 "symmetric"：左端补的是
                        x[119], x[118], …, x[0]（边缘样本 x[0] 出现两次）。
        返回:
            未加窗的原始帧矩阵 (n_frames, frame_size)
        """
        boundary = self.boundary if boundary is None else boundary
        signal = np.asarray(signal, dtype=np.float64).ravel()
        n = len(signal)
        nf = self.num_frames(n)
        if nf <= 0 or n == 0:
            return np.zeros((0, self.frame_size), dtype=np.float64)

        starts = np.arange(nf, dtype=np.int64) * self.frame_step
        if not self.snip_edges:
            starts = starts + self.frame_step // 2 - self.frame_size // 2

        offsets = np.arange(self.frame_size, dtype=np.int64)[None, :]
        if self.snip_edges:
            # 每一帧都完整落在波形内，直接花式索引
            return signal[starts[:, None] + offsets]

        # 先算出左右各要补多少，pad 完再统一取窗
        left = max(0, -int(starts.min()))
        right = max(0, int(starts.max()) + self.frame_size - n)
        if left or right:
            # numpy 的 reflect / symmetric 都要求 pad 宽度小于原数组长度，
            #   极短信号会直接报错，退回 edge（这种长度本来也取不到几帧）
            mode = boundary if (left < n and right < n) else "edge"
            signal = np.pad(signal, (left, right), mode=mode)
            starts = starts + left
        return signal[starts[:, None] + offsets]

    def process_window(self, frames):
        """kaldi ProcessWindow：对切好的每一帧依次做
        加抖 -> 去直流 -> 预加重 -> 加窗。

        顺序不能调换，kaldi 就是这么排的。其中预加重是**逐帧**做的：
        帧内倒序执行 window[i] -= α·window[i-1]，最后 window[0] *= (1-α)
        （相当于假设「帧起点前一个样本」等于帧首样本自身）。

        参数:
            frames —— 未加窗的帧矩阵 (n_frames, frame_size)
        返回:
            处理并加窗后的帧矩阵，形状不变
        """
        out = np.array(frames, dtype=np.float64, copy=True)

        if self.dither != 0.0:
            # kaldi 加的是高斯白噪声；一旦非 0 结果就不可复现
            rng = np.random.default_rng()
            out += rng.normal(scale=self.dither, size=out.shape)

        if self.remove_dc_offset:
            out -= out.mean(axis=1, keepdims=True)

        if self.preemph != 0.0:
            alpha = self.preemph
            # 先 .copy() 出右操作数，避免与左侧的视图重叠、读到已改写的值
            out[:, 1:] -= alpha * out[:, :-1].copy()
            out[:, 0] *= (1.0 - alpha)

        return out * self._win

    # --------------------------------------------------------
    # 步骤 3：FFT + 功率谱
    # --------------------------------------------------------
    def power_spectrum(self, frames, normalize=False):
        """对每帧做 FFT 并取模平方，得到短时功率谱（周期图）。

        先算幅度谱 |X|，再平方得到功率谱 —— 平方后各频率分量的相对强弱
        被拉开，更贴近「能量」的物理含义。也可以用对数代替平方，但平方
        是周期图（periodogram）的标准定义。

        参数:
            frames    —— 加窗后的帧矩阵 (n_frames, frame_size)
            normalize —— 是否除以 n_fft。周期图的标准定义要除以 N，
                         但本模块默认不除（尺度只影响后续对数的常数偏移，
                         归一化后无差别）。
        返回:
            功率谱矩阵 (n_frames, n_freq)，n_freq = n_fft // 2 + 1
        """
        # FFT 需要 float64，scipy.fft.rfft 会把输入转成 float64
        frames = np.asarray(frames, dtype=np.float64)
        # rfft 只取正频率半边：实信号的负频部分是共轭对称的，冗余
        spec = np.fft.rfft(frames, n=self.n_fft, axis=-1)
        # 功率谱 = 幅度谱平方 = |X|²
        power = np.abs(spec) ** 2
        # 周期图的标准定义是除以 N，但本模块默认不除（尺度只影响后续对数的常数偏移，归一化后无差别）
        if normalize:
            power = power / self.n_fft
        return power

    # --------------------------------------------------------
    # 步骤 4：Mel 滤波 + 对数压缩
    # --------------------------------------------------------
    def log_mel_spectrogram(self, power):
        """把线性功率谱过一遍 Mel 滤波器组，再取对数 —— 输出即 LogFBank。

        为什么用 Mel 刻度：257 维线性频谱冗余度高，而且人耳对频率的感知
        是非线性的 —— 低频区分辨力强、高频区弱。Mel 滤波器组在低频排得
        密、高频排得疏，正好模拟这一点，同时起到低通平滑的作用，抹掉
        因人而异的音色谐波和麦克风底噪，只留下决定字音的能量包络。

        为什么取对数：人耳对响度的感知近似对数关系，取对数后距离度量更
        符合听感；顺带把乘性增益变成加性偏移，对音量变化更鲁棒。

        参数:
            power —— 功率谱 (n_frames, n_freq)
        返回:
            LogFBank 特征 (n_frames, n_mels)
        """
        # Mel 滤波器组是 float32，乘法会被 upcast 到 float64
        power = np.asarray(power, dtype=np.float64)
        # 矩阵乘一次算完所有帧：power (F, K) @ filters.T (K, M) -> (F, M)
        mel_energy = power @ self.mel_filters.T

        if self.log_eps is None:
            # kaldi 口径：max(energy, float32_eps) 再取自然对数。
            #   用 max 而不是「加 eps」，静音帧才会停在 -15.94 而不是 -23 一带。
            return np.log(np.maximum(mel_energy, FLOAT32_EPS))
        # 旧口径：加极小值再取对数，避免 log(0) 得到 -inf
        return np.log(mel_energy + self.log_eps)

    # --------------------------------------------------------
    # 步骤 5：DCT
    # --------------------------------------------------------
    def dct(self, log_mel, lifter=None):
        """对 LogFBank 做 DCT，取前 n_mfcc 个系数 —— 输出 MFCC 静态系数。

        为什么还要做 DCT：LogFBank 相邻维度因为 Mel 滤波器相互重叠而高度
        相关，这对假设特征弱相关的传统模型（如 GMM-HMM）不友好。DCT 是
        正交变换，能把相关的 LogFBank 变到近似不相关的倒谱域，并具有
        能量集中特性 —— 前几个低阶系数正好对应缓慢变化的频谱包络
        （声道形状，决定字音），高阶系数才是快速变化的细节
        （声源激励和噪声）。只保留前 K 个系数就同时做到了去相关和降维。

        参数:
            log_mel —— LogFBank 特征 (n_frames, n_mels)
            lifter  —— 倒谱提升参数；None 则用构造时的 self.lifter
        返回:
            MFCC 静态系数 (n_frames, n_mfcc)
        """
        log_mel = np.asarray(log_mel, dtype=np.float64)
        lifter = self.lifter if lifter is None else lifter

        # DCT-II 正交归一化：能量守恒，系数尺度不随维度变化
        ceps = _dct_transform(log_mel, type=2, axis=-1, norm="ortho")
        ceps = ceps[:, :self.n_mfcc]

        if lifter is not None:
            # 倒谱提升：把高阶系数抬起来，抵消它们数值偏小、
            #   在欧氏距离里容易被忽略的问题
            n = np.arange(self.n_mfcc)
            ceps = ceps * (1.0 + (lifter / 2.0) * np.sin(np.pi * n / lifter))
        return ceps

    # --------------------------------------------------------
    # 步骤 6：动态特征（Δ / ΔΔ）
    # --------------------------------------------------------
    def dynamic_features(self, feats, K=2):
        """把静态特征扩展成「静态 + Δ + ΔΔ」的拼接，捕捉帧间的时间动态。

        静态特征只描述每一帧自身，Δ（一阶差分）描述频谱包络随时间的变化
        速度，ΔΔ（二阶差分）描述变化速度本身的变化。三者拼接是传统
        GMM-HMM 系统的标准做法（13 维 MFCC -> 39 维）。

        参数:
            feats —— 静态特征 (n_frames, n_dim)
            K     —— 差分窗口半径，通常取 2
        返回:
            (n_frames, n_dim * 3)，按 [静态, Δ, ΔΔ] 顺序拼接
        """
        feats = np.asarray(feats, dtype=np.float64)
        delta = self._delta(feats, K)
        # 二阶差分 = 对一阶差分再做一次同样的差分运算
        delta_delta = self._delta(delta, K)
        return np.hstack([feats, delta, delta_delta])

    @staticmethod
    def _delta(feats, K):
        """一阶差分：Δ[t] = Σ_{k=1..K} k·(f[t+k] - f[t-k]) / (2·Σ k²)。

        边界处把输入按边缘值延拓 K 帧（mode="edge"），于是首尾帧也有
        完整的对称窗口可用，不用单独写一套退化公式。
        """
        n_frames = len(feats)
        padded = np.pad(feats, ((K, K), (0, 0)), mode="edge")

        numerator = np.zeros_like(feats)
        for k in range(1, K + 1):
            # padded 的第 K 行正是 feats 的第 0 行
            after = padded[K + k: K + k + n_frames]    # f[t+k]
            before = padded[K - k: K - k + n_frames]   # f[t-k]
            numerator += k * (after - before)

        denominator = 2.0 * sum(k ** 2 for k in range(1, K + 1))
        return numerator / denominator

    # --------------------------------------------------------
    # 步骤 7：归一化
    # --------------------------------------------------------
    def normalize(self, feats, eps=1e-5):
        """逐维全局归一化（CMVN）：减均值、除标准差。

        消除说话人之间和信道之间的整体差异（音量、音色偏置），让特征
        分布更接近标准正态，对后面的模型更友好。

        注意：这里用的是「一条样本内部」的统计量。若要做标准的倒谱均值
        归一化（CMN），通常是拿整批训练数据或整句话的统计量来做。

        参数:
            feats —— 特征矩阵 (n_frames, n_dim)
            eps   —— 防止标准差为 0 时除零
        返回:
            归一化后的特征，形状不变
        """
        feats = np.asarray(feats, dtype=np.float64)
        mean = feats.mean(axis=0, keepdims=True)
        std = feats.std(axis=0, keepdims=True)
        return (feats - mean) / (std + eps)

    # --------------------------------------------------------
    # 流水线 A：LogFBank
    # --------------------------------------------------------
    def logfbank(self, signal, with_delta=False, do_normalize=False):
        """LogFBank 流程：预加重 → 分帧加窗 → 功率谱 → Mel 滤波 → 对数压缩。

        深度学习前端（Whisper / Conformer 等）的常用形态：停在取对数，
        保留 Mel 滤波器输出的全部信息，把「如何组合这些维度」交给网络
        自己学。省掉 DCT，计算量更小。

        参数:
            signal       —— 原始波形 (n_samples,)
            with_delta   —— 是否追加 Δ / ΔΔ（默认不加，深度学习一般用静态）
            do_normalize —— 是否做逐维归一化
        返回:
            默认 (n_frames, n_mels)；开 with_delta 则维度 ×3
        """
        feats = self._log_mel(signal)
        if with_delta:
            feats = self.dynamic_features(feats)
        if do_normalize:
            feats = self.normalize(feats)
        return feats

    # --------------------------------------------------------
    # 流水线 B：MFCC
    # --------------------------------------------------------
    def mfcc(self, signal, with_delta=True, do_normalize=True, lifter=None):
        """MFCC 流程：预加重 → 分帧加窗 → 功率谱 → Mel 滤波 → 对数压缩
                     → DCT（取前 K 个系数）→ 可选 Δ/ΔΔ → 可选归一化。

        参数:
            signal       —— 原始波形 (n_samples,)
            with_delta   —— 是否追加 Δ / ΔΔ（默认加，得到经典的 39 维）
            do_normalize —— 是否做逐维归一化（默认做，传统流程的惯例）
            lifter       —— 倒谱提升参数；None 则用构造时的 self.lifter
        返回:
            默认 (n_frames, n_mfcc * 3)
        """
        feats = self.dct(self._log_mel(signal), lifter=lifter)
        if with_delta:
            feats = self.dynamic_features(feats)
        if do_normalize:
            feats = self.normalize(feats)
        return feats

    # --------------------------------------------------------
    # 流水线 C：直通 onnx
    # --------------------------------------------------------
    def frontend_for_onnx(self, signal, tail_pad_frames=100):
        """直通 onnx 推理的前端：LogFBank -> float32 -> 尾部补帧。

        与直接调 logfbank() 相比只多两件事，都是被 onnx 模型倒逼出来的：

            1. 转成 float32。模型输入张量是 tensor(float)，onnxruntime
               不接受 float64，直接喂会报类型错误。
            2. 尾部补 tail_pad_frames 帧 0（默认 100 帧 ≈ 1 秒）。
               流式 zipformer2 的解码循环是 `while start + T < n_frames`，
               每次吃 T 帧、推进 decode_chunk_len 帧；尾部不补的话，最后一段
               音频凑不满一个 chunk 就会被整段丢掉，尾字识别不出来。
               注意补的 0 是**特征域**的 0（log 域对应能量 1），不是静音 ——
               这是照抄官方 test_onnx.py 的做法，不要自作主张改成静音值。

        参数:
            signal           —— 原始波形 (n_samples,)，[-1, 1] 量纲
            tail_pad_frames  —— 尾部补几帧 0；0 表示不补
        返回:
            float32 特征矩阵 (n_frames + tail_pad_frames, n_mels)
        """
        feats = np.asarray(self._log_mel(signal), dtype=np.float32)
        if tail_pad_frames > 0:
            feats = np.pad(feats, ((0, tail_pad_frames), (0, 0)),
                           mode="constant", constant_values=0.0)
        return feats

    # --------------------------------------------------------
    # 内部：两条流水线共用的前半段
    # --------------------------------------------------------
    def _log_mel(self, signal):
        """预加重 → 分帧加窗 → 功率谱 → Mel 滤波 → 对数压缩。

        这正是 LogFBank 流程的全部内容，也是 MFCC 流程的前 7 步。

        按 frame_preemphasis 分两条路：
            True（默认）—— kaldi 口径：先切帧，再逐帧做
                            加抖 / 去直流 / 预加重 / 加窗（process_window）
            False        —— 教科书写法：先对整段波形预加重，再切帧加窗
        """
        if self.frame_preemphasis:
            frames = self.process_window(self.frame_signal(signal))
        else:
            frames = self.frame_signal(self.pre_emphasis(signal)) * self._win
        return self.log_mel_spectrogram(self.power_spectrum(frames))

    # --------------------------------------------------------
    # 内部：窗函数与 Mel 滤波器组（构造时各建一次）
    # --------------------------------------------------------
    def _build_window(self, frame_size):
        """构造窗函数，与 kaldi GetWindow 逐点一致。

        全部手算、不用 scipy.signal.get_window，原因只有一个：**分母**。
        kaldi 的所有窗都用 (N-1) 作分母（对称窗，首尾都取到零点），而
        scipy 默认 periodic=True 用 N 作分母 —— 两者在 N=400 时最大偏差
        5.5e-3（hamming），足以把 log-mel 特征拉开可见的差距。

        支持的类型（与 kaldi 一致）:
            povey                     (0.5 - 0.5·cos(a·n))^0.85   ← zipformer 用这个
            hamming                   0.54 - 0.46·cos(a·n)
            hanning / hann            0.5 - 0.5·cos(a·n)
            sine                      sin(0.5·a·n)
            blackman                  0.42 - 0.5·cos(a·n) + 0.08·cos(2·a·n)
            rectangular               全 1（不做加窗）
        """
        name = str(self.window).lower()
        n = np.arange(frame_size, dtype=np.float64)
        a = 2.0 * np.pi / (frame_size - 1)

        if name == "povey":
            # kaldi 作者自创：形状接近 Hamming，但两端严格归零
            return (0.5 - 0.5 * np.cos(a * n)) ** 0.85
        if name == "hamming":
            return 0.54 - 0.46 * np.cos(a * n)
        if name in ("hanning", "hann"):
            return 0.5 - 0.5 * np.cos(a * n)
        if name == "sine":
            return np.sin(0.5 * a * n)
        if name == "blackman":
            coeff = 0.42
            return coeff - 0.5 * np.cos(a * n) + (0.5 - coeff) * np.cos(2.0 * a * n)
        if name in ("rectangular", "rect", "boxcar", "ones", "none"):
            return np.ones(frame_size, dtype=np.float64)

        # 其余窗型退回 scipy（注意其 periodic 约定与 kaldi 不同）
        return get_window(name, frame_size)

    def _build_mel_filterbank(self):
        """构造 Mel 三角滤波器组，形状 (n_mels, n_freq)，与 kaldi
        MelBanks::InitKaldiMelBanks 逐点一致。

        kaldi 的做法（和「先算出浮点 bin 位置、再按 bin 序号插值」不同）：
            1. 把 low_freq / high_freq 换到 Mel 刻度，在 Mel 刻度上等分
               n_mels + 1 段，每个滤波器得到 left / center / right 三个 Mel 边界；
            2. **遍历每一个整数 FFT bin**，把它的中心频率 i·(fs/N) 换到 Mel；
            3. 若该 Mel 值严格落在 (left, right) 区间内，就按它到 center 的
               距离在 **Mel 域**做线性插值，赋给该 bin。

        为什么必须这样：Hz 到 Mel 是非线性的，第 3 步必须拿 Mel 值（而不是 bin
        序号）去插值才能得到 kaldi 的形状；另外边界用严格不等号，落在边界上的
        bin 直接不参与（权重 0），而不是插值出 0.5。
        """
        nyquist = 0.5 * self.sample_rate
        # kaldi 约定：high_freq = 0 表示不截断（取到 Nyquist）；负数表示 Nyquist + 该值
        high_freq = self.high_freq if self.high_freq > 0.0 else nyquist + self.high_freq
        fft_bin_width = self.sample_rate / float(self.n_fft)

        mel_low = hz_to_mel_kaldi(self.low_freq)
        mel_high = hz_to_mel_kaldi(high_freq)
        mel_delta = (mel_high - mel_low) / (self.n_mels + 1)

        # 每个整数 bin 的中心频率先统一换算到 Mel，避免内层循环重复算
        bin_mels = hz_to_mel_kaldi(fft_bin_width * np.arange(self.n_freq))

        filters = np.zeros((self.n_mels, self.n_freq), dtype=np.float64)
        for m in range(self.n_mels):
            left = mel_low + m * mel_delta
            center = mel_low + (m + 1) * mel_delta
            right = mel_low + (m + 2) * mel_delta
            inside = (bin_mels > left) & (bin_mels < right)
            rising = inside & (bin_mels <= center)
            falling = inside & (bin_mels > center)
            filters[m, rising] = (bin_mels[rising] - left) / (center - left)
            filters[m, falling] = (right - bin_mels[falling]) / (right - center)
        return filters


# ============================================================
# 五、自测
# ============================================================
if __name__ == "__main__":
    # 用一段 1 秒 440Hz 正弦合成信号跑通三条流水线，检查输出形状与口径
    sr = 16000
    t = np.arange(sr) / sr
    signal = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)

    ext = TraditionalFeatureExtractor(sample_rate=sr)

    fb = ext.logfbank(signal)
    fb_dyn = ext.logfbank(signal, with_delta=True, do_normalize=True)
    mc = ext.mfcc(signal, with_delta=False, do_normalize=False)
    mc_full = ext.mfcc(signal)
    onnx_feats = ext.frontend_for_onnx(signal)

    print(f"帧长 {ext.frame_size} 点 / 帧移 {ext.frame_step} 点 / 频率 bin {ext.n_freq} 个")
    print(f"Mel 滤波器组: {ext.mel_filters.shape}")
    print(f"LogFBank          : {fb.shape}  dtype={fb.dtype}")
    print(f"LogFBank + Δ/ΔΔ   : {fb_dyn.shape}")
    print(f"MFCC 静态         : {mc.shape}")
    print(f"MFCC + Δ/ΔΔ + CMVN: {mc_full.shape}")
    print(f"直通 onnx         : {onnx_feats.shape}  dtype={onnx_feats.dtype}")

    # 口径自检：snip_edges=False 时帧数应为 round(N/S)
    print(f"帧数核对: num_frames={ext.num_frames(len(signal))}"
          f"  round(N/S)={round(len(signal) / ext.frame_step)}")
    # 口径自检：静音帧的取值地板应停在 kaldi 的 log(float32_eps) 附近
    print(f"LogFBank 最小值={fb.min():.4f}"
          f"  kaldi 下限 log(float32_eps)={np.log(FLOAT32_EPS):.4f}")
