# -*- coding: utf-8 -*-
"""流式（增量）特征提取：把 TraditionalFeatureExtractor 的分帧改成能一块一块喂。

============================================================
一、为什么不能直接用批量版
============================================================

批量版是一次性拿到整段波形，靠 np.pad 在两端镜像补齐、再一次性滑窗取帧
（traditional_features.py 的 frame_signal）。麦克风场景拿不到「未来」的样本，
只能每 10ms 喂进来 160 个点，于是有三件事必须自己做：

    1. 维护跨块的样本尾巴 —— 一帧要 400 个点，而一块只有 160 个，得攒；
    2. 自己实现两端的镜像补齐 —— 且补法要与批量版的 np.pad 逐点一致；
    3. 丢掉再也用不到的旧样本 —— 否则跑一晚上内存就爆了。

帧内处理（加抖 / 去直流 / 逐帧预加重 / 加窗 / FFT / Mel / 取对数）一步都不
自己写，全部复用 TraditionalFeatureExtractor。流式版只接管「分帧」这一层，
这样两条路径不可能各自漂移。

代码分成两层，是为了让「分帧」这件事能被单独测：

    StreamingFramer     只管把波形切成帧（含两端镜像补齐），输出 (k, 400)
    StreamingLogFBank   在 Framer 的输出上做帧内处理，输出 (k, 80) float32

============================================================
二、帧的样本区间
============================================================

kaldi FirstSampleOfFrame（snip_edges=False，S=帧移 160，W=帧长 400）：

    第 m 帧的样本区间 lo = m·S + S//2 - W//2 = 160m - 120
                        hi = lo + W          = 160m + 280

m=0 时 lo = -120 < 0，需要左镜像补齐；流末尾 hi 会越过已到达的样本数，需要
右镜像补齐。镜像方式由 boundary 决定（默认 symmetric：sig[-k] = sig[k-1]，
即镜像时重复边缘样本；reflect 则不重复）。

注意左补齐量恒为 120，**只有第 0 帧**用得到：m=1 时 lo 已经是 40 > 0。

============================================================
三、一个非显然的口径：短信号会退回 edge
============================================================

批量版 frame_signal 里有这么一行：

    mode = boundary if (left < n and right < n) else "edge"

因为 numpy 的 reflect / symmetric 都不允许 pad 宽度超过原数组长度，短信号
只能退回 edge（左右都复制边缘样本）。代入 kaldi 的左右补齐公式可以算出，
这个回退实际只在 **n <= 140**（不到 9ms）时触发。流式版在收尾、样本总数已
经确定时才判定边界模式，用的是同一个判据，所以两条路径一致。

============================================================
四、用法
============================================================

调用节奏与 kaldi_native_fbank.OnlineFbank 一致::

    fbank = StreamingLogFBank(sample_rate=16000)
    while 还有音频:
        feats = fbank.accept_waveform(chunk)   # (n_new, 80) float32，可能 0 帧
    feats = fbank.input_finished()             # 收尾：补完尾部帧 + 可选尾部补 0

单帧的因果延迟 = W//2 = 200 样本 = 12.5ms：第 m 帧要等样本攒到 160m+280 才能
输出，而它的中心在 160m+80，所以平均滞后 200 个样本。

依赖：只用 numpy + traditional_features.py。
"""

import os
import sys

import numpy as np

# -I 隔离模式下脚本目录不进 sys.path，这里显式加回才能 import 同目录模块
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from traditional_features import TraditionalFeatureExtractor


# ============================================================
# 一、流式分帧（只管切，不管帧内处理）
# ============================================================
class StreamingFramer:
    """增量分帧：喂任意长度的小块，凑够一帧就吐一帧原始样本。

    输出的帧与批量版 TraditionalFeatureExtractor.frame_signal() 逐位相同
    （自测里按 bit-exact 校验），因为两边的切帧公式与镜像补法完全一致。

    参数:
        ext —— 借它拿 frame_size / frame_step / num_frames / boundary，
               以及 snip_edges、dither 等口径约束；不读它的窗和滤波器组
    """

    def __init__(self, ext):
        if not ext.frame_preemphasis:
            raise ValueError(
                "流式版只支持 frame_preemphasis=True（kaldi 口径：先分帧再逐帧"
                "预加重）。教科书写法要把预加重作用在整段波形上，而流式场景下"
                "「整段」是流动的，等价实现没有意义。")
        if ext.snip_edges:
            raise ValueError(
                "流式版只支持 snip_edges=False（kaldi 口径）。snip_edges=True "
                "会丢掉边缘帧，帧序在流式下无法与批量版对齐。")
        if ext.dither != 0.0:
            raise ValueError(
                "流式版不支持 dither != 0。kaldi 的加抖是随机高斯噪声，批量版"
                "一次调用一个随机流、流式版每块一个，两边天然对不上；推理一律"
                "用 dither=0。")

        self.ext = ext
        self.N = ext.frame_size                       # 帧长 400
        self.S = ext.frame_step                       # 帧移 160
        self.first = self.S // 2 - self.N // 2        # 第 0 帧起点 = -120
        self.left_pad = self.N // 2 - self.S // 2     # 恒为 120
        self.reset()

    def reset(self):
        """清空全部状态，回到「一个样本都没喂过」。"""
        # _buf 保存绝对索引 [_base, _base+len(_buf)) 范围内的样本。
        #   用绝对索引而不是「相对当前块」，跨块拼接才不会错位。
        self._buf = np.zeros(0, dtype=np.float64)
        self._base = 0        # _buf[0] 对应的绝对样本序号
        self._n = 0           # 累计已喂入的样本总数
        self._next = 0        # 下一个要输出的帧号
        self._done = False
        self._nf_total = 0    # 收尾时才确定的批量帧数

    @property
    def num_frames_ready(self):
        """已经吐出的帧数（累计）。"""
        return self._next

    @property
    def n_samples_seen(self):
        """累计已喂入的样本数。"""
        return self._n

    # --------------------------------------------------------
    # 喂数据
    # --------------------------------------------------------
    def accept_waveform(self, samples):
        """喂入一块样本，返回本次新凑出来的原始帧。

        参数:
            samples —— 任意长度的一维波形（float32/float64 都行）
        返回:
            未加任何处理的帧矩阵 (n_new, frame_size)，float64
        """
        if self._done:
            raise RuntimeError("流已结束（input_finished 已调用）；"
                               "要复用请先 reset()")
        chunk = np.asarray(samples, dtype=np.float64).ravel()
        if chunk.size == 0:
            return self._empty()
        # 这几百个样本的拷贝可以忽略：16000Hz 下每秒只拼 100 次、每次不到 1KB
        self._buf = (np.concatenate((self._buf, chunk))
                     if self._buf.size else chunk.copy())
        self._n += chunk.size
        return self._stack(self._drain())

    def input_finished(self):
        """宣告流结束，补出尾部帧。

        此时样本总数已定，批量版的帧数 (n + S//2)//S 和边界模式都能算准了。

        返回:
            尾部帧矩阵 (n_rest, frame_size)，float64
        """
        if self._done:
            return self._empty()
        self._done = True
        self._nf_total = self.ext.num_frames(self._n)
        return self._stack(self._drain())

    # --------------------------------------------------------
    # 内部：出帧
    # --------------------------------------------------------
    def _drain(self):
        """把当前已经能确定的帧全部取出来。

        流未结束时，只有 hi <= n 的帧能出（不需要右补齐）；收尾之后，帧数上界
        由批量版的 num_frames 给定，尾部帧允许右补齐。
        """
        frames = []
        while True:
            m = self._next
            if self._done:
                if m >= self._nf_total:
                    break
            elif m * self.S + self.first + self.N > self._n:
                break
            frames.append(self._frame(m))
            self._next = m + 1
            self._trim()
        return frames

    def _frame(self, m):
        """取第 m 帧的 N 个原始样本，两端越界处镜像补齐。

        与批量版的对应关系：批量版对整段做一次 np.pad 再滑窗，本方法是逐帧
        只补自己越界的那一段。因为补出来的点只取决于贴边的那几个样本、且
        symmetric/reflect 的镜像长度固定，逐帧补与整段补得到的 400 个点相同。
        """
        lo = m * self.S + self.first
        hi = lo + self.N
        n = self._n

        a = max(lo, 0)          # 真实样本区间的绝对起点
        b = min(hi, n)          # 真实样本区间的绝对终点
        seg = self._buf[a - self._base: b - self._base]

        nl = a - lo             # 左补齐个数（只在 m == 0 时非 0）
        nr = hi - b             # 右补齐个数（只在流末尾非 0）
        if nl == 0 and nr == 0:
            return seg

        mode = self._pad_mode(n)
        parts = []
        if nl:
            parts.append(self._left_pad(nl, mode))
        parts.append(seg)
        if nr:
            parts.append(self._right_pad(nr, mode))
        return np.concatenate(parts)

    def _pad_mode(self, n):
        """复刻批量版 frame_signal 的边界模式判定（含退化到 edge 的分支）。

        批量版：left = 120，right = (nf-1)·S + first + N - n；两者都小于 n 时
        才用配置的 boundary，否则整段退回 edge。
        """
        nf = self.ext.num_frames(n)
        if n <= 0 or nf <= 0:
            return "edge"
        right = max(0, (nf - 1) * self.S + self.first + self.N - n)
        if self.left_pad < n and right < n:
            return self.ext.boundary
        return "edge"

    def _left_pad(self, k, mode):
        """补在帧首的 k 个点。src 取自信号开头（_base 此时必然还是 0）。"""
        if mode == "edge":
            return np.full(k, self._buf[0 - self._base], dtype=np.float64)
        if mode == "symmetric":                 # sig[-j] = sig[j-1]
            return self._buf[0 - self._base: k - self._base][::-1]
        if mode == "reflect":                   # sig[-j] = sig[j]
            return self._buf[1 - self._base: k + 1 - self._base][::-1]
        raise ValueError("流式版不支持的 boundary=%r（支持 symmetric/reflect/"
                         "edge）" % mode)

    def _right_pad(self, k, mode):
        """补在帧尾的 k 个点。src 取自信号结尾。"""
        n = self._n
        if mode == "edge":
            return np.full(k, self._buf[n - 1 - self._base], dtype=np.float64)
        if mode == "symmetric":                 # sig[n+j] = sig[n-1-j]
            return self._buf[n - k - self._base: n - self._base][::-1]
        if mode == "reflect":                   # sig[n+j] = sig[n-2-j]
            return self._buf[n - k - 1 - self._base:
                             n - 1 - self._base][::-1]
        raise ValueError("流式版不支持的 boundary=%r（支持 symmetric/reflect/"
                         "edge）" % mode)

    def _trim(self):
        """丢掉后面任何一帧都用不到的旧样本。

        下一帧（_next）的起点是 _next·S + first，它之前的样本再也不会被读到。
        唯一的例外是第 0 帧的左镜像要用 sig[0:120]，而那只在 _next == 0 时
        发生 —— 此时 _trim 还没机会跑过，_base 仍是 0。
        """
        keep_from = max(0, self._next * self.S + self.first)
        k = keep_from - self._base
        if k > 0:
            # .copy() 而不是切片：切片是视图，会让整块原始内存一直挂着不释放
            self._buf = self._buf[k:].copy()
            self._base = keep_from

    def _stack(self, frames):
        if not frames:
            return self._empty()
        return np.stack(frames)

    def _empty(self):
        return np.zeros((0, self.N), dtype=np.float64)


# ============================================================
# 二、流式 LogFBank（分帧 + 帧内处理）
# ============================================================
class StreamingLogFBank:
    """增量 LogFBank：分帧交给 StreamingFramer，帧内处理复用批量提取器。

    参数:
        extractor —— 已有的 TraditionalFeatureExtractor；给了就整个沿用它的
                     参数（推荐，保证与批量路径同一份配置），此时 **kwargs 无效
        **kwargs  —— 转发给 TraditionalFeatureExtractor，见其文档
        tail_pad_frames —— input_finished() 时在尾部补几帧 0（特征域 0，不是
                     静音）。给 onnx 流式解码用，默认 0

    输出 dtype 是 float32（onnx 输入要求），批量版 logfbank() 是 float64，所以
    两者相比会有 float32 舍入量级（约 1e-7）的差异 —— 见自测第二节。
    """

    def __init__(self, extractor=None, tail_pad_frames=0, **kwargs):
        self.ext = (extractor if extractor is not None
                    else TraditionalFeatureExtractor(**kwargs))
        self.framer = StreamingFramer(self.ext)
        self.n_mels = self.ext.n_mels
        self.tail_pad_frames = int(tail_pad_frames)

    def reset(self):
        self.framer.reset()

    @property
    def num_frames_ready(self):
        return self.framer.num_frames_ready

    @property
    def n_samples_seen(self):
        return self.framer.n_samples_seen

    def accept_waveform(self, samples):
        """喂入一块样本，返回本次新凑出来的特征。

        参数:
            samples —— 任意长度的一维波形
        返回:
            float32 特征矩阵 (n_new, n_mels)；这一块还没凑够一帧时是 (0, n_mels)
        """
        return self._featurize(self.framer.accept_waveform(samples))

    def input_finished(self):
        """宣告流结束，补出尾部帧，必要时再补若干帧特征域 0。

        返回:
            float32 特征矩阵 (n_rest [+ tail_pad_frames], n_mels)
        """
        feats = self._featurize(self.framer.input_finished())
        if self.tail_pad_frames > 0 and feats.shape[0] > 0:
            feats = np.pad(feats, ((0, self.tail_pad_frames), (0, 0)),
                           mode="constant", constant_values=0.0)
            feats = np.ascontiguousarray(feats, dtype=np.float32)
        return feats

    # --------------------------------------------------------
    # 内部：帧内处理（与批量版 _log_mel 的后半段逐句对应）
    # --------------------------------------------------------
    def _featurize(self, frame_mat):
        """把一组原始帧走完 process_window -> power -> mel -> log。

        顺序与 TraditionalFeatureExtractor._log_mel 完全一致，分帧那一层换成
        了 StreamingFramer。
        """
        if frame_mat.shape[0] == 0:
            return np.zeros((0, self.n_mels), dtype=np.float32)
        proc = self.ext.process_window(frame_mat)
        feats = self.ext.log_mel_spectrogram(self.ext.power_spectrum(proc))
        return np.ascontiguousarray(feats, dtype=np.float32)


# ============================================================
# 三、自测
# ============================================================
def _cat(chunks, n_mels):
    chunks = [c for c in chunks if c.shape[0] > 0]
    if not chunks:
        return np.zeros((0, n_mels), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def _probe(signal, hop):
    """按 hop 切块喂完一条信号，返回 (原始帧, 特征)。"""
    ext = _EXT
    fr = StreamingFramer(ext)
    fb = StreamingLogFBank(extractor=ext)
    raws, feats = [], []
    for i in range(0, len(signal), hop):
        blk = signal[i:i + hop]
        a = fr.accept_waveform(blk)
        if a.shape[0]:
            raws.append(a)
        b = fb.accept_waveform(blk)
        if b.shape[0]:
            feats.append(b)
    for x, y in ((fr.input_finished(), fb.input_finished()),):
        if x.shape[0]:
            raws.append(x)
        if y.shape[0]:
            feats.append(y)
    return _cat(raws, ext.frame_size), _cat(feats, ext.n_mels)


_EXT = None


if __name__ == "__main__":
    sr = 16000
    rng = np.random.default_rng(0)
    t = np.arange(sr) / sr
    # 两个正弦 + 一点噪声：噪声是必要的，否则静音段全被 log 下限压成常数，
    #   段内的细微差异会被掩盖
    signal = (0.6 * np.sin(2.0 * np.pi * 440.0 * t)
              + 0.25 * np.sin(2.0 * np.pi * 1300.0 * t)
              + 0.01 * rng.standard_normal(sr)).astype(np.float64)

    _EXT = TraditionalFeatureExtractor(sample_rate=sr)
    ext = _EXT

    ref_frames = np.asarray(ext.frame_signal(signal), dtype=np.float64)
    ref_feats = np.asarray(ext.logfbank(signal), dtype=np.float64)
    print("批量版：帧数 = %d，帧长 = %d，n_mels = %d"
          % (ref_feats.shape[0], ext.frame_size, ext.n_mels))
    print()

    # ---- 第 1 层：分帧是否逐位相同（这是新写的代码，要求 bit-exact）----
    print("[1] 分帧层  StreamingFramer vs frame_signal（要求逐位相同）")
    for hop in (160, 100, 257, 400, 1000, 33):
        got, _ = _probe(signal, hop)
        n = min(ref_frames.shape[0], got.shape[0])
        d = np.abs(ref_frames[:n] - got[:n])
        print("   块长 %-5d 帧数 批量=%-4d 流式=%-4d  max|Δ|=%.3e  %s"
              % (hop, ref_frames.shape[0], got.shape[0], d.max(),
                 "BIT-EXACT OK" if d.max() == 0.0 else "FAIL"))

    # ---- 第 2 层：特征层是否逐位相同 ----
    # 注意别拿 float64 参考去比 float32 输出：那样量到的是 float32 的表示
    #   精度（约 5e-7），跟流式/批量是否一致无关。要把两边都收敛到同一个
    #   dtype 再比，才是在比代码。
    print()
    print("[2] 特征层  StreamingLogFBank vs logfbank")
    print("    左列 = 两者都转 float32 后逐位比（比的是代码）")
    print("    右列 = float64 参考 vs float32 输出（量的是 float32 表示精度）")
    for hop in (160, 100, 257, 400, 1000, 33):
        _, got = _probe(signal, hop)
        n = min(ref_feats.shape[0], got.shape[0])
        r64 = ref_feats[:n]
        g64 = got[:n].astype(np.float64)
        d_same_dtype = np.abs(r64.astype(np.float32).astype(np.float64) - g64)
        d_dtype = np.abs(r64 - g64)
        print("   块长 %-5d 帧数=%-4d  max|Δ|=%s  %s  |  %.3e"
              % (hop, n,
                 "0.000e+00" if d_same_dtype.max() == 0.0
                 else "%.3e" % d_same_dtype.max(),
                 "BIT-EXACT OK" if d_same_dtype.max() == 0.0 else "FAIL",
                 d_dtype.max()))

    # ---- 第 3 层：说明右列那个数是什么 ----
    print()
    print("[3] 右列那个数的来源：同一批帧，整体过一次 vs 分两半各过一次")
    frames = ext.frame_signal(signal)
    def pipe(f):
        return ext.log_mel_spectrogram(ext.power_spectrum(ext.process_window(f)))
    whole = pipe(frames)
    half = np.vstack([pipe(frames[:50]), pipe(frames[50:])])
    print("   同一批量路径只改分块：max|Δ|=%.3e（float64 eps 量级）"
          % np.abs(whole - half).max())
    print("   同一批量路径只改 dtype：max|Δ|=%.3e（float32 表示精度）"
          % np.abs(whole - whole.astype(np.float32).astype(np.float64)).max())
    print("   -> 左列为 0 说明流水线逐位一致；右列完全由最后转 float32 那一步")
    print("      决定，与分帧和运算顺序无关。")

    # ---- 第 4 层：短信号，专门打 _pad_mode 的 edge 退化分支 ----
    print()
    print("[4] 短信号（n <= 140 时批量版会把镜像补齐退化成 edge）")
    for n in (60, 120, 140, 141, 200, 279, 280, 281):
        sig = signal[:n]
        r = np.asarray(ext.frame_signal(sig), dtype=np.float64)
        g, _ = _probe(sig, 160)
        if r.shape != g.shape:
            print("   n=%-4d 形状不一致 批量=%s 流式=%s  FAIL"
                  % (n, r.shape, g.shape))
            continue
        d = np.abs(r - g)
        print("   n=%-4d 帧数=%-2d  max|Δ|=%.3e  %s"
              % (n, r.shape[0], d.max() if d.size else 0.0,
                 "BIT-EXACT OK" if d.size == 0 or d.max() == 0.0 else "FAIL"))

    # ---- 第 5 层：内存有界性与尾部补帧 ----
    print()
    print("[5] 内存与接口")
    fr = StreamingFramer(ext)
    peak = 0
    for i in range(0, len(signal), 160):
        fr.accept_waveform(signal[i:i + 160])
        peak = max(peak, fr._buf.size)
    print("   滚完 1 秒后缓冲峰值 = %d 个样本（帧长 %d + 块长 160 = %d 是理论上界）"
          % (peak, ext.frame_size, ext.frame_size + 160))
    fb = StreamingLogFBank(extractor=ext, tail_pad_frames=100)
    for i in range(0, len(signal), 160):
        fb.accept_waveform(signal[i:i + 160])
    tot = fb.input_finished()
    print("   tail_pad_frames=100：收尾共 %d 帧，其中包含补的 100 帧 0：%s"
          % (tot.shape[0], bool(np.all(tot[-100:] == 0.0))))
    print("   汇总帧数 = %d，与批量版 %d %s"
          % (fb.num_frames_ready, ref_feats.shape[0],
             "一致" if fb.num_frames_ready == ref_feats.shape[0] else "不一致"))
