# -*- coding: utf-8 -*-
"""麦克风采集：输出裸音频流（mini 版第三条流水线）。

============================================================
这份代码是干什么的
============================================================
打开声卡，把麦克风的音频一块一块收进缓冲，等主循环主动来取。

    PortAudio 采集线程 ──callback()──> deque 缓冲 ──get_last_audio()──> 主循环

我们**不自己创建任何 Python 线程**。回调函数由 PortAudio 内部线程调用，
它只做一件微小的事：把这一块样本复制一份塞进缓冲，然后立刻返回。
真正的识别、打印都在主循环里做。

============================================================
一、回调里为什么必须 .copy()
============================================================
sounddevice 传给回调的 indata 是一块**会被复用**的底层缓冲 —— 回调返回后，
PortAudio 就会拿同一块内存装下一批音频。不 copy 直接存引用的话，主循环
读到的可能是已经被覆盖的脏数据，表现为偶发的、无法复现的识别错误。

所以回调里必须 `indata[:, 0].copy()`，并且除这次拷贝外不做任何别的事
（不碰模型、不打印、不加锁）。

============================================================
二、get_last_audio() 的语义：取走全部，不是只留最新一块
============================================================
主循环在两次拉取之间，可能被一次 encoder 前向（约 40ms）卡住，而这期间
声卡已经送来 3~4 块（每块 10ms）。

如果只保留"最新一块"，中间那 30ms 语音就被静默丢掉了 —— 波形出现断口，
去直流和预加重会把断口当成突变，Mel 谱乱掉，结果是吞字或识别成乱码。

所以这里取走的是**自上次调用以来累积的全部样本**，波形连续无损。慢一点
只是延迟变大，识别结果仍然完整。

（对照：生产版 src/streaming_asr.py 用的是"丢最旧块"，那是为了 GUI 画面的
低延迟；mini 版要的是正确性，所以选择不丢。）

============================================================
三、线程安全：为什么不加锁
============================================================
生产者（回调线程）只往 deque 右端 append，消费者（主循环）只从 deque 左端
popleft。在 CPython 里，deque 的 append 和 popleft 各自是一个原子的 C 操作，
不会读到半截状态；一右一左，两边也不会互相干扰。所以不需要加锁。

**尤其不能在回调里加锁**：回调属于 PortAudio 的实时线程，一旦它卡在锁上，
声卡缓冲就会溢出，表现是丢块和爆音。

deque 设了 maxlen 作内存上界。万一主循环长时间跟不上，缓冲满了会自动淘汰
最旧的块，dropped 计数会涨 —— 这个数字可观测，方便判断是不是该调大 block
或降低负载了。

依赖：只用 numpy + sounddevice。
"""

import collections

import numpy as np
import sounddevice as sd


class MicClient:
    """麦克风裸流客户端。

    用法::

        mic = MicClient()
        mic.start()
        while 想继续:
            samples = mic.get_last_audio()    # 没有新数据时返回 None
            if samples is None:
                continue
            ...                               # 拿去做前端
        mic.stop()
    """

    def __init__(self, sample_rate=16000, channels=1, block=160,
                 device=None, max_blocks=256):
        """
        参数:
            sample_rate —— 采样率（Hz），模型要求 16000
            channels    —— 采集声道数，1 = 单声道
            block       —— 每次回调交付的样本数。160 = 10ms @16kHz
            device      —— 设备编号或名字；None = 系统默认输入设备
            max_blocks  —— 缓冲最多存几块；满了淘汰最旧的（约 2.56s @256 块）
        """
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.block = int(block)
        self.device = device
        self.max_blocks = int(max_blocks)

        # 采集线程写右端、主循环读左端，无需加锁（见模块 docstring 第三节）
        self._q = collections.deque(maxlen=self.max_blocks)
        self._stream = None
        self._dropped = 0
        self._peak = 0.0
        self._n_samples = 0

    # ========================================================
    # 采集线程：唯一的入口就是这个回调，只做 copy + 入队
    # ========================================================
    def _callback(self, indata, frames, time_info, status):
        if status:
            # 溢出/欠载之类的状态提示，不致命，报一声方便排查
            print("[音频状态] %s" % status)

        # 必须 copy：indata 的底层缓冲会被 PortAudio 立刻复用
        blk = indata[:, 0].copy()

        # deque 满了的话，接下来这次 append 会淘汰最旧的块 —— 记账一下
        if len(self._q) == self._q.maxlen:
            self._dropped += 1

        self._q.append(blk)

        # 峰值幅度：只看不写临界，偶尔的竞争无伤大雅，就不加锁了
        p = float(np.abs(blk).max())
        if p > self._peak:
            self._peak = p

    # ========================================================
    # 对外接口
    # ========================================================
    def start(self):
        """打开麦克风，采集线程开始往里送数据。"""
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.block,
            device=self.device,
            callback=self._callback)
        self._stream.start()

    def get_last_audio(self):
        """取走自上次调用以来累积的全部样本。

        返回:
            一维 float32 波形；缓冲为空时返回 None
        """
        chunks = []
        while True:
            try:
                chunks.append(self._q.popleft())
            except IndexError:
                break
        if not chunks:
            return None
        self._n_samples += sum(c.size for c in chunks)
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)

    def stop(self):
        """关闭麦克风。之后仍可调 get_last_audio() 取走残余样本。"""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # ========================================================
    # 状态（给入口脚本打印 / 判断用）
    # ========================================================
    @property
    def dropped(self):
        """因缓冲上限被淘汰的块数。持续增长说明主循环跟不上。"""
        return self._dropped

    @property
    def peak(self):
        """至今为止的最大样本幅度，用来确认麦克风真的收到了声音。"""
        return self._peak

    @property
    def running(self):
        """麦克风是否还开着。"""
        return self._stream is not None and self._stream.active

    @property
    def n_samples(self):
        """已被取走的累计样本数。"""
        return self._n_samples


def list_devices():
    """打印本机所有音频设备，以及默认输入设备。"""
    print(sd.query_devices())
    print("\n默认输入设备：%s" % (sd.query_devices(kind="input"),))
