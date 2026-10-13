# -*- coding: utf-8 -*-
"""流式 ASR 解码：LogFBank 特征 -> 文字（mini 版第二条流水线）。

============================================================
这份代码是干什么的
============================================================
拿 onnxruntime 跑三段模型（encoder / decoder / joiner），对每一帧特征做
贪心搜索（greedy search），把得到的 token 序列拼成中文文本。

模型是 streaming zipformer2 transducer，三段的分工：

    encoder  —— 吃特征，吐出"声学编码"；它带着一大堆**状态张量**，跨块累积，
                这就是"流式"能记住前文的全部机制。
    decoder  —— 吃最近若干个 token，吐出"语言编码"（也叫 predictor）。
    joiner   —— 把 encoder 的一帧和 decoder 的输出合起来，输出整个词表的打分。

============================================================
一、流式的核心：T 与 decode_chunk_len 是两个不同的数
============================================================
每次喂给 encoder 的是 T = 61 帧，但时间轴只**前进** decode_chunk_len = 48 帧
（480ms）。多出来的 13 帧是卷积模块需要的重叠上下文 —— 它们下一轮还要再喂
一遍。实测每个 chunk 的 encoder 输出是 12 帧，不是 61 也不是 48。

所以有三条铁律：
    · T 是**输入帧数**，不是输出帧数；
    · encoder 输出的帧数必须**运行时取** encoder_out.shape[1]；
    · 这两个数都从模型的 metadata 读，不硬编码（换模型就变）。

============================================================
二、贪心搜索：blank 不刷新 decoder
============================================================
对 encoder 吐出的每一帧，跟当前 decoder 输出一起过 joiner，取 argmax：

    · 如果结果不是 blank —— 把它接到假设尾部，并用最近 context_size 个
      token 重算一次 decoder 输出；
    · 如果是 blank —— **什么都不做**，沿用上一次的 decoder 输出。

第二条是省算力的关键：绝大多数帧都是 blank，只有真出新 token 才需要跑一次
decoder 前向。写出"每帧都刷新"不会报错，只是白白慢好几倍。

假设 hyp 的初始值是 context_size 个 blank，最终结果要把开头这 context_size
个去掉（它们是占位，不是识别出来的内容）。

============================================================
三、跨块必须存活的状态
============================================================
    1. encoder 的全部状态张量（实测 116 个）—— 除 reset() 外绝不清零；
    2. hyp（已识别出的 token 序列）；
    3. 最近一次的 decoder 输出。

============================================================
四、收尾：尾部补 100 帧特征域 0
============================================================
由前端（AudioPreprocess.input_finished）在特征尾部补 100 帧 0 完成，本模块
不需要额外处理。补的 0 是 log 域对应能量 1 的 0，不是静音 —— 目的是让最后
不满一个 chunk 的尾段也能凑齐、被解码到，否则尾字会丢。

依赖：只用 numpy + onnxruntime。
"""

import os
import re

import numpy as np
import onnxruntime as ort


# ============================================================
# 一、词表与权重定位（模块级小工具）
# ============================================================
def load_tokens(path):
    """读 tokens.txt，返回 {id: token}。

    每行的格式是 `token id`（token 在前、id 在后）。用 rpartition 而不是
    split()：BPE token 里用 ▁ 表示词首空格，本身不含空格，但 rpartition
    对这种格式更稳，遇到多余空白的行也不会崩。
    """
    id2token = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            token, sep, idx = line.rpartition(" ")
            if not sep:
                continue
            id2token[int(idx)] = token
    return id2token


def find_blank(id2token):
    """在词表里找出 blank 的 id。找不到才退回 0 并提示。

    不要硬编码 0：blank 的 id 是模型定的，换个词表就可能变。
    """
    for i, t in id2token.items():
        if t in ("<blk>", "<blank>"):
            return i
    print("警告：tokens.txt 里没找到 <blk>/<blank>，退回 blank_id=0")
    return 0


# 文本还原用的字符区间：
#   _CJK      —— 统一表意文字 + 扩展 A
#   _FULLWIDTH —— CJK 标点 + 全角形
_CJK = r"一-鿿㐀-䶿"
_FULLWIDTH = r"　-〿＀-￯"


def detokenize(tokens):
    """把 BPE token 序列拼成可读文本（后处理）。

    不能只做 "".join()，原因有两层：

        1. ▁ 是 BPE 的词首标记，得还原成空格，否则得到「▁昨▁天▁是」。
        2. 本模型的词表把中文按单字切（▁礼▁拜▁二），单把 ▁ 换成空格会散成
           「礼 拜 二」。所以还得去空格，但只能去特定位置：
              - CJK 之间           去   「昨 天」 -> 「昨天」
              - 全角标点两侧       去   「Monday ，today」 -> 「Monday，today」
              - CJK 与 ASCII 之间  保留 「是 Monday」不能粘成「是Monday」
    """
    text = "".join(tokens).replace("▁", " ")
    text = re.sub(r"(?<=[%s])\s+(?=[%s])" % (_CJK, _CJK), "", text)
    text = re.sub(r"\s+(?=[%s])" % _FULLWIDTH, "", text)
    text = re.sub(r"(?<=[%s])\s+" % _FULLWIDTH, "", text)
    return text.strip()


def find_onnx(model_dir, stem):
    """在权重目录里定位一个 onnx 文件：优先 int8，再退回 fp32。"""
    for name in ("%s.int8.onnx" % stem, "%s.onnx" % stem):
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("在 %s 下找不到 %s.int8.onnx 或 %s.onnx"
                            % (model_dir, stem, stem))


# ============================================================
# 二、解码器
# ============================================================
class AsrDecoder:
    """三段 onnx 会话 + 流式贪心搜索。

    用法::

        dec = AsrDecoder("weight/xxx-2026-06-05")
        while 还有特征:
            text = dec.accept_features(feats)   # 喂一块 (n, 80)，返回当前完整文本

    也可以绕开 accept_features，自己按 run_encoder / run_decoder / run_joiner
    逐个调用 —— 那三个方法正是 C++ 侧三个 session 的一一对应物。
    """

    # onnx 张量类型 -> numpy dtype，用来给状态张量选初值类型
    _DTYPES = {"tensor(float)": np.float32,
               "tensor(int64)": np.int64,
               "tensor(int32)": np.int32}

    def __init__(self, model_dir, sample_rate=16000):
        """
        参数:
            model_dir   —— 权重目录，内含 encoder/decoder/joiner 三个 onnx
                           与 tokens.txt
            sample_rate —— 采样率，目前只用于记录（模型固定 16kHz）
        """
        model_dir = os.path.abspath(model_dir)
        tok_path = os.path.join(model_dir, "tokens.txt")
        if not os.path.isfile(tok_path):
            raise FileNotFoundError("找不到词表：%s" % tok_path)
        self.model_dir = model_dir
        self.sample_rate = int(sample_rate)

        # ---- 三个会话 ----
        # 单线程 + 静音日志：这是实时语音场景，多个 chunk 之间本来就串行，
        # 让 onnxruntime 再开线程池只会增加调度开销。
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        so.log_severity_level = 3

        def _sess(path):
            return ort.InferenceSession(path, so,
                                        providers=["CPUExecutionProvider"])

        self.encoder = _sess(find_onnx(model_dir, "encoder"))
        self.decoder = _sess(find_onnx(model_dir, "decoder"))
        self.joiner = _sess(find_onnx(model_dir, "joiner"))

        # ---- 从 metadata 读尺寸，不硬编码 ----
        meta = self.encoder.get_modelmeta().custom_metadata_map
        model_type = meta.get("model_type", "")
        if model_type != "zipformer2":
            raise RuntimeError("encoder 的 model_type=%r，本解码循环只支持 "
                               "zipformer2" % model_type)
        self.T = int(meta["T"])                              # 输入帧数，如 61
        self.shift = int(meta["decode_chunk_len"])           # 每轮前移，如 48

        dmeta = self.decoder.get_modelmeta().custom_metadata_map
        self.context_size = int(dmeta["context_size"])
        self.vocab_size = int(dmeta["vocab_size"])

        # ---- 缓存输入输出名字，免得解码循环里每帧都做一次模型反射查询 ----
        self._enc_in = [n.name for n in self.encoder.get_inputs()]
        self._enc_out = [n.name for n in self.encoder.get_outputs()]
        self._dec_in = self.decoder.get_inputs()[0]
        self._dec_out = self.decoder.get_outputs()[0].name
        self._joi_in = [n.name for n in self.joiner.get_inputs()]
        self._joi_out = self.joiner.get_outputs()[0].name

        # 输入特征维度：encoder 第一个输入是 (batch, T, n_mels)
        # 最后一维正常是具体的整数；万一是符号维度（字符串）就退回 80
        x_shape = self.encoder.get_inputs()[0].shape
        last = x_shape[-1]
        self.n_mels = int(last) if not isinstance(last, str) else 80

        # ---- 状态张量的原型：形状与 dtype 全部从 encoder 输入签名推 ----
        # 签名里的字符串维度（如 'N'）按 batch=1 记 1。这样换模型不用改代码。
        self._state_proto = []
        for n in self.encoder.get_inputs()[1:]:
            if n.type not in self._DTYPES:
                raise RuntimeError("状态张量 %s 的类型 %s 未支持"
                                   % (n.name, n.type))
            shape = [1 if isinstance(s, str) else int(s) for s in n.shape]
            self._state_proto.append((shape, self._DTYPES[n.type]))

        # ---- 词表 ----
        self.id2token = load_tokens(tok_path)
        self.blank_id = find_blank(self.id2token)

        self.reset()

    # --------------------------------------------------------
    def reset(self):
        """清空全部跨块状态，回到「刚开始听」。"""
        self.states = [np.zeros(shape, dtype=dt)
                       for shape, dt in self._state_proto]
        self.hyp = [self.blank_id] * self.context_size
        self.decoder_out = self.run_decoder(self.hyp)
        self._feat_buf = np.zeros((0, self.n_mels), dtype=np.float32)
        self.n_chunks = 0        # 喂过几次 encoder
        self.n_enc_frames = 0    # encoder 一共吐了多少帧
        self._text = ""

    # --------------------------------------------------------
    # 属性
    # --------------------------------------------------------
    @property
    def text(self):
        """当前识别文本。"""
        return self._text

    @property
    def token_ids(self):
        """已确定的 token id，**不含**开头那 context_size 个占位 blank。"""
        return self.hyp[self.context_size:]

    def info(self):
        """把关键尺寸打成一包，方便入口脚本打印。"""
        return {"T": self.T, "decode_chunk_len": self.shift,
                "context_size": self.context_size,
                "vocab_size": self.vocab_size,
                "n_states": len(self.states), "blank_id": self.blank_id,
                "n_mels": self.n_mels}

    # --------------------------------------------------------
    # 三个 session：C++ 侧逐个对应的低层接口
    # --------------------------------------------------------
    def run_encoder(self, x):
        """喂 (1, T, n_mels) 特征，返回 encoder 输出；状态就地累积到 self.states。

        注意返回值是 out[0]（编码结果），out[1:] 是更新后的状态张量，
        必须原样存回去 —— 这就是流式能记住前文的全部机制。
        """
        feed = {self._enc_in[0]: x}
        for i, name in enumerate(self._enc_in[1:]):
            feed[name] = self.states[i]
        out = self.encoder.run(self._enc_out, feed)
        self.states = out[1:]
        return out[0]

    def run_decoder(self, hyp):
        """喂最近 context_size 个 token id，返回 decoder（predictor）输出。"""
        dtype = (np.int64 if self._dec_in.type == "tensor(int64)" else np.int32)
        y = np.array([hyp], dtype=dtype)
        return self.decoder.run([self._dec_out], {self._dec_in.name: y})[0]

    def run_joiner(self, encoder_out, decoder_out):
        """把 encoder 一帧与 decoder 输出合起来，返回整个词表的打分。"""
        return self.joiner.run([self._joi_out],
                               {self._joi_in[0]: encoder_out,
                                self._joi_in[1]: decoder_out})[0]

    # --------------------------------------------------------
    # 一站式：喂特征，边攒边解码
    # --------------------------------------------------------
    def accept_features(self, feats):
        """喂一块特征，把能解码的都解掉，返回当前完整文本。

        参数:
            feats —— float32 特征矩阵 (n, n_mels)，可能是 0 帧
        返回:
            当前识别文本（没有新 token 时与上次相同）
        """
        if feats.shape[0]:
            self._feat_buf = (np.concatenate((self._feat_buf, feats))
                              if self._feat_buf.shape[0] else feats)

        n_new = 0
        # 判据必须是 > T（**严格大于**），不能写成 >= T。
        # 批量版每次条件是 start + T < 总帧数，即 48k + 61 < L，等价于
        # L - 48k > 61。写成 >= 会在帧数恰好等于 48k + 61 时多喂一轮，
        # 尾部多吐出 token。
        while self._feat_buf.shape[0] > self.T:
            # 取最前面 T 帧给 encoder
            x = np.ascontiguousarray(self._feat_buf[:self.T][None])
            # 时间轴只前进 shift 帧，多出来的那十几帧是下一轮的重叠上下文
            self._feat_buf = np.ascontiguousarray(self._feat_buf[self.shift:])
            self.n_chunks += 1

            enc = self.run_encoder(x)          # 状态在 self.states 里累积
            nf = enc.shape[1]                  # **运行时**取输出帧数
            self.n_enc_frames += nf

            for k in range(nf):
                cur = enc[0, k:k + 1]          # encoder 的第 k 帧
                tid = int(self.run_joiner(cur, self.decoder_out).argmax())
                if tid != self.blank_id:
                    # 非 blank：接进假设，并刷新 decoder
                    self.hyp.append(tid)
                    n_new += 1
                    self.decoder_out = self.run_decoder(
                        self.hyp[-self.context_size:])
                # blank：什么都不做，沿用上一次的 decoder 输出（省算力的关键）

        if n_new:
            self._text = detokenize([self.id2token.get(i, "?")
                                     for i in self.token_ids])
        return self._text
