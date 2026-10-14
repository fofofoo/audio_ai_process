# -*- coding: utf-8 -*-
"""实时麦克风语音识别 —— mini 版入口（三条流水线串起来）。

============================================================
这份代码是干什么的
============================================================
把 mini/ 下的三条流水线按顺序接起来，做一个能对着麦克风说话的实时识别
程序。不用 Qt、不做可视化，全程用 print 交互，方便看懂整体结构。

三条流水线各管一段，彼此不知道对方的存在：

    mic_client.py       MicClient          声卡 -> 裸音频块
    audio_preprocess.py AudioPreprocess    波形块 -> LogFBank 特征
    asr_decode.py       AsrDecoder         特征 -> 中文文本

主循环干的事非常直白：

    取一块音频 -> 算特征 -> 喂解码器 -> 打印文本，如此往复。

============================================================
调用顺序（对照 main 里的编号）
============================================================
    1) 连接麦克风          mic.start()
    2) 建三个 onnx session build_sessions()（路径写在文件顶部）
       装配解码器          AsrDecoder(encoder, decoder, joiner, tokens_path)
    3) 预处理              front.accept_waveform(samples)
    4) 模型推理 + 5) 后处理 dec.accept_features(feats)
    6) 打印结果            stdout 原地刷新

结束（Ctrl+C）后还要走一次收尾：补完尾部帧、并在特征尾部补 100 帧 0，
再把剩下的一起解掉 —— 否则最后几个字会丢。

用法：
    python -I test_mic_asr_mini.py

不收命令行参数 —— 运行配置全部硬编码在文件顶部的「运行配置」一节：
三个 onnx 与词表的路径、执行器顺序（GPU/CPU）、麦克风设备、限时，都在那里改。
模型怎么建、跑在哪块硬件上，也由入口这一层决定；AsrDecoder 只接收现成的
session，既不定位权重文件也不碰 onnxruntime。
"""

import os
import sys
import time

# -I 隔离模式下脚本目录不进 sys.path（也忽略 PYTHONPATH），
#   这里显式把 mini/ 插进去，才能 import 那三条流水线
_HERE = os.path.dirname(os.path.abspath(__file__))
_MINI = os.path.join(_HERE, "mini")
if _MINI not in sys.path:
    sys.path.insert(0, _MINI)


# ============================================================
# CUDA 运行库搜索路径 —— 必须赶在 import onnxruntime 之前
# ============================================================
# onnxruntime 的 CUDAExecutionProvider 要加载 cudnn64_9.dll，而这个 dll 由
# 同一个环境里的 torch 自带（site-packages/torch/lib），却不在 Windows 的
# DLL 搜索路径上。不补这一步，onnxruntime 会**静默退回 CPU** ——
# 就算 asr_decode.PROVIDERS 里写死了 CUDA 也照样跑在 CPU 上，只是启动时
# 打印的「执行器」那一行会诚实地显示 CPU。
#
# 路径用 sys.prefix 拼（= 当前解释器的环境根，跑 py310-gpu 时就是
# ...\anaconda3\envs\py310-gpu），等价于写死，但换环境不用改。
# 顺序要求：asr_decode 内部 import onnxruntime，所以这段必须在下一段
# import 之前执行，否则 provider dll 已经找过一遍了。
_TORCH_LIB = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
_DLL_HANDLES = []       # add_dll_directory 返回的句柄一被回收目录就失效，存住
if hasattr(os, "add_dll_directory") and os.path.isdir(_TORCH_LIB):
    _DLL_HANDLES.append(os.add_dll_directory(_TORCH_LIB))

from audio_preprocess import AudioPreprocess
from asr_decode import AsrDecoder
from mic_client import MicClient

# ============================================================
# 运行配置 —— 全部硬编码在这里，本脚本不收命令行参数
# ============================================================
# 三个 onnx 的路径全部写死，不扫目录、不自动查找。
# weight/ 下有两套权重，换哪套就把下面四条路径一起改：
#   fp32 版（当前用的，encoder 593MB）：
#     ...-zh-en-punct-2026-06-05\      encoder.onnx      / joiner.onnx
#   int8 版（小、快）：
#     ...-zh-en-punct-int8-2026-06-05\ encoder.int8.onnx / joiner.int8.onnx
# decoder 两套同名，都是 decoder.onnx。
_ASR_DIR = (r"c:\Users\Liyun\Desktop\audio_ai_process\weight"
            r"\sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer"
            r"-zh-en-punct-2026-06-05")

ENCODER_ONNX = os.path.join(_ASR_DIR, "encoder.onnx")
DECODER_ONNX = os.path.join(_ASR_DIR, "decoder.onnx")
JOINER_ONNX = os.path.join(_ASR_DIR, "joiner.onnx")
TOKENS_TXT = os.path.join(_ASR_DIR, "tokens.txt")

# 执行器顺序 —— 硬编码写死，不做运行时探测。
# onnxruntime 从上往下挑第一个能用的：CUDA 排前面 = 优先 GPU，
# CPU 垫底兜底（CUDA 不可用时 onnxruntime 自己会退回 CPU）。
# 想强制纯 CPU，把这行改成 ["CPUExecutionProvider"] 即可。
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# 麦克风设备：None = 系统默认输入设备；也可以写设备编号（如 1）或设备名
MIC_DEVICE = None

# 跑几秒后自动退出；0 表示不限时（Ctrl+C 结束）
RUN_SECONDS = 0.0

# 置 True 则列出本机音频设备后直接退出（不跑识别）
LIST_DEVICES = False


def setup_stdout_utf8():
    """把 stdout/stderr 切成 UTF-8。

    Windows 控制台默认是 GBK，不转的话打印中文会乱码或直接抛
    UnicodeEncodeError。reconfigure 是 Python 3.7+ 的标准做法，
    两行就够，不需要额外的辅助模块。
    """
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:      # 极老的 Python 才没有 reconfigure
            pass


def build_sessions():
    """按写死的三条路径建三个 onnx session，返回 (encoder, decoder, joiner)。

    建会话放在入口这一层、而不是 AsrDecoder 内部：「模型从哪来、跑在哪块
    硬件上」是运行环境的决定，不是解码逻辑 —— 解码类只接收现成的 session。
    """
    # 延迟到用时才 import：LIST_DEVICES 分支根本用不着 onnxruntime
    import onnxruntime as ort

    # 单线程 + 静音日志：这是实时语音场景，多个 chunk 之间本来就串行，
    # 让 onnxruntime 再开线程池只会增加调度开销。
    so = ort.SessionOptions()
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    so.log_severity_level = 3

    def _sess(path):
        if not os.path.isfile(path):
            raise FileNotFoundError("onnx 不存在：%s" % path)
        return ort.InferenceSession(path, so, providers=PROVIDERS)

    return _sess(ENCODER_ONNX), _sess(DECODER_ONNX), _sess(JOINER_ONNX)


def normalize_device(device=MIC_DEVICE):
    """设备号既可以是编号（"1"）也可以是名字，能转成整数就转。"""
    if isinstance(device, str):
        try:
            return int(device)
        except ValueError:
            return device
    return device


def main():
    setup_stdout_utf8()

    if LIST_DEVICES:
        from mic_client import list_devices
        list_devices()
        return

    device = normalize_device()

    # ========================================================
    # 按顺序把三条流水线接起来
    # ========================================================
    # 1) 连接麦克风（此时还没开始采集，start() 之后才有数据）
    mic = MicClient(sample_rate=16000, device=device)

    # 2) 建三个 onnx session（路径全部写死在文件顶部），再装进解码器；
    #    模型尺寸全部由解码器从 session 签名里推，不用手填
    enc, dec_sess, joi = build_sessions()
    dec = AsrDecoder(enc, dec_sess, joi, TOKENS_TXT)

    # 预处理类：n_mels 跟着模型走，避免两边维度对不上
    front = AudioPreprocess(sample_rate=16000, n_mels=dec.n_mels)

    info = dec.info()
    print("=" * 74)
    print("mini 流式 ASR —— 预处理 / 解码 / 麦克风 三条流水线")
    print("=" * 74)
    print("权重  : %s" % _ASR_DIR)
    print("onnx  : encoder=%s  decoder=%s  joiner=%s"
          % (os.path.basename(ENCODER_ONNX), os.path.basename(DECODER_ONNX),
             os.path.basename(JOINER_ONNX)))
    # 执行器：硬编码 CUDA 优先、CPU 兜底；CUDA 不可用时 onnxruntime 会自己
    # 退回 CPU，这里报的是 session 实际用上的，不是上面写死的期望值
    print("执行器: %s" % " -> ".join(
        p.replace("ExecutionProvider", "") for p in info["providers"]))
    print("模型  : T=%d(输入帧数)  decode_chunk_len=%d(每轮前移)  "
          "context_size=%d  vocab=%d  状态张量=%d"
          % (info["T"], info["decode_chunk_len"], info["context_size"],
             info["vocab_size"], info["n_states"]))
    print("前端  : %d Hz  帧长 %d 点(%.0fms)  帧移 %d 点(%.0fms)  n_mels=%d"
          % (front.sample_rate, front.frame_size,
             front.frame_size / front.sample_rate * 1e3,
             front.frame_step, front.frame_step / front.sample_rate * 1e3,
             front.n_mels))
    print("词表  : %d 个  blank_id=%d (%s)"
          % (len(dec.id2token), dec.blank_id,
             dec.id2token.get(dec.blank_id, "?")))
    print()

    mic.start()
    print("麦克风已开：device=%s  每块 %d 样本 (%.1f ms)  缓冲上限 %d 块"
          % (device if device is not None else "系统默认",
             mic.block, mic.block / mic.sample_rate * 1e3, mic.max_blocks))
    if RUN_SECONDS:
        print("限时 %g 秒。\n" % RUN_SECONDS)
    else:
        print("说话即可看到文字；Ctrl+C 结束。\n")

    # ========================================================
    # 主循环：取音频 -> 预处理 -> 推理 -> 后处理 -> 打印
    # ========================================================
    shown = ""
    t_start = time.perf_counter()
    try:
        while True:
            if RUN_SECONDS and time.perf_counter() - t_start >= RUN_SECONDS:
                break

            samples = mic.get_last_audio()
            if samples is None:
                time.sleep(0.002)      # 还没攒到新数据，让出 CPU 别空转
                continue

            feats = front.accept_waveform(samples)   # 3) 预处理
            text = dec.accept_features(feats)        # 4) 推理 + 5) 后处理

            if text != shown:                        # 6) 打印：只在变化时刷新
                shown = text
                sys.stdout.write("\r%s" % text.ljust(72))
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass

    # ========================================================
    # 收尾：把最后一点音频解完，否则尾字会丢
    # ========================================================
    mic.stop()

    leftover = mic.get_last_audio()      # 关流后缓冲里可能还压着几块
    if leftover is not None:
        dec.accept_features(front.accept_waveform(leftover))

    # input_finished() 会补出尾部帧，并在特征尾部补 100 帧 0（log 域能量 1，
    # 不是静音）—— 让最后不满一个 chunk 的尾段也能被解码到
    text = dec.accept_features(front.input_finished())

    dur = time.perf_counter() - t_start
    print("\n")
    print("=" * 74)
    print("识别结果：%s" % text)
    print("=" * 74)
    print("时长 %.1fs   喂入样本 %d   产出帧 %d"
          % (dur, front.n_samples, front.n_frames))
    print("encoder 调用 %d 次 / 共 %d 帧   token %d 个"
          % (dec.n_chunks, dec.n_enc_frames, len(dec.token_ids)))
    print("丢块 %d   峰值幅度 %.3e" % (mic.dropped, mic.peak))
    if mic.dropped:
        print("提示：出现丢块说明主循环跟不上采集速度，可调大 MicClient 的 "
              "block 参数（在 mic_client.py 里）。")
    if mic.peak == 0.0:
        print("提示：峰值幅度为 0，麦克风可能没采到声音，"
              "把本文件顶部的 LIST_DEVICES 改成 True 跑一次可以列出设备，"
              "再用 MIC_DEVICE 指定换一个。")


if __name__ == "__main__":
    main()
