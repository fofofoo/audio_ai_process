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
    2) 建三个 onnx session build_ort_sessions()（路径写在文件顶部）
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
#   不显式插一次，下面的 from mini.* 会直接 ImportError
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from mini.audio_preprocess import AudioPreprocess
from mini.asr_decode import AsrDecoder
from mini.mic_client import MicClient, list_devices
import onnxruntime as ort

# onnxruntime 推理用cpu或gpu
PROVIDERS = ["CPUExecutionProvider"]    #"CUDAExecutionProvider",

# 麦克风设备：None = 系统默认输入设备；也可以写设备编号（如 1）或设备名
MIC_DEVICE = None


def build_ort_sessions(ENCODER_ONNX,DECODER_ONNX,JOINER_ONNX):
    """按写死的三条路径建三个 onnx session，返回 (encoder, decoder, joiner)。

    建会话放在入口这一层、而不是 AsrDecoder 内部：「模型从哪来、跑在哪块
    硬件上」是运行环境的决定，不是解码逻辑 —— 解码类只接收现成的 session。
    """

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


def setup_stdout_utf8():
    """把 stdout/stderr 切成 UTF-8。

    Windows 控制台默认 GBK，不转的话打印中文会乱码或直接抛
    UnicodeEncodeError（重定向到文件或管道时尤其明显）。
    """
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:      # 极老的 Python 才没有 reconfigure
            pass


def main():
    setup_stdout_utf8()

    # ========================================================
    # 1.连接麦克风
    # ========================================================
    # list_devices()        # 想看本机有哪些设备，把这一行取消注释
    device = normalize_device(MIC_DEVICE)
    # 初始化麦克风客户端
    mic = MicClient(sample_rate=16000, device=device)
   
    # ========================================================
    #  2.加载三个onnx模型，初始化预处理和推理实例
    # ========================================================
    #  asr各个模型路径
    _ASR_DIR = (r"weight\sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05")
    ENCODER_ONNX = os.path.join(_ASR_DIR, "encoder.int8.onnx")
    DECODER_ONNX = os.path.join(_ASR_DIR, "decoder.onnx")
    JOINER_ONNX = os.path.join(_ASR_DIR, "joiner.int8.onnx")
    TOKENS_TXT = os.path.join(_ASR_DIR, "tokens.txt")
    # 2.1 加载onnx模型，返回三个session
    enc, dec_sess, joi = build_ort_sessions(ENCODER_ONNX, DECODER_ONNX, JOINER_ONNX)
    # 2.2 初始化asr推理器，组合三个session和词表的语言转文字工作流水线
    dec = AsrDecoder(enc, dec_sess, joi, TOKENS_TXT)
    # 2.3 初始化预处理类，将音频转换为特征，输入asr模型，n_mels 跟着模型走，避免两边维度对不上
    front = AudioPreprocess(sample_rate=16000, n_mels=dec.n_mels)

    # ========================================================
    # 3.打印模型信息、预处理参数、麦克风参数
    # ========================================================  
    info = dec.info()
    print("=" * 74)
    print("mini 流式 ASR —— 预处理 / 解码 / 麦克风 三条流水线")
    print("=" * 74)
    print("权重  : %s" % _ASR_DIR)
    print("onnx  : encoder=%s  decoder=%s  joiner=%s" % (os.path.basename(ENCODER_ONNX), os.path.basename(DECODER_ONNX), os.path.basename(JOINER_ONNX)))
    # 执行器：硬编码 CUDA 优先、CPU 兜底
    print("执行器: %s" % " -> ".join(p.replace("ExecutionProvider", "") for p in info["providers"]))
    print("模型  : T=%d(输入帧数)  decode_chunk_len=%d(每轮前移)  ""context_size=%d  vocab=%d  状态张量=%d"
          % (info["T"], info["decode_chunk_len"], info["context_size"], info["vocab_size"], info["n_states"]))
    print("预处理端  : %d Hz  帧长 %d 点(%.0fms)  帧移 %d 点(%.0fms)  n_mels=%d"
          % (front.sample_rate, front.frame_size, front.frame_size / front.sample_rate * 1e3,  front.frame_step, front.frame_step / front.sample_rate * 1e3,  front.n_mels))
    print("词表  : %d 个  blank_id=%d (%s)"% (len(dec.id2token), dec.blank_id, dec.id2token.get(dec.blank_id, "?")))
    print()

    # ========================================================
    # 4.主循环：取音频 -> 预处理 -> 推理 -> 后处理 -> 打印
    # ========================================================
    # 4.1 打开麦克风
    mic.start()
    print("麦克风已开：device=%s  每块 %d 样本 (%.1f ms)  缓冲上限 %d 块"
            % (device if device is not None else "系统默认", mic.block, mic.block / mic.sample_rate * 1e3, mic.max_blocks))
    # 4.2 开始语音转文字工作流
    shown = ""
    t_start = time.perf_counter()
    print("=" * 74)
    print("识别结果(ctrl+C 退出)：" )
    try:
        while True:
            # 4.1 取音频
            samples = mic.get_last_audio()
            if samples is None:
                time.sleep(0.002)      # 还没攒到新数据，让出 CPU 别空转
                continue
            # 4.2 预处理
            feats = front.accept_waveform(samples)   
            # 4.3 推理 + 后处理
            text = dec.accept_features(feats)       
            # 4.4 打印：只在变化时刷新
            if text != shown:                        
                shown = text
                sys.stdout.write("\r%s" % text.ljust(72))
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass

    # ========================================================
    # 5.结束，关闭麦克风，统计结果
    # ========================================================
    # 5.1 关闭麦克风
    mic.stop()
    # 5.2 收尾：关流后缓冲里可能还压着几块，先取走解完；再让前端补出尾部帧、
    #     并在特征尾补 100 帧 0（log 域能量 1，不是静音值），使最后不满一个
    #     chunk 的尾段也能被解码 —— 少了这一步，最后几个字会被整段丢掉
    leftover = mic.get_last_audio()
    if leftover is not None:
        dec.accept_features(front.accept_waveform(leftover))
    text = dec.accept_features(front.input_finished())
    # 5.3 正常退出，统计结果
    dur = time.perf_counter() - t_start
    print("\n")
    print("=" * 74)
    print("识别结果：%s" % text)
    print("=" * 74)
    print("时长 %.1fs   喂入样本 %d   产出帧 %d" % (dur, front.n_samples, front.n_frames))
    print("encoder 调用 %d 次 / 共 %d 帧   token %d 个"% (dec.n_chunks, dec.n_enc_frames, len(dec.token_ids)))
    print("丢块 %d   峰值幅度 %.3e" % (mic.dropped, mic.peak))
    # 5.4 try异常退出提示
    if mic.dropped:
        print("提示：出现丢块说明主循环跟不上采集速度，可调大 MicClient 的 "
              "block 参数（在 mic_client.py 里）。")
    if mic.peak == 0.0:
        print("提示：峰值幅度为 0，麦克风可能没采到声音，"
              "可改顶部 MIC_DEVICE 指定别的设备；"
              "取消第 1 节里 list_devices() 那行的注释即可列出本机设备。")


if __name__ == "__main__":
    main()
