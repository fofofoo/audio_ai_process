# -*- coding: utf-8 -*-
"""离线 wav 语音识别 —— mini 版入口（与 test_mic_asr_mini.py 同构）。

============================================================
这份代码是干什么的
============================================================
把 mini/ 下的流水线按顺序接起来，读一个 wav 文件从头解到尾。
和 test_mic_asr_mini.py 的区别只有一个 —— 音频从**文件**来，不是从麦克风来：

    test_mic_asr_mini.py   麦克风 -> 裸音频块（MicClient）
    test_wav_asr_mini.py   wav 文件 -> 裸音频块（本脚本自己切）

预处理与解码两条流水线原封不动（同一个 AudioPreprocess / AsrDecoder），
连每次喂入的块长都取和 MicClient 默认值一样的 160 样本（10ms），所以同一段
音频走这两条路径，结果应当完全一致 —— 这也是本脚本的另一个用途：互为对照。

============================================================
调用顺序（对照 main 里的编号）
============================================================
    1) 读 wav              sf.read -> float32 单声道
    2) 建三个 onnx session build_sessions()（路径写在文件顶部）
       装配解码器          AsrDecoder(encoder, decoder, joiner, tokens_path)
    3) 预处理              front.accept_waveform(block)
    4) 模型推理 + 5) 后处理 dec.accept_features(feats)
    6) 打印结果            stdout 原地刷新，看得到文字一个个蹦出来

wav 喂完之后还有一次收尾：front.input_finished() 补出尾部帧、并在特征尾部
补 100 帧 0，再把剩下的解掉 —— 不补则最后几个字会丢。

不模拟实时节奏（不按音频时长 sleep）：wav 是离线场景，尽快跑完更有用；
想看流式效果，看 stdout 那个逐字刷新的过程就够了。

用法：
    python -I test_wav_asr_mini.py

不收命令行参数 —— wav 路径与模型路径全部硬编码在文件顶部「运行配置」一节。
"""

import os
import sys
import time

# -I 隔离模式下脚本目录不进 sys.path（也忽略 PYTHONPATH），
#   这里显式把 mini/ 插进去，才能 import 那几条流水线
_HERE = os.path.dirname(os.path.abspath(__file__))
_MINI = os.path.join(_HERE, "mini")
if _MINI not in sys.path:
    sys.path.insert(0, _MINI)

from audio_preprocess import AudioPreprocess
from asr_decode import AsrDecoder


# ============================================================
# 运行配置 —— 全部硬编码在这里，本脚本不收命令行参数
# ============================================================
# 权重目录（与 test_mic_asr_mini.py 用的是同一套），三个 onnx 路径全部写死。
# weight/ 下有两套权重，换哪套就把 _ASR_DIR 连文件名一起改：
#   fp32 版（当前用的，encoder 593MB）：encoder.onnx      / joiner.onnx
#   int8 版（小、快）：                 encoder.int8.onnx / joiner.int8.onnx
_ASR_DIR = (r"c:\Users\Liyun\Desktop\audio_ai_process\weight"
            r"\sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer"
            r"-zh-en-punct-2026-06-05")

ENCODER_ONNX = os.path.join(_ASR_DIR, "encoder.onnx")
DECODER_ONNX = os.path.join(_ASR_DIR, "decoder.onnx")
JOINER_ONNX = os.path.join(_ASR_DIR, "joiner.onnx")
TOKENS_TXT = os.path.join(_ASR_DIR, "tokens.txt")

# 要识别的 wav。权重包里自带 test_wavs/0.wav ~ 3.wav，换文件名即可换样本。
WAV_PATH = os.path.join(_ASR_DIR, "test_wavs", "0.wav")

# 执行器：固定 CPU。CUDAExecutionProvider 要的 cuDNN 版本与本机对不上，写上去
# 只会每次启动刷一屏加载失败的报错、再自己退回 CPU，不如直接写 CPU 干净。
PROVIDERS = ["CPUExecutionProvider"]

# 每次喂给前端的样本数。160 = 10ms @16kHz，与 MicClient 的默认 block 相同，
# 这样「文件路径」与「麦克风路径」的喂入节奏完全一致，两边结果可以互相对照。
BLOCK = 160


def setup_stdout_utf8():
    """把 stdout/stderr 切成 UTF-8。

    Windows 控制台默认是 GBK，不转的话打印中文会乱码或直接抛
    UnicodeEncodeError。reconfigure 是 Python 3.7+ 的标准做法。
    """
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:      # 极老的 Python 才没有 reconfigure
            pass


def load_wav(path):
    """读 wav，返回 (float32 单声道样本, 采样率)。

    多声道只取第一声道（模型是单声道模型）。这里**不做重采样** —— 采样率
    不对时宁可明确报出来，也不要悄悄喂进错误的数据。
    """
    import soundfile as sf

    if not os.path.isfile(path):
        raise FileNotFoundError("wav 不存在：%s" % path)
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return data[:, 0], int(sr)


def build_sessions():
    """按写死的三条路径建三个 onnx session，返回 (encoder, decoder, joiner)。

    建会话放在入口这一层、而不是 AsrDecoder 内部：「模型从哪来、跑在哪块
    硬件上」是运行环境的决定，不是解码逻辑 —— 解码类只接收现成的 session。
    """
    # 延迟到用时才 import，让本模块顶部的配置阶段不必加载 onnxruntime
    import onnxruntime as ort

    # 单线程 + 静音日志：多个 chunk 之间本来就串行，
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


def main():
    setup_stdout_utf8()

    # ========================================================
    # 1) 读 wav
    # ========================================================
    samples, sr = load_wav(WAV_PATH)
    duration = len(samples) / float(sr)

    # ========================================================
    # 2) 建三个 onnx session（路径写死在文件顶部），再装进解码器
    # ========================================================
    enc, dec_sess, joi = build_sessions()
    dec = AsrDecoder(enc, dec_sess, joi, TOKENS_TXT)

    # 预处理类：n_mels 跟着模型走，避免两边维度对不上
    front = AudioPreprocess(sample_rate=sr, n_mels=dec.n_mels)

    info = dec.info()
    print("=" * 74)
    print("mini 离线 wav ASR —— 文件 -> 特征 -> 文字")
    print("=" * 74)
    print("wav   : %s" % WAV_PATH)
    print("        %.2f 秒 / %d 样本 @ %d Hz" % (duration, len(samples), sr))
    print("onnx  : encoder=%s  decoder=%s  joiner=%s"
          % (os.path.basename(ENCODER_ONNX), os.path.basename(DECODER_ONNX),
             os.path.basename(JOINER_ONNX)))
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
    if sr != front.sample_rate:
        print("注意：wav 是 %d Hz，前端按 %d Hz 算，结果会不对。\n"
              % (sr, front.sample_rate))

    # ========================================================
    # 3~6) 主循环：切块 -> 预处理 -> 推理 -> 后处理 -> 打印
    # ========================================================
    shown = ""
    t_start = time.perf_counter()
    for i in range(0, len(samples), BLOCK):
        blk = samples[i:i + BLOCK]               # 取一块（末尾那块可能不满）
        feats = front.accept_waveform(blk)       # 3) 预处理
        text = dec.accept_features(feats)        # 4) 推理 + 5) 后处理

        if text != shown:                        # 6) 打印：只在变化时刷新
            shown = text
            sys.stdout.write("\r%s" % text.ljust(72))
            sys.stdout.flush()

    # ========================================================
    # 收尾：把最后一点音频解完，否则尾字会丢
    # ========================================================
    # input_finished() 会补出尾部帧，并在特征尾部补 100 帧 0（log 域能量 1，
    # 不是静音）—— 让最后不满一个 chunk 的尾段也能被解码到
    text = dec.accept_features(front.input_finished())

    dur = time.perf_counter() - t_start
    print("\n")
    print("=" * 74)
    print("识别结果：%s" % text)
    print("=" * 74)
    print("音频 %.2fs   耗时 %.2fs   RTF %.3f（< 1 即快于实时）"
          % (duration, dur, dur / duration if duration else 0.0))
    print("喂入样本 %d   产出帧 %d" % (front.n_samples, front.n_frames))
    print("encoder 调用 %d 次 / 共 %d 帧   token %d 个"
          % (dec.n_chunks, dec.n_enc_frames, len(dec.token_ids)))


if __name__ == "__main__":
    main()
