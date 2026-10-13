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
    2) 读取 onnx 模型      AsrDecoder(model_dir) 内部建三个 session
    3) 预处理              front.accept_waveform(samples)
    4) 模型推理 + 5) 后处理 dec.accept_features(feats)
    6) 打印结果            stdout 原地刷新

结束（Ctrl+C）后还要走一次收尾：补完尾部帧、并在特征尾部补 100 帧 0，
再把剩下的一起解掉 —— 否则最后几个字会丢。

用法：
    python -I test_mic_asr_mini.py                      # 自动找权重，默认麦克风
    python -I test_mic_asr_mini.py --seconds 8          # 跑 8 秒自动退出
    python -I test_mic_asr_mini.py --list-devices       # 列出音频设备
    python -I test_mic_asr_mini.py --model-dir <目录> --device 1
"""

import argparse
import glob
import os
import sys
import time

# -I 隔离模式下脚本目录不进 sys.path（也忽略 PYTHONPATH），
#   这里显式把 mini/ 插进去，才能 import 那三条流水线
_HERE = os.path.dirname(os.path.abspath(__file__))
_MINI = os.path.join(_HERE, "mini")
if _MINI not in sys.path:
    sys.path.insert(0, _MINI)

from audio_preprocess import AudioPreprocess
from asr_decode import AsrDecoder
from mic_client import MicClient

# 权重默认放在项目根的 weight/ 下（目录名带日期后缀，不能写死，要靠 find_model_dir 找）
WEIGHT_ROOT = os.path.join(_HERE, "weight")


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


def find_model_dir(root=WEIGHT_ROOT):
    """在权重根目录下自动找出权重目录。

    判据：目录里**同时**有 tokens.txt 和 encoder*.onnx。
    不写死目录名 —— 这批权重从 ModelScope 下载，带日期后缀，换版本就变。
    """
    if not os.path.isdir(root):
        raise FileNotFoundError("权重根目录不存在：%s" % root)
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        if (os.path.isfile(os.path.join(d, "tokens.txt"))
                and glob.glob(os.path.join(d, "encoder*.onnx"))):
            return d
    raise FileNotFoundError(
        "%s 下没找到权重目录（需要同时有 tokens.txt 和 encoder*.onnx）" % root)


def parse_args():
    ap = argparse.ArgumentParser(
        description="mini 版实时麦克风语音识别（预处理 / 解码 / 麦克风 三条流水线）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model-dir", default=None,
                    help="权重目录；不给则自动在 weight/ 下找")
    ap.add_argument("--device", default=None,
                    help="输入设备编号或名字；不给则用系统默认麦克风")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="跑几秒后自动退出；0 表示不限时（Ctrl+C 结束）")
    ap.add_argument("--list-devices", action="store_true",
                    help="列出本机音频设备后退出")
    return ap.parse_args()


def main():
    setup_stdout_utf8()
    args = parse_args()

    if args.list_devices:
        from mic_client import list_devices
        list_devices()
        return

    # 设备号既可以是编号（"1"）也可以是名字，能转成整数就转
    device = args.device
    if device is not None:
        try:
            device = int(device)
        except ValueError:
            pass

    model_dir = os.path.abspath(args.model_dir) if args.model_dir \
        else find_model_dir()

    # ========================================================
    # 按顺序把三条流水线接起来
    # ========================================================
    # 1) 连接麦克风（此时还没开始采集，start() 之后才有数据）
    mic = MicClient(sample_rate=16000, device=device)

    # 2) 读取 onnx 模型（三个 session + 词表；尺寸全部从模型签名推）
    dec = AsrDecoder(model_dir)

    # 预处理类：n_mels 跟着模型走，避免两边维度对不上
    front = AudioPreprocess(sample_rate=16000, n_mels=dec.n_mels)

    info = dec.info()
    print("=" * 74)
    print("mini 流式 ASR —— 预处理 / 解码 / 麦克风 三条流水线")
    print("=" * 74)
    print("权重  : %s" % os.path.basename(model_dir))
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
    if args.seconds:
        print("限时 %g 秒。\n" % args.seconds)
    else:
        print("说话即可看到文字；Ctrl+C 结束。\n")

    # ========================================================
    # 主循环：取音频 -> 预处理 -> 推理 -> 后处理 -> 打印
    # ========================================================
    shown = ""
    t_start = time.perf_counter()
    try:
        while True:
            if args.seconds and time.perf_counter() - t_start >= args.seconds:
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
        print("提示：出现丢块说明主循环跟不上采集速度，可调大 --device 之外的 "
              "block（改 MicClient 的 block 参数）。")
    if mic.peak == 0.0:
        print("提示：峰值幅度为 0，麦克风可能没采到声音，"
              "可用 --list-devices 换个设备试试。")


if __name__ == "__main__":
    main()
