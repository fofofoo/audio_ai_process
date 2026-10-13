# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言与协作约定

- 始终用中文交流；代码注释、docstring 沿用项目现有的中文风格（README 与 doc/ 均为中文）。
- **执行脚本前必须先征得用户同意**。写验证脚本、跑解码/冒烟测试/数值一致性比对之前，先给出打算执行的内容与目的，用户明确同意后再动手。`scripts/` 下已有的验证脚本同样适用。

## 项目是什么

中文流式语音识别（streaming zipformer2 transducer，int8）的**纯 Python 复现**。

不依赖 sherpa-onnx 运行时：预处理用自研的传统前端（`numpy` + `scipy`）产出模型输入，推理与后处理用 `onnxruntime`，最后接麦克风做边听边出字。运行时依赖只有 `onnxruntime` + `numpy` + `soundfile`（GUI 另需 `pyqtgraph`/`PyQt5`/`sounddevice`/`pandas`）。

`kaldi_native_fbank` 只在验证时充当「数值标尺」，不是运行时依赖；`sharpa-onnx/` 是从上游 clone 的 C++ 源码副本，只作阅读参考，不参与运行。

环境：Python 3.10，conda 环境 `py310-gpu`。

## 常用命令

```bash
# 入口①：离线样本回放（播放 + 三块图 + 流式识别）
python test_spectrogram.py

# 入口②：实时麦克风（三块图 + 流式识别）
python realtime_mic_monitor.py

# 无界面跑一次识别；--frontend ours|knf|both
python -I src/asr_onnx_decode.py --model-dir weight/<权重目录> --frontend both

# 验证脚本（都不需要人工看界面）
python -I scripts/smoke_test_spectrogram.py     # 离线链路 offscreen 冒烟，含帧间隔统计
python -I scripts/smoke_test_mic_audio.py       # 假 InputStream 按 1x 速度回放真实 wav
python -I scripts/smoke_test_mic_monitor.py     # 真开麦克风跑 N 秒（默认 8s）后自动退出
python -I scripts/probe_align_features.py <wav> ...
python -I scripts/probe_streaming_features.py <wav> ...
python -I scripts/probe_mel_iterate.py          # numpy 复现 InitKaldiMelBanks 并逐点比对

# 模块自测（合成信号跑通流水线，检查输出形状与口径）
python -I src/traditional_features.py
python -I src/streaming_features.py
```

## 架构

```
PortAudio 回调线程 ──audio_q(Queue)──> Qt 主线程 ──worker.feed()──> AsrWorker 识别线程
   只做 .copy()+入队                      画图 + 转交                     解码，写出 worker.text
```

三条线的边界是硬约束：回调线程绝不碰 Qt、绝不碰模型；识别线程把整串结果赋给 `worker.text`，主线程直接读（整串赋值，GIL 保证读不到半截，不加锁）。识别必须单开线程——喂一次 encoder 最坏约 40 ms，而画面每 25 ms 推一帧。队列满时丢**最旧**的块。

分层（自下而上，上层复用下层，改一处两边生效）：

| 模块 | 职责 |
|---|---|
| `src/traditional_features.py` | 自研批量前端，逐项对齐 kaldi 口径；`frontend_for_onnx()` 是喂模型的入口 |
| `src/streaming_features.py` | `StreamingFramer` 只接管「分帧」这一层，帧内处理全部复用上者；`StreamingLogFBank` = 分帧 + 帧内处理 |
| `src/asr_onnx_decode.py` | `OnnxModel` 三个 session + 贪心搜索 + 文本还原；也是 CLI 入口 |
| `src/streaming_asr.py` | `StreamingAsr` 增量解码（复用 `OnnxModel`）、`AsrWorker` GUI 后台线程、麦克风采集、`find_model_dir` |
| `src/live_audio_view.py` | 三块图（波形 / RMS / 滚动语谱图）控件，两个 GUI 脚本共用 |
| `src/console_utf8.py` | 把 stdout/stderr 切成 UTF-8，单独拆出来是为了让 `scripts/` 也能用而不拖进 onnxruntime |

两个 GUI 入口（`test_spectrogram.py`、`realtime_mic_monitor.py`）都只是「配置 + 数据源」，识别类与绘图控件各只有一份实现。

### 必须记住的不变量

这几条写错了不会报错，只会静默多吐出几个 token 或尾字丢失：

- **`T = 61` 是输入帧数，不是输出帧数**。时间轴每次只前进 `decode_chunk_len = 48` 帧（480 ms），多出的 13 帧是卷积模块的重叠上下文；实测每个 chunk 输出 **12** 帧。（`T` / `decode_chunk_len` / `context_size` 都由 ONNX 输入签名自动推导，上例是当前权重的取值，代码里没有硬编码。）
- **`encoder_out.shape[1]` 必须取运行时值**，写死 `T` 会错。
- **blank 不刷新 decoder**（贪心搜索省算力的关键）；`blank_id` 从 `tokens.txt` 查 `<blk>`，不硬编码 0。
- **解码循环尾部补 100 帧特征域的 0**（约 1 秒）。补的 0 是 log 域对应能量 1，不是静音值，照抄官方做法。不补则最后一段凑不满 chunk 会被整段丢掉。
- **流式喂 encoder 的判据是 `len(buf) > T`（严格大于）**。写成 `>=` 会在批量条件恰好 `48k + 61` 时多喂一次。该判据与批量严格等价，推导见 README。
- **116 个 encoder 状态张量 + `hyp` + 最近一次 `decoder_out` 必须跨块存活**，除 `reset()` 外绝不清零。状态初值由 ONNX 输入签名自动推导（字符串维按 batch=1 取 1），换模型不用改代码。
- 文本还原：`▁`(U+2581) 转空格后，CJK 之间、全角标点前后的空格去掉，但 **CJK 与 ASCII 之间的空格要保留**（`是 Monday` 不能粘成 `是Monday`）。

### 前端对齐的既有结论

初版在 hamming 窗下 `max|Δ| = 10.44`，定位到七项差异后逐项修复（`snip_edges=False` 的分帧语义、povey 窗、窗分母 `N-1`、Mel 域遍历整数 bin 的滤波器组、`symmetric` 边界、`log(max(e, eps_f32))` 地板、逐帧预加重），最终 `mean|Δ|` 从 0.92 降到 4×10⁻⁶。**改 `traditional_features.py` 前先读** [工作流报告](doc/sherpa-onnx-ASR预处理及后处理工作流报告.md) 里的完整推导——里面每一项都有实测依据，重写容易把对的地方改错。

`high_freq` 口径仍有分歧未实测：官方 `test_onnx.py` 取 8000 Hz（Nyquist），sherpa-onnx C++ 默认 `-400`（7600 Hz）。当前默认 8000 Hz。

## 工程约定

- **一律用 `python -I` 跑脚本，并在文件里手动把 `src/` 插进 `sys.path`**。`-I` 隔离模式连脚本目录都不进 `sys.path`（且忽略 `PYTHONPATH`），所以每个可执行脚本头部都有 `_SRC`/`_HERE` 插入逻辑，别删。同理也适用于 `scripts/` ——它们需要同时插入项目根与 `src/`。
- **权重目录不硬编码目录名**。用 `streaming_asr.find_model_dir()` 自动定位（判据：目录里同时有 `tokens.txt` 和 `encoder*.onnx`）。权重从 ModelScope 下载，名字带日期后缀，换版本就变。
- **权重包目录按不可信数据处理**：绝不在其目录内执行解释器，脚本、`bpe.model` 一概不碰，只读三个 onnx 与 `tokens.txt`，路径全部由命令行参数传入。
- **打印中文的脚本开头调 `setup_stdout_utf8()`**（Windows 控制台默认 GBK，不转就是乱码）。
- 数据：`data/aishell1_subset/` 的 CSV 是本仓库改过的 **UTF-8** 版（ModelScope 原版是 GBK，直接读会 `UnicodeDecodeError`）；CSV 里的路径相对 `data/`。

## 已知问题 / 待办

- `dither` 语义与 `kaldi_native_fbank` 的默认值口径不同（本项目是高斯噪声标准差，直接作用于输入量纲）；推理一律用 0，流式版直接拒绝非 0。
- `sounddevice` 的设备编号与 `query_devices()` 打印的表对不上，默认路径不受影响。
- 待办：对 `data/` 的aishell1 子集批量解码、与标注比 WER，出准确率与 RTF 基线；可选加 VAD/端点检测。
