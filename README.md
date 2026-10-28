# audio_ai_process

中文流式语音识别（streaming zipformer2 transducer）的**纯 Python 复现**。

不依赖 sherpa-onnx 运行时：预处理用自研的传统前端（numpy + scipy）产出模型输入，
推理与后处理用 `onnxruntime` 完成，最后接上麦克风做实时的边听边出字。

模型：`sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05`
（int8 量化，中英混合，自带标点）
git archive --format=zip --prefix=audio_ai_process/ -o audio_ai_process_2026-10-14.zip HEAD

---

## 成果速览

| 指标 | 实测结果 |
|---|---|
| 自研前端 vs `kaldi_native_fbank` 参考实现 | `mean\|Δ\|` ≈ 4×10⁻⁶，`max\|Δ\|` ≤ 5.4×10⁻⁴ |
| 两条前端的识别结果 | **4/4 逐 token 完全相同** |
| 流式 vs 批量（分帧层 / 特征层） | **逐位相同**（`max\|Δ\| = 0`） |
| 流式 vs 批量（token 序列） | **4/4 完全一致** |
| `src/` 独立重写 vs `qt/`（前端） | **逐位相同**（`max\|Δ\| = 0`） |
| `src/` 独立重写 vs `qt/`（token 序列） | **4/4 完全一致** |
| 离线实时率 RTF | 0.059 ~ 0.068（比实时快约 16 倍） |
| 实时识别 RTF（`src/` 流式链路） | ≈ 0.10（比实时快约 10 倍） |
| 实时链路峰值负载（单次喂 encoder / 喂间隔） | **6 ~ 9%** |
| 模型会话加载耗时 | ≈ 1.5 s |
| 端到端运行时依赖 | `onnxruntime` + `numpy` + `soundfile` |
| `src/` 端到端运行时依赖 | `onnxruntime` + `numpy` + `sounddevice` |

也就是说，`kaldi_native_fbank` 在本项目里只保留「数值标尺」这一个用途，
用来证明自研前端是对的——**交付形态不依赖它**。

---

## 环境

Python 3.10（本项目在 conda 环境 `py310-gpu` 下开发验证）。

```
numpy 2.1.1        scipy 1.15.3       pandas 2.3.1
soundfile 0.14.0   sounddevice 0.5.6
onnxruntime 1.23.0
pyqtgraph 0.14.0   PyQt5 5.15.2
kaldi_native_fbank 1.22.3   # 仅参考实现 / 验证脚本需要，不是运行时依赖
```

---

## 快速开始

权重目录放在 `weight/` 下（脚本会自动找：目录里同时有 `tokens.txt` 和 `encoder*.onnx` 即可）。

```bash
# 1) 离线：播一条 aishell1 样本，边放边显示波形/RMS/语谱图 + 流式识别文字
python -I qt/test_spectrogram.py

# 2) 实时：麦克风采音，同样的三块图 + 识别文字
python -I qt/realtime_mic_monitor.py
```

两个脚本都会在窗口底部实时显示识别文字，关掉窗口即退出，退出时把完整识别结果打到终端。

不想看界面、只想跑一次识别：

```bash
python -I qt/asr_onnx_decode.py --model-dir weight/<权重目录> --frontend ours
```

`src/` 那套自包含实现另有三个入口（不用 Qt，OpenCV 窗口 + 终端 `print`）：

```bash
python -I test_mic_asr_mini.py                  # 实时麦克风 → 识别文字 + 滚动语谱图
python -I test_wav_asr_mini.py                  # wav 文件 → 识别文字 + 滚动语谱图
python -I test_spectrogram_cv_mini.py           # 只看 wav 的语谱图，不加载 onnx
```

识别结果用 `print` 原地刷新，同时用 `cv2.putText` 画在语谱图底部的白条上
（该函数只支持 ASCII，中文会显示为问号，终端输出不受影响）。
最后一个只用到 `numpy` + `soundfile` + OpenCV，专门看前端产出的 LogFBank 长什么样。

不收命令行参数：三个 onnx 与词表的路径、执行器顺序、麦克风设备都硬编码在文件顶部；
想看设备列表，把入口第 1 节里 `list_devices()` 那一行的注释取消即可。

---

## 项目结构

```
audio_ai_process/
├── test_mic_asr_mini.py       ← 入口③：mini 版实时麦克风（OpenCV 语谱图 + 识别文字）
├── test_wav_asr_mini.py       ← 入口④：mini 版 wav 识别（与③同构，输入换成文件）
├── test_spectrogram_cv_mini.py ← 入口⑤：只看 wav 的滚动语谱图（不加载 onnx）
├── qt/                        ← Qt 入口与旧版核心库
│   ├── test_spectrogram.py        ← 入口①：离线样本回放 + 三块图 + 流式识别
│   ├── realtime_mic_monitor.py    ← 入口②：实时麦克风 + 三块图 + 流式识别
│   ├── traditional_features.py   自研前端（对齐 kaldi 口径）
│   ├── streaming_features.py     增量分帧 + 增量 LogFBank
│   ├── asr_onnx_decode.py        onnxruntime 解码（贪心搜索 + 文本还原）
│   ├── streaming_asr.py          增量解码 + 麦克风采集
│   ├── live_audio_view.py        三块实时可视化控件
│   └── console_utf8.py           控制台 UTF-8（Windows 默认 GBK）
├── src/                       ← 自包含的 mini 实现（学习 / C++ 移植用，不 import qt/）
│   ├── audio_preprocess.py       波形块 → LogFBank 特征
│   ├── asr_decode.py             特征 → 文本（贪心搜索）
│   ├── mic_client.py             回调采集 + get_last_audio()
│   └── spectrogram_image.py      特征 → BGR 滚动语谱图画布（OpenCV）
├── scripts/                   ← 验证与冒烟测试脚本（不需要人工看界面）
├── data/                      ← aishell1 子集（train/test CSV + wav），离线样本来源
├── doc/                       ← 调研报告与工作流报告
├── weight/                    ← ONNX 权重（自行下载放置）
└── sharpa-onnx/               ← sherpa-onnx 上游源码（只作阅读参考，不参与运行）
```

`scripts/` 与 `sharpa-onnx/` 都不是运行必需：前者是开发期的验证工具，
后者是从上游 clone 下来对照阅读 C++ 实现的源码副本。`src/` 则是与 `qt/` 平行的
另一套自包含实现（见下方专节），两者互不引用——`src/` 不 import `qt/`，`qt/` 也不知道
`src/` 的存在。

---

## 两个 GUI 入口脚本

两个脚本共用 `qt/` 里的同一套核心库——识别类、绘图控件、权重定位都只有一份实现，
改一处两边同时生效。（另有 `src/` 那套独立实现的三个入口，见下方专节。）

### `test_spectrogram.py` — 离线样本回放

从 `data/aishell1_subset/` 读一条样本，用 `sounddevice` 出声播放，
同时按 1:1 的时间比例把同一份音频喂给识别线程，画面与声音严格同步。
播完后窗口保持打开，底部的「参考 / 识别」两行可以直接对照。

常改的参数都集中在文件开头的「二、全局配置」：

| 参数 | 默认 | 说明 |
|---|---|---|
| `SAMPLE_INDEX` | `16` | 用 test split 的第几条样本 |
| `ENABLE_ASR` | `True` | 设 `False` 退回纯可视化（省掉 1~2 s 权重加载） |
| `PLAY_GAIN` | `3.0` | 播放增益；太大会削顶，脚本会提示削顶比例 |
| `HOP_MS` | `25.0` | 帧步进，决定刷新率（25 ms → 40 fps） |
| `CANVAS_COLS` | `200` | 语谱图列数 → 200 × 25 ms = 5 秒滚动窗口 |

### `realtime_mic_monitor.py` — 实时麦克风

麦克风采集 → 三块图 → 流式识别，全部在一个窗口里。启动时先打印所有输入设备及其编号。

| 参数 | 默认 | 说明 |
|---|---|---|
| `DEVICE_ID` | `None` | 输入设备编号；`None` 用系统默认设备 |
| `LIST_DEVICES_ON_START` | `True` | 启动时打印设备表，方便挑编号 |
| `CHANNELS` | `1` | 麦克风阵列若是 2 in 可改成 2（代码只取第 0 通道） |
| `BLOCK_SIZE` | `400` | 每块样本数 = 一个 hop，保证 1 块 = 语谱图 1 列 |
| `CALIBRATE_SEC` | `2.0` | 启动后先收集约 2 秒，再锁定语谱图色阶 |
| `ENABLE_ASR` | `True` | 设 `False` 退回纯三图监控 |

输出设备/权限出问题时，脚本会在异常分支里打印排查建议（Win11 隐私设置、设备占用、编号核对）。

### 线程模型（为什么识别要单独开线程）

喂一次 encoder 最坏约 40 ms，而画面每 25 ms 就要推一帧，识别放主线程会让滚动明显卡顿。
所以一共三条线：

```
PortAudio 回调线程 ──audio_q(Queue)──> Qt 主线程 ──worker.feed()──> AsrWorker 识别线程
   只做 .copy()+入队                       画图 + 转交                     解码
   绝不碰 Qt、绝不碰模型                                                   写出 worker.text
```

回调线程只做一次 `.copy()` 再入队（PortAudio 会复用 `indata` 的底层缓冲，不 copy 会读到脏数据）；
队列满时丢**最旧**的块——实时识别要的是「现在」，堆积只会让延迟越来越大。
识别线程把结果整串写进 `worker.text`，主线程直接读（整串赋值，GIL 保证读不到半截，不需要锁）。

实测对照组（offscreen，同一段音频，开/关识别）：帧间隔中位数 31.3 ms vs 31.2 ms、
最坏 47.2 ms vs 47.1 ms——差异是噪声级别的。

---

## 核心实现

### 一、前端：把 kaldi 口径一项项抠出来

以 `kaldi_native_fbank` 的输出为 ground truth（它与模型训练时的前端一致）。初版实现
在 hamming 窗下偏差 `max|Δ| = 10.44`、`mean|Δ| = 0.92`，定位到七项差异后逐项修复：

| # | 差异项 | 原实现 | kaldi 口径 |
|---|---|---|---|
| 1 | **分帧语义** | `snip_edges=True`，帧数 `(N−W)//S + 1` | `snip_edges=False`，帧数 `round(N/S)`，帧中心钉在 `t·S`，越界镜像补齐 |
| 2 | **窗函数** | `hamming` | `povey = (0.5 − 0.5·cos(2πn/(N−1)))^0.85` |
| 3 | **窗函数分母** | `N`（periodic） | `N − 1`（对称窗，首尾都取到零点） |
| 4 | **Mel 滤波器组** | 在 bin 序号域线性插值 | 遍历整数 FFT bin，在 **Mel 域**插值 |
| 5 | **边界填充** | `reflect` | `symmetric`（镜像时重复边缘样本） |
| 6 | **对数下限** | `log(energy + 1e−10)` | `log(max(energy, float32_eps))`，静音地板停在 −15.94 |
| 7 | **预加重顺序** | 先对整段波形预加重再分帧 | 分帧后**逐帧**预加重，帧首样本乘 `(1−α)` |

其中第 4 项是算法层面的不同，不是精度问题：kaldi 的 `InitKaldiMelBanks` 是
「遍历每个整数 bin，把它的中心频率换算到 Mel，再在 Mel 域按到 `center` 的距离插值」，
且边界用严格不等号（落在边界上的 bin 权重直接是 0）。

修完后 `mean|Δ|` 从 **0.92 降到 4×10⁻⁶**（约 22 万倍），残余偏差集中在普通中间帧，
量级在 float32 表示精度附近，属于纯浮点运算顺序差异。

> 第 5 项是实测确定的：`reflect` 的最差点落在**第 0 帧**（边缘帧），`symmetric` 的最差点
> 落在中间帧——说明前者是边界模型错误，后者只剩浮点残差。
>
> 详细推导见 [工作流报告 第三节](doc/sherpa-onnx-ASR预处理及后处理工作流报告.md)。

### 二、解码：transducer 贪心搜索

三个 ONNX 各开一个 `InferenceSession`，`intra_op`/`inter_op` 线程数都为 1。

```python
hyp = [blank_id] * context_size        # 初始假设：全是 blank
decoder_out = run_decoder(hyp)

start = 0
while start + T < n_feat_frames:            # T = 61
    x = features[start : start + T]
    start += decode_chunk_len               # 时间轴前进 48 帧 = 480 ms
    encoder_out, states = run_encoder(x, states)
    for k in range(encoder_out.shape[1]):   # 注意：是运行时帧数，不是 T
        joiner_out = run_joiner(encoder_out[:, k:k+1], decoder_out)
        token_id = argmax(joiner_out)
        if token_id != blank_id:
            hyp.append(token_id)
            decoder_out = run_decoder(hyp[-context_size:])   # 只有出字才刷新
```

几个关键点：

- **116 个状态张量的初值由输入签名自动推导**（字符串维按 batch=1 取 1），换模型不用改代码。
- **`T = 61` 是输入帧数，不是输出帧数。** 时间轴只前进 48 帧，多出的 13 帧是卷积模块需要的
  重叠上下文，输出时被 encoder 裁掉；实测每 chunk 输出 **12** 帧。
- **`encoder_out.shape[1]` 必须取运行时值**，写死 `T` 会错。
- **blank 不刷新 decoder**，这是贪心搜索省算力的关键。
- `blank_id` 从 `tokens.txt` 查 `<blk>`，不硬编码。

**尾部补 100 帧 0**（约 1 秒）：解码循环的终止条件是 `start + T < n_frames`，
不补的话最后一段凑不满一个 chunk 会被整段丢掉，尾字识别不出来。注意补的 0 是
**特征域的 0**（log 域对应能量 1），不是静音值——这是照抄官方做法。

文本还原要处理 `▁`（U+2581，词首标记）：转空格后，**CJK 之间**的空格、**全角标点前后**的
空格要去掉，但 **CJK 与 ASCII 之间**的空格要保留（`是 Monday` 不能粘成 `是Monday`）。

### 三、流式：能复用的绝不重写

| 模块 | 职责 | 复用什么 |
|---|---|---|
| `streaming_features.py` | 增量分帧 + 增量 LogFBank | 帧内处理全用 `traditional_features.py` |
| `streaming_asr.py` | 增量解码 + 文本还原 + 麦克风采集 | 会话与 joiner 全用 `asr_onnx_decode.py` |

增量分帧只接管「分帧」这一层：维护跨块的样本尾巴、自己实现与批量版 `np.pad` 逐点一致的
两端镜像补齐、丢掉再也用不到的旧样本。单帧因果延迟 = `W//2` = 200 样本 = 12.5 ms。

**什么时候可以喂 encoder** 是流式解码的关键，判据必须与批量严格等价：

```
批量第 k 次喂的条件      48k + 61 < L        （L = 总帧数）
等价于                   L − 48k > 61
而流式喂完第 k 次后剩余   L − 48k 帧
=> 流式判据必须是  len(buf) > T   （严格大于）
```

写成 `>=` 就会在 `L` 恰好等于 `48k + 61` 时多喂一次，尾部多吐出几个 token。
代价是麦克风场景下多攒 1 帧（10 ms）才喂。

三样状态必须跨块存活、`reset()` 之外绝不清零：encoder 的 116 个状态张量、假设序列 `hyp`、
最近一次 `decoder_out`。

---

## `src/`（原 mini）：给学习和 C++ 移植的另一套

`qt/` 是为**生产可用**写的（原来的 `src/`）：分层抽象、跨模块复用、GUI 后台线程。它跑得
对，但读者要在多个文件间来回跳，才能拼出「一帧音频到一段文字」的完整路径。`src/`（原来的
`mini/`）是同一件事的另一种写法，唯一的优化目标是**能逐行对照移植**：

- **完全自包含**，不 `import qt/` 任何模块；三个类之间零耦合，正好对应 C++ 的三个编译单元。
- **前端与推理都写成显式逐帧循环**（去直流 → 逐帧倒序预加重 → 加窗 → FFT → 功率谱 →
  Mel 滤波 → log），能看见裸循环，没有大矩阵批量。
- **不用 Qt、不起 Python 线程、不引入 scipy**（窗函数手算）。麦克风靠 PortAudio 回调线程
  采集，主循环用 `get_last_audio()` 主动拉取。
- 注释按最密写：解释不变量（为什么预加重必须倒序、为什么补的 0 不是静音），而不是复述语法。

```
src/audio_preprocess.py   AudioPreprocess   波形块 → LogFBank 特征
src/asr_decode.py         AsrDecoder        特征 → 文本（贪心搜索）
src/mic_client.py         MicClient         回调采集 + get_last_audio()
src/spectrogram_image.py  SpectrogramImage  特征 → BGR 滚动语谱图画布（numpy + OpenCV）
```

有一条与 GUI 那条线**刻意不同**的取舍：`get_last_audio()` 取走的是自上次调用以来累积的
**全部**样本，而 `qt/streaming_asr.py` 在队列满时丢**最旧**的块。前者要的是波形连续无损
（慢一点只是延迟变大），后者要的是画面低延迟。只保留「最新一块」会让两次拉取之间的
30 ms 语音出现断口，去直流和预加重会把断口当突变，Mel 谱错乱、吞字。

因为它是**独立重写**的，还顺带成了第三把数值标尺——`scripts/verify_mini.py` 把它与 `qt/`
逐点对拍，两边算的是同一件事，一致就说明两份实现都没写错：

| 对拍项 | 覆盖范围 | 结果 |
|---|---|---|
| 前端 vs `frontend_for_onnx` | 4 段真实音频 × 块长 160/257/400/1000/一次全喂；短信号 n = 0/60/79/80/120/139/140/141/280/281（含 `edge` 退化分支） | **逐位相同**，`max\|Δ\| = 0` |
| 解码 token 序列 vs 批量解码 | 4 段音频 × 每次喂 1/7/37 帧/一次全喂 | **完全一致**，`n_chunks` 与 `n_enc_frames` 也逐项相同 |
| 实时麦克风 | RTF ≈ 0.10 | 识别正确，丢块 0 |

前端能逐位相同，是因为两边唯一的差异只有求和次序（`power @ filters.T` 走 BLAS vs 显式稀疏
累加）和 `np.abs(z)**2` vs `real² + imag²`——在 float64 下这些差异确实是 0。

```bash
python -I scripts/verify_mini.py --part a    # 只跑前端对拍（快，不加载模型）
python -I scripts/verify_mini.py --part b    # 只跑解码对拍（要加载 onnx）
python -I scripts/verify_mini.py             # 全跑
```

---

## 验证

所有验证都不需要人工看界面，脚本在 `scripts/` 下：

```bash
python -I scripts/smoke_test_spectrogram.py     # 离线链路：offscreen 跑完整播放+识别，统计帧间隔
python -I scripts/smoke_test_mic_audio.py       # 麦克风链路：假 InputStream 按 1× 速度回放真实 wav
python -I scripts/smoke_test_mic_monitor.py     # 麦克风链路：真开麦克风跑 N 秒自动退出
python -I scripts/verify_mini.py                # src/ 与 qt/ 逐点对拍（前端 + token 序列）
```

其中 `smoke_test_mic_audio.py` 只替换最外层的 PortAudio 那一层，
`realtime_mic_monitor.py` **一行不改**地跑完——验的是真实代码路径。

分帧层与特征层用 `scripts/probe_streaming_features.py` 对真实音频做了 bit-exact 验证；
前端对齐的推导过程用 `probe_align_features.py` / `probe_knf_components.py` /
`probe_mel_iterate.py` 复现。

四条测试音频的识别结果：

```
0.wav  昨天是 Monday，today is 礼拜二，the day after tomorrow 是星期三
1.wav  这是第一种，第二种叫呃与 always always 什么意思啊？
2.wav  就是频繁的啊，不认识记下来，frequently 频繁的
3.wav  第一句是个什么时态？加了 yes 是一般现代时，对，我们把它时态写上去。
```

---

## 已知问题

1. **`high_freq` 口径分歧未实测**。官方 `test_onnx.py` 取 8000 Hz（Nyquist），
   sherpa-onnx C++ 默认 `-400`（即 7600 Hz）。两条口径在本项目里都能与 kaldi 良好对齐，
   但识别结果是否一致**尚未实测**。当前默认 8000 Hz。
2. **`dither` 语义**。本模块的 `dither` 是高斯噪声标准差，直接作用于输入量纲；
   `kaldi_native_fbank` 的默认值假设输入是 16-bit 整数域。推理一律用 0——非 0 时结果不可复现，
   流式版更是直接拒绝 `dither != 0`。
3. **`sounddevice` 的设备编号与 `query_devices()` 打印的表对不上**。默认路径不受影响，暂未深究。

## 后续工作

- **基准测试**：对 `data/` 下的 aishell1 子集批量解码，与标注比 WER，出具准确率与 RTF 基线。
- 可选：VAD / 端点检测，避免静音段空转。

---

## 相关文档

| 文档 | 内容 |
|---|---|
| [工作流报告](doc/sherpa-onnx-ASR预处理及后处理工作流报告.md) | 模型契约、七项前端差异的完整推导、解码链路、流式等价性证明、全部实测数据 |
| [部署调研报告](doc/sherpa-onnx-ASR部署调研报告.md) | 调用路径与可行性调研 |
