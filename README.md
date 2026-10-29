# audio_ai_process

中文流式语音识别（streaming zipformer2 transducer）的**纯 Python 复现** —— mini 版。

这个分支只做一件事：把底层 ASR 讲清楚、跑起来。前端用自研的传统实现
（numpy）产出模型输入，推理与后处理用 `onnxruntime`，再配一个 OpenCV 语谱图窗口。
不依赖 sherpa-onnx 运行时，只保留底层 ASR 与一个 OpenCV 窗口。

模型：`sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05`
（int8 量化，中英混合，自带标点）

---

## 成果速览

| 指标 | 实测结果 |
|---|---|
| 自研前端 vs `kaldi_native_fbank` 参考实现 | `mean\|Δ\|` ≈ 4×10⁻⁶，`max\|Δ\|` ≤ 5.4×10⁻⁴ |
| 流式前端 vs 批量前端（分帧层 / 特征层） | **逐位相同**（`max\|Δ\| = 0`） |
| 离线实时率 RTF | 0.059 ~ 0.068（比实时快约 16 倍） |
| 实时识别 RTF | ≈ 0.10（比实时快约 10 倍） |
| 实时链路峰值负载（单次喂 encoder / 喂间隔） | **6 ~ 9%** |
| 模型会话加载耗时 | ≈ 1.5 s |
| 端到端运行时依赖 | `onnxruntime` + `numpy` + `soundfile` + `sounddevice` + `opencv-python` |

`kaldi_native_fbank` 只在开发期当「数值标尺」用，证明自研前端是对的——
**交付形态不依赖它**。

---

## 环境

Python 3.10（在 conda 环境 `py310-gpu` 下开发验证）。

```
numpy 2.1.1          onnxruntime 1.23.0
soundfile 0.14.0     sounddevice 0.5.6
opencv-python
kaldi_native_fbank 1.22.3   # 仅数值对拍需要，不是运行时依赖
```

---

## 快速开始

权重目录放在 `weight/` 下（脚本会自动找：目录里同时有 `tokens.txt` 和 `encoder*.onnx` 即可）。

```bash
python -I test_mic_asr_mini.py        # 实时麦克风 → 识别文字 + 滚动语谱图
python -I test_wav_asr_mini.py        # wav 文件 → 识别文字 + 滚动语谱图
python -I test_spectrogram_cv_mini.py # 只看 wav 的语谱图，不加载 onnx
```

识别结果用 `print` 原地刷新，同时用 `cv2.putText` 画在语谱图底部的白条上
（该函数只支持 ASCII，中文会显示为问号，终端输出不受影响）。
最后一个只用到 `numpy` + `soundfile` + OpenCV，专门看前端产出的 LogFBank 长什么样。

不收命令行参数：三个 onnx 与词表的路径、执行器顺序、麦克风设备都硬编码在文件顶部；
想看设备列表，把入口第 1 节里 `list_devices()` 那一行的注释取消即可。

### 打包

```bash
git archive --format=zip --output=audio_ai_process.zip HEAD
```

把当前工作区的内容打成当前目录下的 `audio_ai_process.zip`：`weight/`、`sharpa-onnx/`、wav
等被 `.gitignore` 排除的东西不会进去。`git archive` 只打包已提交的内容，工作区里
没提交的改动要先 `git add` / `git commit` 才会进包。

---

## 项目结构

```
audio_ai_process/
├── test_mic_asr_mini.py       ← 实时麦克风（OpenCV 语谱图 + 识别文字）
├── test_wav_asr_mini.py       ← wav 识别（与上面同构，输入换成文件）
├── test_spectrogram_cv_mini.py ← 只看 wav 的滚动语谱图（不加载 onnx）
├── src/                       ← 自包含的 mini 实现（学习 / C++ 移植用）
│   ├── audio_preprocess.py       波形块 → LogFBank 特征
│   ├── asr_decode.py             特征 → 文本（贪心搜索）
│   ├── mic_client.py             回调采集 + get_last_audio()
│   └── spectrogram_image.py      特征 → BGR 滚动语谱图画布（OpenCV）
├── data/                      ← aishell1 子集（train/test CSV + wav），离线样本来源
├── doc/                       ← 调研报告与工作流报告
├── weight/                    ← ONNX 权重（自行下载放置）
└── sharpa-onnx/               ← sherpa-onnx 上游源码（只作阅读参考，不参与运行）
```

三个入口都只是「配置 + 数据源」：装配 `AudioPreprocess` / `AsrDecoder` /
`SpectrogramImage`，然后按块喂数据。三个类互不知道对方的存在，靠特征矩阵相连。

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

### 三、流式：同一份代码维护跨块状态

mini 版没有另写一套流式实现，而是让每个类自己保管跨块状态：

| 模块 | 跨块状态 | 作用 |
|---|---|---|
| `AudioPreprocess.accept_waveform()` | 样本尾巴、预加重历史 | 增量分帧 + 增量 LogFBank，帧内处理与批量口径逐点一致 |
| `AsrDecoder.accept_features()` | 特征缓冲、encoder 状态、`hyp`、`decoder_out` | 攒够一个 chunk 就喂 encoder，增量出字 |

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

### 四、可视化：特征 → BGR 画布

`SpectrogramImage` 只接收特征、不碰波形，与 `AsrDecoder` 只接收特征是同一个边界。
每块特征进来后：

1. **转置 + 上下翻转**：`(帧数, n_mels)` → `(n_mels, 帧数)`，并让 Mel 0 落在画面底部；
2. **对数能量线性拉伸**：`clip((值 − lo) / (hi − lo) × 255)`，色阶前 0.5 秒标定一次后锁死，
   避免安静段和说话段互相挤压亮度导致画面「呼吸」；
3. **viridis 伪彩 + 放大**：80 行太矮，先 `cv2.resize` 再 `cv2.applyColorMap`；
4. **滚动窗口**：只保留最近 400 帧（4 秒），新帧贴右端，左侧补黑。

可视化路径的前端要传 `tail_pad_frames=0`：那 100 帧补出来的 0 在图上是一块比静音还亮的
假色块，只该喂解码器，不该喂给眼睛。

---

## `src/`：给学习和 C++ 移植的实现

`src/` 是同一件事的另一种写法，唯一的优化目标是**能逐行对照移植**：

- **完全自包含**：不引入本仓库以外的实现，三个类之间零耦合，正好对应 C++ 的三个编译单元。
- **前端与推理都写成显式逐帧循环**（去直流 → 逐帧倒序预加重 → 加窗 → FFT → 功率谱 →
  Mel 滤波 → log），能看见裸循环，没有大矩阵批量。
- **不起 Python 线程、不引入 scipy**（窗函数手算）。麦克风靠 PortAudio 回调线程
  采集，主循环用 `get_last_audio()` 主动拉取。

```
src/audio_preprocess.py   AudioPreprocess   波形块 → LogFBank 特征
src/asr_decode.py         AsrDecoder        特征 → 文本（贪心搜索）
src/mic_client.py         MicClient         回调采集 + get_last_audio()
src/spectrogram_image.py  SpectrogramImage  特征 → BGR 滚动语谱图画布（numpy + OpenCV）
```

`get_last_audio()` 取走的是自上次调用以来累积的**全部**样本，而不是只取最新一块：
只保留「最新一块」会让两次拉取之间的语音出现断口，去直流和预加重会把断口当突变，
Mel 谱错乱、吞字。波形连续无损比单次延迟更重要。

---

## 验证

三个入口都能直接跑；只想确认前端与解码链路是否正常，跑一次 wav 入口即可
（权重包自带 `test_wavs/`）：

```bash
python -I test_wav_asr_mini.py          # 前端 + 解码 + 可视化
python -I test_spectrogram_cv_mini.py   # 只看前端产出的语谱图
```

四条测试音频的识别结果：

```
0.wav  昨天是 Monday，today is 礼拜二，the day after tomorrow 是星期三
1.wav  这是第一种，第二种叫呃与 always always 什么意思啊？
2.wav  就是频繁的啊，不认识记下来，frequently 频繁的
3.wav  第一句是个什么时态？加了 yes 是一般现代时，对，我们把它时态写上去。
```

---

## C++ 移植计划

最终交付是 C/C++ 库：外部拿到麦克风 PCM 后直接调用，库内完成前端 → onnxruntime 推理 → 文本。

1. **wav 前端 + 可视化**：C++ 复刻 `test_spectrogram_cv_mini.py`（读 wav、逐帧预处理、OpenCV 语谱图），先不接模型。
2. **wav ASR**：链接 onnxruntime C++ 库，移植贪心搜索与文本还原。
3. **产出库**：封装成库（初始化 / reset / 喂 PCM / 取文本），麦克风采集留在库外。

每步都与 Python 版逐点或逐 token 对拍；`src/` 只作参考实现，不是部署产物。

---

## 已知问题

1. **`high_freq` 口径分歧未实测**。官方 `test_onnx.py` 取 8000 Hz（Nyquist），
   sherpa-onnx C++ 默认 `-400`（即 7600 Hz）。两条口径在本项目中都能与 kaldi 良好对齐，
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
