# Qwenvl-Smit

Qwenvl-Smit 是一组面向本地 Qwen3-VL 和 Qwen3.5 工作流的 ComfyUI 自定义节点。它的目标是让 ComfyUI 用户可以直接在节点里选择模型，完成图片理解、多图参考、视频帧理解、OCR、视觉问答、目标定位、目标检测、文本对话和 JSON 结构化输出。

本项目不包含任何模型权重。模型会优先从 ComfyUI 的模型目录中读取；如果用户选择的是支持的 Hugging Face 模型 ID，并且本地还没有对应模型，节点会自动下载到 ComfyUI 的模型目录中。

## 中文说明

### 主要功能

- 每个功能节点都内置 `模型` 下拉栏，不需要额外连接模型加载节点。
- 支持 Qwen3-VL 4B、8B 级别模型，避免默认展示不适合普通本地显存的大模型。
- 支持 Qwen3.5 4B、9B、27B、35B-A3B 级别模型。
- 图片理解节点支持最多 6 张图片输入。
- 视频理解节点支持最多 3 路视频帧输入。
- 支持 OCR、视觉问答、图像描述、目标定位、目标检测、自定义提示词。
- 支持强制 JSON 提示，并提供 JSON 与坐标 JSON 辅助输出。
- 支持 4bit、8bit 量化加载；是否可用取决于用户本地的 Python、CUDA、PyTorch 和 bitsandbytes 兼容情况。
- 支持清理模型缓存，便于本地显存紧张时释放资源。

### 节点列表

- `QwenVL-Smit 图片理解`

  使用 Qwen3-VL 模型进行图片理解、多图参考、OCR、视觉问答、图像描述、目标定位、目标检测或自定义分析。最多支持 `图片1` 到 `图片6`。

- `QwenVL-Smit 视频理解`

  使用 Qwen3-VL 模型理解 ComfyUI 的 IMAGE 帧批次。最多支持 `视频帧1` 到 `视频帧3`，可以限制最大帧数以降低显存压力。

- `Qwen3.5-Smit 文本对话`

  使用 Qwen3.5 模型进行纯文本对话。

- `Qwen3.5-Smit 图片理解`

  使用 Qwen3.5 模型进行图片或多图理解。最多支持 `图片1` 到 `图片6`。

- `Qwen3.5-Smit 视频理解`

  使用 Qwen3.5 模型理解 ComfyUI 的 IMAGE 帧批次。最多支持 `视频帧1` 到 `视频帧3`。

- `QwenVL-Smit 提示词预设`

  这个节点不会加载模型，也不会运行推理。它只负责生成可复用的提示词，适合把 OCR、图像描述、视觉问答、目标检测、目标定位、JSON 输出等常见任务预设给其他节点使用。如果你的工作流很简单，可以完全不使用这个节点，直接在图片、视频或文本节点里写提示词。

- `QwenVL-Smit 清理缓存`

  清理当前进程中的模型缓存，并尝试释放 CUDA 显存缓存。

### 安装方式

#### 使用 ComfyUI Manager

当前 GitHub 版本可以通过 ComfyUI Manager 的 Git URL 安装：

1. 打开 ComfyUI Manager。
2. 选择 `Install via Git URL` 或 `Install from Git URL`。
3. 输入：

```text
https://github.com/jiuman994/Qwenvl-Smit
```

4. 安装完成后重启 ComfyUI。

如果希望用户可以直接在 Manager 搜索结果里找到该节点，还需要后续把仓库提交到 ComfyUI Manager 节点列表或 Comfy Registry。本项目已经包含 `node_list.json` 和 `pyproject.toml` 元数据，方便后续上架。

#### 手动安装

把仓库克隆到 `ComfyUI/custom_nodes`：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jiuman994/Qwenvl-Smit.git
```

进入插件目录，用 ComfyUI 正在使用的 Python 安装依赖：

```bash
cd ComfyUI/custom_nodes/Qwenvl-Smit
python install.py
```

也可以手动安装依赖：

```bash
pip install -r requirements.txt
```

安装完成后重启 ComfyUI。

### 推荐环境

- Python `3.10` 或更高版本。
- 已正确安装 CUDA 版 PyTorch。
- 推荐 GPU 显存 8GB 或更高。
- `transformers>=4.57.0`
- `qwen-vl-utils>=0.0.14`
- `accelerate>=1.0.0`
- `huggingface_hub>=0.34.0`
- `bitsandbytes` 为可选依赖，仅在用户环境支持时用于 4bit 或 8bit 量化加载。

### 显存建议

模型下拉栏默认只放入更适合本地 ComfyUI 的模型。Qwen3-VL 默认只展示 4B 和 8B。Qwen3.5 默认展示 4B、9B、27B、35B-A3B。更大的模型不会出现在默认列表中，避免普通用户误选后无法加载。

| 显存 | Qwen3-VL 建议上限 | Qwen3.5 建议上限 | 建议设置 |
| --- | --- | --- | --- |
| 8GB | 4B | 4B 或 9B | 优先使用 `4bit`，视频帧数建议控制在 16 到 32 帧 |
| 16GB | 8B | 9B 或 27B | 优先使用 `4bit`，多图和视频任务减少帧数 |
| 24GB | 8B | 27B 或 35B-A3B | 优先使用 `4bit`，复杂视频任务继续控制帧数 |

实际能否运行还取决于分辨率、输入图片数量、视频帧数、系统占用、PyTorch 版本和量化库兼容情况。

视频节点里的 `最大帧数` 参数默认值是 32，参数上限是 512。README 中出现的 32 帧只是适合 8GB 显存起步测试的建议值，不是节点功能上限。

### 模型选择与模型目录

每个功能节点都有自己的 `模型` 下拉栏：

1. 已放入 ComfyUI 模型目录的本地模型会优先显示。
2. 本地模型后面会显示当前支持的 Hugging Face 模型 ID。
3. 如果选择 Hugging Face 模型 ID，并且本地没有该模型，节点会自动下载到对应的 ComfyUI 模型目录。

推荐把 Qwen3-VL 模型放在：

```text
ComfyUI/models/LLM/Qwen-VL/模型文件夹
```

示例：

```text
ComfyUI/models/LLM/Qwen-VL/Qwen3-VL-4B-Instruct
```

推荐把 Qwen3.5 模型放在：

```text
ComfyUI/models/LLM/Qwen3.5/模型文件夹
```

节点也会尽量扫描一些常见模型目录，例如：

```text
ComfyUI/models/LLM/Qwen-VL
ComfyUI/models/LLM/Qwen3.5
ComfyUI/models/LLM
ComfyUI/models/VQA
ComfyUI/models/hugface
ComfyUI/models/transformers
```

### 基础工作流示例

图片 OCR：

```text
Load Image -> QwenVL-Smit 图片理解
任务类型: OCR文字识别
提示词: 提取图片中的全部文字。
```

多图参考：

```text
Load Image -> 图片1
Load Image -> 图片2
Load Image -> 图片3
QwenVL-Smit 图片理解
```

视觉问答：

```text
Load Image -> QwenVL-Smit 图片理解
任务类型: 视觉问答
提示词: 画面主体是什么？它正在做什么？
```

目标检测：

```text
Load Image -> QwenVL-Smit 图片理解
任务类型: 目标检测
强制JSON: true
提示词: 检测画面中所有可见的商品包装，返回类别和坐标。
```

视频理解：

```text
Load Video frames as IMAGE batch -> QwenVL-Smit 视频理解
任务类型: 图像描述
最大帧数: 32  # 示例值，可按显存和输入帧数调高
提示词: 总结这个视频的时间线。
```

### 输出说明

图片、视频和文本节点会返回以下内容中的一部分或全部：

- `文本`：模型原始回答。
- `JSON`：如果回答中包含有效 JSON，会尝试提取并返回。
- `坐标JSON`：如果回答中包含简单的 `[x1, y1, x2, y2]` 或 `<box>...</box>` 坐标格式，会尝试辅助提取。

最终输出格式仍由模型生成结果决定。如果需要更稳定的结构化输出，请打开 `强制JSON`，并在提示词中明确写出需要的 schema。

### 注意事项

- 大模型会占用较多显存。8GB 显卡建议先从 4B 模型开始。
- `bitsandbytes` 的可用性取决于 Python、CUDA、PyTorch 和操作系统版本。
- 视频节点接收的是 ComfyUI 的 IMAGE 帧批次，不直接解码视频文件。请先用其他视频加载节点把视频转成帧。
- 部分 Hugging Face 模型可能需要先执行 `huggingface-cli login`。
- 自动下载的模型会存放在对应的 ComfyUI 模型目录下。

### 模型来源与声明

本项目只是 ComfyUI 节点适配层，不包含任何 Qwen 模型权重。

Qwen3-VL 与 Qwen3.5 模型由 Qwen 团队提供。本项目在这些模型的基础上做 ComfyUI 节点适配，包括中文界面、本地模型目录识别、多图输入、视频帧输入、JSON 辅助输出和适合本地显存的模型列表。使用模型时请遵守对应 Qwen 模型的开源协议与使用政策。

Qwenvl-Smit 节点代码本身使用 Apache-2.0 协议开源。

### 开发检查

```bash
python -m py_compile __init__.py nodes.py
```

## English

Qwenvl-Smit is a ComfyUI custom node extension for local Qwen3-VL and Qwen3.5 workflows. It lets users select models directly inside each functional node and run image understanding, multi-image reference, video-frame understanding, OCR, visual question answering, grounding, detection, text chat, and JSON-oriented structured output.

This project does not include model weights. It reads local models from ComfyUI model folders first. If a supported Hugging Face model ID is selected and the model is not available locally, the node downloads it into the matching ComfyUI model folder.

### Features

- Built-in model selector inside each functional node.
- Qwen3-VL presets are limited to practical 4B and 8B options.
- Qwen3.5 presets are limited to 4B, 9B, 27B, and 35B-A3B options.
- Image nodes support up to 6 image inputs.
- Video nodes support up to 3 video-frame inputs.
- Supports OCR, VQA, captioning, grounding, detection, and custom prompts.
- Supports JSON-oriented prompting and helper outputs for JSON and bounding boxes.
- Supports 4-bit and 8-bit loading when the user's Python, CUDA, PyTorch, and bitsandbytes stack is compatible.
- Includes a cache cleanup node for local VRAM-limited workflows.

### Nodes

- `QwenVL-Smit 图片理解`

  Runs Qwen3-VL image understanding, multi-image reference, OCR, VQA, captioning, grounding, detection, or custom analysis. Supports `图片1` through `图片6`.

- `QwenVL-Smit 视频理解`

  Runs Qwen3-VL video-frame understanding from ComfyUI IMAGE batches. Supports `视频帧1` through `视频帧3`.

- `Qwen3.5-Smit 文本对话`

  Runs text-only chat with a Qwen3.5 model.

- `Qwen3.5-Smit 图片理解`

  Runs image or multi-image understanding with a Qwen3.5 model. Supports `图片1` through `图片6`.

- `Qwen3.5-Smit 视频理解`

  Runs video-frame understanding with a Qwen3.5 model. Supports `视频帧1` through `视频帧3`.

- `QwenVL-Smit 提示词预设`

  Builds reusable prompt text only. It does not load a model or run inference.

- `QwenVL-Smit 清理缓存`

  Clears the in-process model cache and attempts to release CUDA cache.

### Installation

#### ComfyUI Manager

Install the current GitHub version through ComfyUI Manager's Git URL installer:

1. Open ComfyUI Manager.
2. Choose `Install via Git URL` or `Install from Git URL`.
3. Paste:

```text
https://github.com/jiuman994/Qwenvl-Smit
```

4. Restart ComfyUI.

To make the node appear directly in Manager search results, the repository still needs to be submitted to the ComfyUI Manager custom node list or Comfy Registry. This project already includes `node_list.json` and `pyproject.toml` metadata for that step.

#### Manual Install

Clone the repository into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jiuman994/Qwenvl-Smit.git
```

Install dependencies with the Python used by ComfyUI:

```bash
cd ComfyUI/custom_nodes/Qwenvl-Smit
python install.py
```

Or install manually:

```bash
pip install -r requirements.txt
```

Restart ComfyUI after installation.

### Recommended Environment

- Python `3.10` or newer.
- CUDA-enabled PyTorch for GPU inference.
- Recommended GPU VRAM: 8GB or higher.
- `transformers>=4.57.0`
- `qwen-vl-utils>=0.0.14`
- `accelerate>=1.0.0`
- `huggingface_hub>=0.34.0`
- `bitsandbytes` is optional and only used when the local environment supports 4-bit or 8-bit loading.

### VRAM Guide

The default model dropdown avoids very large models that are unrealistic for most local ComfyUI users.

| VRAM | Qwen3-VL practical maximum | Qwen3.5 practical maximum | Suggested settings |
| --- | --- | --- | --- |
| 8GB | 4B | 4B or 9B | Prefer `4bit`; recommended video range is 16 to 32 frames |
| 16GB | 8B | 9B or 27B | Prefer `4bit`; reduce image count and video frames |
| 24GB | 8B | 27B or 35B-A3B | Prefer `4bit`; still reduce frames for complex video tasks |

Actual usability depends on resolution, image count, video frame count, system memory usage, PyTorch version, and quantization compatibility.

The `最大帧数` parameter defaults to 32 and can be set up to 512. The 32-frame value shown in examples is a starting recommendation for 8GB GPUs, not a hard node limit.

### Model Folders

Each functional node has its own `模型` dropdown:

1. Local models found in ComfyUI model folders are listed first.
2. Supported Hugging Face model IDs are listed after local models.
3. If a selected Hugging Face model is not available locally, it downloads into the matching ComfyUI model folder.

Recommended Qwen3-VL folder:

```text
ComfyUI/models/LLM/Qwen-VL/model-folder
```

Recommended Qwen3.5 folder:

```text
ComfyUI/models/LLM/Qwen3.5/model-folder
```

The node also scans common existing model folders when available:

```text
ComfyUI/models/LLM/Qwen-VL
ComfyUI/models/LLM/Qwen3.5
ComfyUI/models/LLM
ComfyUI/models/VQA
ComfyUI/models/hugface
ComfyUI/models/transformers
```

### Notes

- Start with a 4B model on 8GB GPUs.
- Video nodes expect ComfyUI IMAGE frame batches and do not decode video files directly.
- Some Hugging Face models may require `huggingface-cli login`.
- Downloaded models are stored under the matching ComfyUI model folder.

### Credits

This extension is an integration layer and does not include Qwen model weights.

Qwen3-VL and Qwen3.5 models are provided by the Qwen team. Qwenvl-Smit adapts those models for local ComfyUI workflows with Chinese UI labels, local model-folder discovery, multi-image inputs, video-frame inputs, JSON helper outputs, and local-friendly model presets. Please follow the license and use policy of the selected upstream Qwen model.

Qwenvl-Smit node code is released under Apache-2.0.

## License

Apache-2.0. See [LICENSE](LICENSE).
