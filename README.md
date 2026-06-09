# Qwenvl-Smit

Qwenvl-Smit 是一组面向本地 Qwen3-VL 工作流的 ComfyUI 自定义节点。它的目标是让用户可以直接在功能节点里选择模型，完成智能问答、图片理解、多图参考、视频帧理解、OCR、视觉问答、目标定位和目标检测。

本项目不包含任何模型权重。节点会优先读取 ComfyUI 的本地模型目录；如果选择的是支持的模型名称，并且本地没有对应模型，节点会自动下载到 ComfyUI 的模型目录中。

## 中文说明

### 主要功能

- 每个功能节点都内置 `模型` 下拉栏，不需要额外连接模型加载节点。
- 支持 Qwen3-VL 4B、8B 级别模型。
- 智能问答节点可不接图片直接做文本问答，也可接入最多 6 张图片进行视觉分析。
- 视频理解节点支持最多 3 路视频帧输入。
- 支持 OCR、视觉问答、图像描述、目标定位、目标检测、自定义提示词。
- 支持中文和英文输出语言选择。
- 支持固定、随机、递增、递减四种种子模式。
- 支持 4bit、8bit 量化加载；量化是否可用取决于用户本地的 Python、CUDA、PyTorch 和 bitsandbytes 兼容情况。
- 支持清理模型缓存，便于本地显存紧张时释放资源。

### 节点列表

- `QwenVL-Smit 智能问答`

  使用 Qwen3-VL 模型进行文本问答、图片理解、多图参考、OCR、视觉问答、图像描述、目标定位、目标检测或自定义分析。图片输入是可选的；不接图片时就是普通智能问答，接入图片时最多支持 `图片1` 到 `图片6`。

- `QwenVL-Smit 视频理解`

  使用 Qwen3-VL 模型理解 ComfyUI 的 IMAGE 帧批次。最多支持 `视频帧1` 到 `视频帧3`，可以限制最大帧数以降低显存压力。

- `QwenVL-Smit 提示词预设`

  这个节点不会加载模型，也不会运行推理。它只负责生成可复用的提示词，适合把 OCR、图像描述、视觉问答、目标检测、目标定位等常见任务预设给其他节点使用。

- `QwenVL-Smit 清理缓存`

  清理当前进程中的模型缓存，并尝试释放 CUDA 显存缓存。这个节点有一个可选的 `触发` 输入和一个 `触发` 输出，可以接入任意节点输出作为执行触发；清理完成后会把输入内容原样传出去，方便继续连接后面的节点。如果不接 `触发`，也可以单独运行用于手动清理。

### 安装方式

#### 使用 ComfyUI Manager

当前 GitHub 版本可以通过 ComfyUI Manager 的 Git URL 安装：

```text
https://github.com/jiuman994/Qwenvl-Smit
```

安装完成后重启 ComfyUI。

如果希望用户可以直接在 Manager 搜索结果里找到该节点，还需要后续把仓库提交到 ComfyUI Manager 节点列表或 Comfy Registry。本项目已经包含 `node_list.json` 和 `pyproject.toml` 元数据，方便后续上架。

#### 手动安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jiuman994/Qwenvl-Smit.git
cd Qwenvl-Smit
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
- `modelscope` 为可选依赖，安装成功时会优先用于国内下载；如果当前 Python 版本不支持，可以不安装，节点会自动回退到 Hugging Face。

### 显存建议

模型下拉栏默认只放入更适合本地 ComfyUI 的模型。Qwen3-VL 默认展示 4B、8B。更大的模型不会出现在默认列表中，避免普通用户误选后无法加载。

| 显存 | Qwen3-VL 建议上限 | 建议设置 |
| --- | --- | --- |
| 8GB | 4B | 优先使用 `4bit`，视频帧数建议控制在 16 到 32 帧 |
| 16GB | 8B | 优先使用 `4bit`，多图和视频任务减少帧数 |
| 24GB | 8B | 可使用 8B，复杂视频任务继续控制帧数 |

视频节点里的 `最大帧数` 参数默认值是 32，参数上限是 512。README 中出现的 32 帧只是适合 8GB 显存起步测试的建议值，不是节点功能上限。

### 模型选择与模型目录

每个功能节点都有自己的 `模型` 下拉栏：

1. 已放入 ComfyUI 模型目录的本地模型会优先显示。
2. 本地模型后面会显示当前支持的模型名称。
3. 如果本地已存在同名模型，则隐藏对应的下载预设，避免重复显示。
4. 下拉栏只显示模型文件夹名或模型名称，不显示完整本机路径，不显示 `Qwen/`、`QwenVL/` 这类前缀，也不会给重复模型添加括号后缀。
5. 如果选择模型名称，并且本地没有该模型，节点会自动下载到对应的 ComfyUI 模型目录。
6. 自动下载顺序为：ModelScope 魔搭社区优先，Hugging Face 兜底。如果未安装 `modelscope`，会自动跳过 ModelScope。

推荐把 Qwen3-VL 模型放在：

```text
ComfyUI/models/LLM/Qwen-VL/模型文件夹
```

示例：

```text
ComfyUI/models/LLM/Qwen-VL/Qwen3-VL-4B-Instruct
```

节点也会尽量扫描一些常见模型目录，例如：

```text
ComfyUI/models/LLM/Qwen-VL
ComfyUI/models/LLM
ComfyUI/models/VQA
ComfyUI/models/hugface
ComfyUI/models/transformers
```

### 基础工作流示例

智能问答，不接图片：

```text
QwenVL-Smit 智能问答
任务类型: 自定义
提示词: 请解释一下什么是视觉语言模型。
```

图片 OCR：

```text
Load Image -> QwenVL-Smit 智能问答
任务类型: OCR文字识别
提示词: 提取图片中的全部文字。
```

多图参考：

```text
Load Image -> 图片1
Load Image -> 图片2
Load Image -> 图片3
QwenVL-Smit 智能问答
```

视频理解：

```text
Load Video frames as IMAGE batch -> QwenVL-Smit 视频理解
任务类型: 图像描述
最大帧数: 32  # 示例值，可按显存和输入帧数调高
提示词: 总结这个视频的时间线。
```

清理缓存，接入工作流：

```text
QwenVL-Smit 智能问答 文本输出 -> QwenVL-Smit 清理缓存 触发输入
QwenVL-Smit 清理缓存 触发输出 -> 后续节点
```

如果只是想手动释放显存，也可以直接添加 `QwenVL-Smit 清理缓存` 节点并运行，不需要连接其他节点。

### 输出说明

图片和视频节点只返回：

- `文本`：模型原始回答。

如果需要结构化内容，可以直接在 `提示词` 中描述你希望模型按什么格式回答；节点本身不再额外提供独立的 JSON 或坐标输出。

### 下载失败处理

如果运行时出现 `LocalEntryNotFoundError`，通常表示节点正在自动下载模型，但当前环境无法连接 ModelScope / Hugging Face，或者本地缓存里没有完整模型文件。

可以按下面方式处理：

1. 国内网络优先安装可选依赖：`pip install modelscope`。
2. 检查网络是否可以访问 ModelScope 或 Hugging Face。
3. 如果模型需要权限，先执行 `huggingface-cli login`。
4. 国内网络也可以在启动 ComfyUI 前设置 Hugging Face 镜像：

```bash
set HF_ENDPOINT=https://hf-mirror.com
```

5. 如需强制下载源，可以在启动 ComfyUI 前设置：

```bash
set QWENVL_SMIT_DOWNLOAD_SOURCE=modelscope
```

或者：

```bash
set QWENVL_SMIT_DOWNLOAD_SOURCE=huggingface
```

6. 也可以手动下载模型，并把完整模型文件夹放入对应目录：

```text
ComfyUI/models/LLM/Qwen-VL/模型文件夹
```

### 模型来源与声明

本项目只是 ComfyUI 节点适配层，不包含任何 Qwen 模型权重。

Qwen3-VL 模型由 Qwen 团队提供。本项目在这些模型的基础上做 ComfyUI 节点适配，包括中文界面、本地模型目录识别、多图输入、视频帧输入、语言选择、种子模式和适合本地显存的模型列表。使用模型时请遵守对应 Qwen 模型的开源协议与使用政策。

Qwenvl-Smit 节点代码本身使用 Apache-2.0 协议开源。

## English

Qwenvl-Smit is a ComfyUI custom node extension for local Qwen3-VL workflows. It lets users select models directly inside each functional node and run smart Q&A, image understanding, multi-image reference, video-frame understanding, OCR, visual question answering, grounding, and detection.

This project does not include model weights. It reads local models from ComfyUI model folders first. If a supported model name is selected and the model is not available locally, the node downloads it into the matching ComfyUI model folder.

### Features

- Built-in model selector inside each functional node.
- Qwen3-VL presets are limited to practical 4B and 8B options.
- The smart Q&A node can run text-only Q&A without images, or analyze up to 6 image inputs.
- Video nodes support up to 3 video-frame inputs.
- Supports OCR, VQA, captioning, grounding, detection, and custom prompts.
- Supports Chinese and English output language selection.
- Supports fixed, random, increment, and decrement seed modes.
- Supports 4-bit and 8-bit loading when the user's Python, CUDA, PyTorch, and bitsandbytes stack is compatible.
- Includes a cache cleanup node for local VRAM-limited workflows.

### Nodes

- `QwenVL-Smit 智能问答`

  Runs Qwen3-VL text Q&A, image understanding, multi-image reference, OCR, VQA, captioning, grounding, detection, or custom analysis. Image input is optional. Without images it works as a normal Q&A node; with images it supports `图片1` through `图片6`.

- `QwenVL-Smit 视频理解`

  Runs Qwen3-VL video-frame understanding from ComfyUI IMAGE batches. Supports `视频帧1` through `视频帧3`.

- `QwenVL-Smit 提示词预设`

  Builds reusable prompt text only. It does not load a model or run inference.

- `QwenVL-Smit 清理缓存`

  Clears the in-process model cache and attempts to release CUDA cache. It also has an optional `触发` passthrough input/output, so users can place it inside a workflow and continue from the same value after cleanup.

### Installation

Use ComfyUI Manager's Git URL installer:

```text
https://github.com/jiuman994/Qwenvl-Smit
```

Manual install:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jiuman994/Qwenvl-Smit.git
cd Qwenvl-Smit
python install.py
```

Restart ComfyUI after installation.

### Model Folders

Each functional node has its own `模型` dropdown:

1. Local models found in ComfyUI model folders are listed first.
2. Supported model names are listed after local models.
3. If a local model with the same folder name exists, the matching download preset is hidden.
4. The dropdown shows only model folder names or supported model names, not full local paths, not `Qwen/` or `QwenVL/` prefixes, and not duplicate suffixes.
5. Automatic download order is ModelScope first, then Hugging Face.

Recommended Qwen3-VL folder:

```text
ComfyUI/models/LLM/Qwen-VL/model-folder
```

### Credits

This extension is an integration layer and does not include Qwen model weights.

Qwen3-VL models are provided by the Qwen team. Qwenvl-Smit adapts those models for local ComfyUI workflows with Chinese UI labels, local model-folder discovery, multi-image inputs, video-frame inputs, language selection, seed modes, and local-friendly model presets. Please follow the license and use policy of the selected upstream Qwen model.

Qwenvl-Smit node code is released under Apache-2.0.

## License

Apache-2.0. See [LICENSE](LICENSE).
