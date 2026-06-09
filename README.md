# Qwenvl-Smit

ComfyUI custom nodes for local Qwen3-VL and Qwen3.5 multimodal workflows. Each functional node includes its own model selector, uses ComfyUI model folders by default, and downloads supported Hugging Face models only when needed.

Qwenvl-Smit 是一组面向本地 Qwen3-VL 和 Qwen3.5 多模态工作流的 ComfyUI 自定义节点。每个功能节点都内置模型选择，默认适配 ComfyUI 模型目录；本地没有模型时才会自动下载支持的 Hugging Face 模型。

## Features / 功能

- Qwen3-VL local model loading, limited to practical 4B and 8B options
- Qwen3.5 multimodal model loading through separate nodes, limited to practical local options
- Built-in model dropdown inside each functional node
- Hugging Face model ID and ComfyUI model folder support
- Image understanding and visual question answering
- Multi-image input through ComfyUI IMAGE batches
- Video understanding through frame batches
- OCR-oriented prompting
- Grounding and detection-oriented prompting
- JSON-oriented structured output
- Extracted JSON and bounding-box helper outputs
- 4-bit / 8-bit quantized loading options when `bitsandbytes` supports your Python/CUDA build

## Nodes / 节点

- `QwenVL-Smit 图片理解`
  Selects a Qwen3-VL model inside the node and runs image understanding, VQA, OCR, grounding, detection, captioning, or custom prompts. Supports up to 6 image inputs.

- `QwenVL-Smit 视频理解`
  Selects a Qwen3-VL model inside the node and treats IMAGE batches as video frames. Supports up to 3 video-frame inputs and can downsample frames before inference.

- `Qwen3.5-Smit 文本对话`
  Selects a Qwen3.5 model inside the node and runs text-only chat.

- `Qwen3.5-Smit 图片理解`
  Selects a Qwen3.5 model inside the node and runs image or multi-image understanding. Supports up to 6 image inputs.

- `Qwen3.5-Smit 视频理解`
  Selects a Qwen3.5 model inside the node and runs video understanding from IMAGE frame batches. Supports up to 3 video-frame inputs.

- `QwenVL-Smit 提示词预设`
  Builds reusable prompt text for common tasks. It is optional; use it when you want to share one task preset with several nodes or keep a workflow cleaner.

- `QwenVL-Smit 清理缓存`
  Clears the in-process model cache and CUDA cache.

## Installation / 安装

### ComfyUI Manager

After this repository is published on GitHub, other users can install it with ComfyUI Manager:

1. Open ComfyUI Manager
2. Choose `Install via Git URL` or `Install from Git URL`
3. Paste:

```text
https://github.com/jiuman994/Qwenvl-Smit
```

4. Restart ComfyUI

发布到 GitHub 后，其他用户可以通过 ComfyUI Manager 的 Git URL 安装。节点代码没有硬编码任何本机路径，会跟随用户自己的 `ComfyUI/custom_nodes/Qwenvl-Smit` 安装目录加载。

### Manual Install

Clone this repository into `ComfyUI/custom_nodes`:

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

安装完成后请重启 ComfyUI。

## Recommended Environment / 推荐环境

Recommended environment:

- Python `3.10` or newer
- CUDA-enabled PyTorch for GPU inference
- GPU VRAM: 8GB or higher
- `transformers>=4.57.0`
- `qwen-vl-utils[decord]>=0.0.14`
- `accelerate>=1.0.0`
- `huggingface_hub>=0.34.0`
- `bitsandbytes` is optional but recommended for 4-bit / 8-bit loading

### VRAM Guide / 显存建议

The dropdown intentionally avoids very large models because most local ComfyUI users are limited by VRAM.

模型下拉框会主动避开过大的模型，因为大多数本地 ComfyUI 环境都受显存限制。

| VRAM | Qwen3-VL practical maximum | Qwen3.5 practical maximum | Suggested settings |
| --- | --- | --- | --- |
| 8GB | 4B, preferably 4-bit | 4B or 9B with 4-bit | `4bit`, 16-32 video frames |
| 16GB | 8B with 4-bit or FP8 | 9B or 27B with 4-bit | `4bit` or FP8 models |
| 24GB | 8B comfortably | 27B or 35B-A3B with 4-bit | `4bit`, reduce video frames if needed |

Qwen3-VL exposes only 4B and 8B model presets. Qwen3.5 exposes 4B, 9B, 27B, and 35B-A3B presets. Larger 30B/32B/100B+ style models are intentionally not shown in the default UI.

Qwen3-VL 默认只展示 4B 和 8B。Qwen3.5 默认展示 4B、9B、27B、35B-A3B。30B/32B/100B+ 这类对本地 ComfyUI 用户不现实的模型不会出现在默认列表里。

## Model Selection / 模型选择

Each functional node has its own `模型` dropdown:

1. Local models detected in ComfyUI model folders are listed first.
2. Supported Hugging Face model IDs are listed after local models.
3. If a selected Hugging Face model is not local yet, it downloads into the matching ComfyUI model folder.

每个功能节点都内置一个 `模型` 下拉框：

1. 优先显示已经放在 ComfyUI 模型目录里的本地模型。
2. 本地模型后面显示支持的 Hugging Face 模型。
3. 如果选择的 Hugging Face 模型还没有下载，会自动下载到对应的 ComfyUI 模型目录。

Recommended local model layout:

```text
ComfyUI/
  models/
    LLM/
      Qwen-VL/
        Qwen3-VL-4B-Instruct/
          config.json
          model.safetensors.index.json
          model-00001-of-000xx.safetensors
          ...
```

推荐把本地模型放在：

```text
ComfyUI/models/LLM/Qwen-VL/模型文件夹
```

The node also scans common existing folders when present:

```text
ComfyUI/models/LLM/Qwen-VL
ComfyUI/models/LLM
ComfyUI/models/VQA
ComfyUI/models/hugface
ComfyUI/models/transformers
```

节点会优先读取 ComfyUI 自己的模型目录。只有本地没有可用模型时，才会使用 Hugging Face 模型 ID 自动下载。默认下载位置也是 `ComfyUI/models/LLM/Qwen-VL/模型文件夹`，更接近现有 QwenVL 节点的习惯。

Examples:

```text
Qwen/Qwen3-VL-4B-Instruct
ComfyUI/models/LLM/Qwen-VL/Qwen3-VL-4B-Instruct
```

Qwen3.5 uses separate nodes and a separate recommended local folder:

```text
ComfyUI/models/LLM/Qwen3.5/Qwen3.5-4B
```

Qwen3.5 使用独立节点，推荐本地模型目录：

```text
ComfyUI/models/LLM/Qwen3.5/模型文件夹
```

Both Qwen3-VL and Qwen3.5 nodes use `transformers.AutoModelForImageTextToText` plus `AutoProcessor`, matching the multimodal model interface. The Qwen3.5 chat node simply runs the same model in text-only mode.

Qwen3-VL 和 Qwen3.5 节点都使用 `transformers.AutoModelForImageTextToText` 与 `AutoProcessor`。Qwen3.5 文本对话节点只是把同一个多模态模型按纯文本方式调用。

## Prompt Preset Node / 提示词预设节点

`QwenVL-Smit 提示词预设` does not run a model. It only outputs reusable prompt text for common tasks such as image captioning, OCR, visual question answering, object detection, grounding, and JSON output.

`QwenVL-Smit 提示词预设` 不会加载或运行模型，它只是输出一段可复用的提示词。适合以下情况：

- 多个节点复用同一种任务描述。
- 想把“任务类型”和“补充要求”拆出来，让主节点更干净。
- 想快速生成 OCR、视觉问答、目标检测、目标定位、JSON 结构化输出等提示词。

If you prefer simple workflows, you can ignore this node and write prompts directly inside the image, video, or chat nodes.

如果你喜欢简单工作流，可以完全不使用这个节点，直接在图片、视频或文本节点里填写提示词。

## Basic Workflows / 基础工作流

Image OCR:

```text
Load Image -> QwenVL-Smit 图片理解
任务类型: OCR文字识别
提示词: 提取图片中的全部文字。
```

Multi-image:

```text
Load Image -> 图片1
Load Image -> 图片2
Load Image -> 图片3
QwenVL-Smit 图片理解
```

Visual question answering:

```text
Load Image -> QwenVL-Smit 图片理解
任务类型: 视觉问答
提示词: 画面主体是什么？它正在做什么？
```

Detection / grounding:

```text
Load Image -> QwenVL-Smit 图片理解
任务类型: 目标检测
强制JSON: true
提示词: 检测画面中所有可见的商品包装，返回类别和坐标。
```

Video understanding:

```text
Load Video frames as IMAGE batch -> QwenVL-Smit 视频理解
任务类型: 图像描述
最大帧数: 32
提示词: 总结这个视频的时间线。
```

## Outputs / 输出

`QwenVL-Smit 图片理解` and `QwenVL-Smit 视频理解` return:

- `文本`: raw model answer
- `JSON`: parsed JSON if the answer contains valid JSON
- `坐标JSON`: helper extraction for simple `[x1, y1, x2, y2]` or `<box>...</box>` patterns

The model controls the final answer format. For stricter output, enable `强制JSON` and describe the schema in the prompt.

最终输出格式仍由模型生成决定。如果需要更严格的结构化结果，请开启 `force_json` 并在提示词里写清楚 schema。

## Notes / 注意事项

- Large QwenVL models can require substantial VRAM. Start with the 4B model on 8GB GPUs.
- `bitsandbytes` support depends on Python, CUDA, PyTorch, and platform compatibility.
- Video input is expected as a ComfyUI `IMAGE` batch. Use existing video loader nodes to decode videos into frames.
- Model downloads are stored under the matching ComfyUI model folder.
- Some gated models may require `huggingface-cli login`.

## Model Sources and Credits / 模型来源与声明

This ComfyUI extension is an integration layer. It does not include Qwen model weights.

本项目只是 ComfyUI 节点适配层，不包含任何 Qwen 模型权重。

- Qwen3-VL models are provided by the Qwen team on Hugging Face, such as `Qwen/Qwen3-VL-4B-Instruct` and `Qwen/Qwen3-VL-8B-Instruct`.
- Qwen3.5 models are provided by the Qwen team on Hugging Face, such as `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-27B`, and `Qwen/Qwen3.5-35B-A3B`.
- Qwenvl-Smit adapts those models for local ComfyUI workflows, with simplified Chinese UI labels, ComfyUI model-folder discovery, multi-image inputs, video-frame inputs, JSON helper outputs, and local-friendly model presets.
- Please follow the license and use policy of the selected upstream Qwen model. Qwenvl-Smit itself is released under Apache-2.0.

Qwen3-VL 与 Qwen3.5 模型由 Qwen 团队提供。本项目在这些模型的基础上做 ComfyUI 节点适配，包括中文界面、本地模型目录识别、多图输入、视频帧输入、JSON 辅助输出和适合本地显存的模型列表。使用模型时请遵守对应 Qwen 模型的开源协议与使用政策；Qwenvl-Smit 节点代码本身使用 Apache-2.0。

## Development / 开发

Run a syntax check:

```bash
python -m py_compile __init__.py nodes.py
```

## License / 许可证

Apache-2.0. See [LICENSE](LICENSE).
