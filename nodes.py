import gc
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from PIL import Image

try:
    import torch
except Exception:  # pragma: no cover - ComfyUI will surface this in node errors.
    torch = None

try:
    from comfy import model_management
except Exception:  # pragma: no cover - Allows basic import outside ComfyUI.
    model_management = None

try:
    import folder_paths
except Exception:  # pragma: no cover - Allows basic import outside ComfyUI.
    folder_paths = None


NODE_CATEGORY = "QwenVL-Smit"
DEFAULT_MODEL = "Qwen3-VL-4B-Instruct"
LEGACY_QWENVL_DIR = os.path.join("LLM", "Qwen-VL")
DEFAULT_IMAGE_MIN_PIXELS = 3136
DEFAULT_IMAGE_MAX_PIXELS = 1003520
DEFAULT_VIDEO_MAX_PIXELS = 200704

MODEL_PRESETS = [
    "Qwen3-VL-4B-Instruct",
    "Qwen3-VL-4B-Thinking",
    "Qwen3-VL-8B-Instruct",
    "Qwen3-VL-8B-Thinking",
]

MODEL_PRESET_IDS = {name: f"Qwen/{name}" for name in MODEL_PRESETS}

TASK_PRESETS = {
    "自定义": "",
    "图像描述": "Describe the visual content in detail.",
    "视觉问答": "Answer the user's question based only on the visual content.",
    "OCR文字识别": "Extract all visible text. Preserve reading order and line breaks where possible.",
    "目标定位": (
        "Find the object or region requested by the user. Return concise reasoning and any "
        "available coordinates or bounding boxes."
    ),
    "目标检测": (
        "Detect the requested objects. Return JSON with an objects array. Each object should "
        "include label, confidence if known, and bbox if available."
    ),
    "JSON结构化输出": "Return only valid JSON that matches the user's requested schema.",
}

MODEL_CACHE: Dict[str, "QwenVLModelBundle"] = {}


class AnyType(str):
    def __ne__(self, other: object) -> bool:
        return False


ANY_TYPE = AnyType("*")


def _register_model_dirs():
    if folder_paths is None:
        return
    try:
        base_models_dir = folder_paths.models_dir
        folder_paths.add_model_folder_path(
            "qwen_vl",
            os.path.join(base_models_dir, LEGACY_QWENVL_DIR),
        )
    except Exception:
        pass


_register_model_dirs()


@dataclass
class QwenVLModelBundle:
    model: Any
    processor: Any
    model_id: str
    dtype: str
    device_map: str
    quantization: str


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required. Install torch in the ComfyUI Python environment.")


def _looks_like_hf_model_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    required = ["config.json"]
    optional = [
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "model.safetensors",
        "pytorch_model.bin",
    ]
    return all(os.path.exists(os.path.join(path, name)) for name in required) and any(
        os.path.exists(os.path.join(path, name)) for name in optional
    )


def _models_dir() -> str:
    if folder_paths is not None:
        try:
            return folder_paths.models_dir
        except Exception:
            pass
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _primary_qwen_vl_models_dir() -> str:
    path = os.path.join(_models_dir(), LEGACY_QWENVL_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def _comfy_model_roots() -> List[str]:
    roots = []
    if folder_paths is not None:
        for key in ["qwen_vl", "LLM", "llm", "VQA", "hugface", "transformers"]:
            try:
                roots.extend(folder_paths.get_folder_paths(key))
            except Exception:
                pass
    base = _models_dir()
    roots.extend(
        [
            os.path.join(base, LEGACY_QWENVL_DIR),
            os.path.join(base, "LLM"),
            os.path.join(base, "VQA"),
            os.path.join(base, "hugface"),
            os.path.join(base, "transformers"),
        ]
    )
    return [root for root in dict.fromkeys(roots) if os.path.isdir(root)]


def _iter_local_hf_models():
    for root in _comfy_model_roots():
        for current, dirs, _files in os.walk(root):
            if _looks_like_hf_model_dir(current):
                label = os.path.basename(os.path.normpath(current))
                lowered = label.lower()
                if "qwen-vl" in lowered or "qwen3-vl" in lowered or "qwenvl" in lowered:
                    yield label, current
                dirs[:] = []


def _local_model_map() -> Dict[str, str]:
    return _dedupe_model_map(_iter_local_hf_models())


def _dedupe_model_map(items) -> Dict[str, str]:
    result: Dict[str, str] = {}
    used = set()
    for label, path in items:
        clean = os.path.basename(os.path.normpath(label)) or label
        key = clean.lower()
        if key in used:
            continue
        used.add(key)
        result[clean] = path
    return result


def _model_sort_key(name: str):
    lowered = name.lower()
    size_match = re.search(r"(\d+(?:\.\d+)?)\s*b", lowered)
    size = float(size_match.group(1)) if size_match else 9999.0
    kind = 0 if "instruct" in lowered else 1 if "thinking" in lowered else 2
    return (size, kind, lowered)


def _list_comfy_qwen_vl_models() -> List[str]:
    return sorted(_local_model_map().keys(), key=_model_sort_key)


def _resolve_comfy_model_name(model_name: str) -> str:
    model_name = (model_name or "").strip()
    if not model_name or model_name == "none":
        return ""
    return _local_model_map().get(model_name, "")


def _first_comfy_model_path() -> str:
    for _label, path in _iter_local_hf_models():
        return path
    return ""


def _is_repo_id(value: str) -> bool:
    value = (value or "").strip()
    return "/" in value and not os.path.exists(value) and not re.match(r"^[A-Za-z]:[\\/]", value)


def _hf_repo_folder_name(repo_id: str) -> str:
    return repo_id.rstrip("/").split("/")[-1]


def _find_repo_in_local_models(repo_id: str) -> str:
    expected = _hf_repo_folder_name(repo_id).lower()
    for _label, path in _iter_local_hf_models():
        if os.path.basename(os.path.normpath(path)).lower() == expected:
            return path
    return ""


def _ensure_repo_in_comfy_models(repo_id: str) -> str:
    target = os.path.join(_primary_qwen_vl_models_dir(), _hf_repo_folder_name(repo_id))
    if _looks_like_hf_model_dir(target):
        print(f"[QwenVL-Smit] Using local model: {target}")
        return target
    return _download_repo_to_dir(
        repo_id=repo_id,
        target=target,
        folder_hint="ComfyUI/models/LLM/Qwen-VL",
    )


def _download_repo_to_dir(repo_id: str, target: str, folder_hint: str) -> str:
    os.makedirs(target, exist_ok=True)
    errors = []
    source = (os.environ.get("QWENVL_SMIT_DOWNLOAD_SOURCE") or "auto").strip().lower()
    if source not in {"auto", "modelscope", "huggingface"}:
        source = "auto"

    if source in {"auto", "modelscope"}:
        try:
            from modelscope import snapshot_download as modelscope_snapshot_download

            print(f"[QwenVL-Smit] Downloading {repo_id} from ModelScope to {target}")
            modelscope_snapshot_download(repo_id, local_dir=target)
            return target
        except Exception as exc:
            errors.append(f"ModelScope: {type(exc).__name__}: {exc}")
            if source == "modelscope":
                return _raise_download_error(repo_id, target, folder_hint, errors)

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        errors.append(f"Hugging Face: missing huggingface_hub: {exc}")
        return _raise_download_error(repo_id, target, folder_hint, errors)

    endpoint = os.environ.get("QWENVL_SMIT_HF_ENDPOINT") or os.environ.get("HF_ENDPOINT")
    endpoint_text = f" via {endpoint}" if endpoint else ""
    print(f"[QwenVL-Smit] Downloading {repo_id} to {target}{endpoint_text}")
    try:
        kwargs = {
            "repo_id": repo_id,
            "local_dir": target,
            "local_dir_use_symlinks": False,
            "ignore_patterns": ["*.md", ".git*", ".gitattributes"],
            "resume_download": True,
        }
        if endpoint:
            kwargs["endpoint"] = endpoint
        snapshot_download(**kwargs)
    except Exception as exc:
        errors.append(f"Hugging Face: {type(exc).__name__}: {exc}")
        return _raise_download_error(repo_id, target, folder_hint, errors)
    return target


def _raise_download_error(repo_id: str, target: str, folder_hint: str, errors: List[str]) -> str:
    detail = "\n".join(f"- {item}" for item in errors) if errors else "- unknown error"
    raise RuntimeError(
        f"QwenVL-Smit 自动下载模型失败：{repo_id}\n\n"
        "下载顺序：本地模型目录 -> ModelScope 魔搭社区 -> Hugging Face / HF_ENDPOINT。\n\n"
        "常见原因：\n"
        "1. 当前网络无法连接 ModelScope 或 Hugging Face。\n"
        "2. 模型需要登录、权限或接受协议。\n"
        "3. 本地目录里只有部分下载文件，模型不完整。\n"
        "4. 未安装可选的 modelscope 包时，会自动跳过 ModelScope 并尝试 Hugging Face。\n\n"
        "处理方式：\n"
        f"- 手动下载模型，并把完整模型文件夹放到：{folder_hint}\n"
        "- 国内网络可以优先安装可选依赖：pip install modelscope\n"
        "- 也可以在启动 ComfyUI 前设置 Hugging Face 镜像："
        "HF_ENDPOINT=https://hf-mirror.com\n"
        "- 如需强制下载源，可设置 QWENVL_SMIT_DOWNLOAD_SOURCE=modelscope 或 huggingface\n"
        "- 如果模型需要权限，请先执行：huggingface-cli login\n\n"
        f"当前目标目录：{target}\n"
        f"原始错误：\n{detail}"
    )


def _resolve_selected_vl_model(model_name: str) -> str:
    model_name = (model_name or "").strip()
    local_path = _resolve_comfy_model_name(model_name)
    if local_path:
        return local_path
    if model_name in MODEL_PRESET_IDS:
        local_repo = _find_repo_in_local_models(MODEL_PRESET_IDS[model_name])
        if local_repo:
            return local_repo
        return _ensure_repo_in_comfy_models(MODEL_PRESET_IDS[model_name])
    if _is_repo_id(model_name):
        local_repo = _find_repo_in_local_models(model_name)
        if local_repo:
            return local_repo
        return _ensure_repo_in_comfy_models(model_name)
    return _ensure_repo_in_comfy_models(MODEL_PRESET_IDS[DEFAULT_MODEL])


def _vl_model_choices() -> List[str]:
    local_models = _list_comfy_qwen_vl_models()
    local_keys = {name.lower() for name in local_models}
    presets = [
        preset for preset in MODEL_PRESETS
        if preset.lower() not in local_keys
    ]
    return sorted(list(dict.fromkeys(local_models + presets)), key=_model_sort_key)


def _load_selected_vl_bundle(模型, 精度, 设备, 量化, 注意力模式) -> QwenVLModelBundle:
    return _load_model(
        _resolve_selected_vl_model(模型),
        "",
        _ui_dtype(精度),
        _ui_device_map(设备),
        _ui_quantization(量化),
        _ui_attention(注意力模式),
        True,
    )


def _torch_dtype(dtype_name: str):
    _require_torch()
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    return "auto"


def _ui_dtype(value: str) -> str:
    return {"自动": "auto"}.get(value, value)


def _ui_device_map(value: str) -> str:
    return {"自动": "auto", "CUDA": "cuda", "CPU": "cpu"}.get(value, "auto")


def _ui_quantization(value: str) -> str:
    return {"不量化": "none"}.get(value, value)


def _ui_attention(value: str) -> str:
    return {
        "自动": "auto",
        "SDPA": "sdpa",
        "Flash Attention 2": "flash_attention_2",
        "Eager": "eager",
    }.get(value, "auto")


def _device_map_value(device_map: str):
    if device_map == "cuda":
        return {"": 0}
    if device_map == "cpu":
        return {"": "cpu"}
    return device_map


def _cache_key(
    model_id: str,
    cache_dir: str,
    dtype: str,
    device_map: str,
    quantization: str,
    attention_implementation: str,
    trust_remote_code: bool,
) -> str:
    return json.dumps(
        {
            "model_id": model_id,
            "cache_dir": cache_dir,
            "dtype": dtype,
            "device_map": device_map,
            "quantization": quantization,
            "attn": attention_implementation,
            "trust": trust_remote_code,
        },
        sort_keys=True,
    )


def _raise_transformers_load_error(exc: Exception, model_id: str) -> None:
    message = str(exc)
    if "does not recognize this architecture" in message:
        raise RuntimeError(
            f"QwenVL-Smit 无法加载模型架构：{model_id}\n\n"
            "模型已经被读取到 config.json，但当前 ComfyUI Python 环境中的 transformers "
            "还不认识该模型架构。\n\n"
            "处理方式：\n"
            "1. 先尝试升级 transformers：python -m pip install -U transformers\n"
            "2. 如果正式版仍不支持该模型，可以安装源码版："
            "python -m pip install -U git+https://github.com/huggingface/transformers.git\n"
            "3. 安装完成后必须完全重启 ComfyUI。\n\n"
            "便携包用户请使用 ComfyUI 自带的 Python 执行上面的 pip 命令，不要用系统 Python。\n\n"
            f"原始错误：{type(exc).__name__}: {exc}"
        ) from exc
    raise exc


def _load_model(
    model_id: str,
    cache_dir: str,
    dtype: str,
    device_map: str,
    quantization: str,
    attention_implementation: str,
    trust_remote_code: bool,
) -> QwenVLModelBundle:
    _require_torch()
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except Exception as exc:
        raise RuntimeError(
            "Missing transformers. Install requirements.txt in the ComfyUI Python environment."
        ) from exc

    key = _cache_key(
        model_id,
        cache_dir,
        dtype,
        device_map,
        quantization,
        attention_implementation,
        trust_remote_code,
    )
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    kwargs: Dict[str, Any] = {
        "torch_dtype": _torch_dtype(dtype),
        "device_map": _device_map_value(device_map),
        "trust_remote_code": trust_remote_code,
    }
    processor_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}

    if cache_dir.strip():
        kwargs["cache_dir"] = cache_dir.strip()
        processor_kwargs["cache_dir"] = cache_dir.strip()

    if attention_implementation != "auto":
        kwargs["attn_implementation"] = attention_implementation

    if quantization in {"4bit", "8bit"}:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError(
                "4bit/8bit loading requires bitsandbytes and a compatible CUDA build."
            ) from exc
        if quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    try:
        processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    except Exception as exc:
        _raise_transformers_load_error(exc, model_id)
    model.eval()

    bundle = QwenVLModelBundle(
        model=model,
        processor=processor,
        model_id=model_id,
        dtype=dtype,
        device_map=device_map,
        quantization=quantization,
    )
    MODEL_CACHE[key] = bundle
    return bundle


def _image_tensor_to_pil_list(images) -> List[Image.Image]:
    if images is None:
        return []

    if torch is not None and isinstance(images, torch.Tensor):
        array = images.detach().cpu().float().numpy()
    else:
        array = np.asarray(images)

    if array.ndim == 3:
        array = array[None, ...]

    pil_images = []
    for img in array:
        img = np.clip(img, 0, 1)
        img = (img * 255.0).round().astype(np.uint8)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        pil_images.append(Image.fromarray(img[..., :3], "RGB"))
    return pil_images


def _collect_pil_images(*image_inputs) -> List[Image.Image]:
    collected: List[Image.Image] = []
    for image_input in image_inputs:
        if image_input is None:
            continue
        collected.extend(_image_tensor_to_pil_list(image_input))
    return collected


def _select_frames(images: List[Image.Image], max_frames: int) -> List[Image.Image]:
    if max_frames <= 0 or len(images) <= max_frames:
        return images
    indexes = np.linspace(0, len(images) - 1, max_frames).round().astype(int).tolist()
    return [images[i] for i in indexes]


def _combine_prompt(task_preset: str, prompt: str, force_json: bool) -> str:
    preset = TASK_PRESETS.get(task_preset, "")
    prompt = (prompt or "").strip()
    parts = [part for part in [preset, prompt] if part]
    combined = "\n\n".join(parts) if parts else "Describe the visual content."
    if force_json and "json" not in combined.lower():
        combined += "\n\nReturn only valid JSON."
    return combined


def _parse_history(history_json: str) -> List[Dict[str, Any]]:
    history_json = (history_json or "").strip()
    if not history_json:
        return []
    try:
        parsed = json.loads(history_json)
    except json.JSONDecodeError as exc:
        raise ValueError("history_json must be valid JSON.") from exc
    if not isinstance(parsed, list):
        raise ValueError("history_json must be a JSON array of chat messages.")
    return parsed


def _extract_json(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            return json.dumps(json.loads(candidate), ensure_ascii=False, indent=2)
        except Exception:
            pass

    start = min([i for i in [text.find("{"), text.find("[")] if i >= 0], default=-1)
    if start >= 0:
        for end in range(len(text), start, -1):
            candidate = text[start:end].strip()
            try:
                return json.dumps(json.loads(candidate), ensure_ascii=False, indent=2)
            except Exception:
                continue
    return ""


def _extract_boxes(text: str) -> str:
    boxes = []
    patterns = [
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        r"<box>\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?\s*</box>",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text or "", re.IGNORECASE):
            values = [float(v) for v in match.groups()]
            boxes.append({"bbox": values})
    return json.dumps({"boxes": boxes}, ensure_ascii=False, indent=2) if boxes else ""


def _build_visual_content(
    images: List[Image.Image],
    prompt: str,
    mode: str,
    fps: float,
    min_pixels: int,
    max_pixels: int,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if mode == "video":
        content.append(
            {
                "type": "video",
                "video": images,
                "fps": fps,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            }
        )
    else:
        for image in images:
            content.append({"type": "image", "image": image, "min_pixels": min_pixels, "max_pixels": max_pixels})
    content.append({"type": "text", "text": prompt})
    return content


def _run_generation(
    bundle: QwenVLModelBundle,
    images: List[Image.Image],
    prompt: str,
    system_prompt: str,
    history_json: str,
    mode: str,
    fps: float,
    min_pixels: int,
    max_pixels: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> str:
    _require_torch()
    try:
        from qwen_vl_utils import process_vision_info
    except Exception as exc:
        raise RuntimeError(
            "Missing qwen-vl-utils. Install requirements.txt in the ComfyUI Python environment."
        ) from exc

    if seed >= 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.extend(_parse_history(history_json))
    user_content: Any
    if images:
        user_content = _build_visual_content(images, prompt, mode, fps, min_pixels, max_pixels)
    else:
        user_content = prompt.strip() or "请回答用户的问题。"
    messages.append(
        {
            "role": "user",
            "content": user_content,
        }
    )

    text = bundle.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if images:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
        )
        inputs = bundle.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
    else:
        inputs = bundle.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )

    target_device = None
    if hasattr(bundle.model, "device"):
        target_device = bundle.model.device
    elif torch.cuda.is_available() and bundle.device_map != "cpu":
        target_device = torch.device("cuda")
    else:
        target_device = torch.device("cpu")

    try:
        inputs = inputs.to(target_device)
    except Exception:
        pass

    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.inference_mode():
        generated_ids = bundle.model.generate(**inputs, **generation_kwargs)

    input_ids = inputs["input_ids"]
    generated_ids_trimmed = [
        output_ids[len(input_ids_item) :] for input_ids_item, output_ids in zip(input_ids, generated_ids)
    ]
    output_text = bundle.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0] if output_text else ""


class QwenVLSmitModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        models = _vl_model_choices()
        return {
            "required": {
                "模型": (models, {"default": models[0] if models else DEFAULT_MODEL}),
                "精度": (["自动", "bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "设备": (["自动", "CUDA", "CPU"], {"default": "自动"}),
                "量化": (["不量化", "4bit", "8bit"], {"default": "4bit"}),
                "注意力模式": (["自动", "SDPA", "Flash Attention 2", "Eager"], {"default": "自动"}),
            }
        }

    RETURN_TYPES = ("QWENVL_MODEL", "STRING")
    RETURN_NAMES = ("模型", "模型信息")
    FUNCTION = "load"
    CATEGORY = NODE_CATEGORY

    def load(
        self,
        模型,
        精度,
        设备,
        量化,
        注意力模式,
    ):
        model_id = _resolve_selected_vl_model(模型)
        bundle = _load_model(
            model_id,
            "",
            _ui_dtype(精度),
            _ui_device_map(设备),
            _ui_quantization(量化),
            _ui_attention(注意力模式),
            True,
        )
        info = {
            "model_id": bundle.model_id,
            "dtype": bundle.dtype,
            "device_map": bundle.device_map,
            "quantization": bundle.quantization,
        }
        return (bundle, json.dumps(info, ensure_ascii=False, indent=2))


class QwenVLSmitImage:
    @classmethod
    def INPUT_TYPES(cls):
        models = _vl_model_choices()
        return {
            "required": {
                "模型": (models, {"default": models[0] if models else DEFAULT_MODEL}),
                "精度": (["自动", "bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "设备": (["自动", "CUDA", "CPU"], {"default": "自动"}),
                "量化": (["不量化", "4bit", "8bit"], {"default": "4bit"}),
                "注意力模式": (["自动", "SDPA", "Flash Attention 2", "Eager"], {"default": "自动"}),
                "任务类型": (list(TASK_PRESETS.keys()), {"default": "自定义"}),
                "提示词": ("STRING", {"default": "请回答我的问题。", "multiline": True}),
                "系统提示词": ("STRING", {"default": "你是一个有帮助的智能问答助手。", "multiline": True}),
                "历史记录JSON": ("STRING", {"default": "", "multiline": True}),
                "强制JSON": ("BOOLEAN", {"default": False}),
                "最大输出Token": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "温度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "核采样": ("FLOAT", {"default": 0.9, "min": 0.01, "max": 1.0, "step": 0.01}),
                "随机种子": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFF, "step": 1}),
            },
            "optional": {
                "图片1": ("IMAGE",),
                "图片2": ("IMAGE",),
                "图片3": ("IMAGE",),
                "图片4": ("IMAGE",),
                "图片5": ("IMAGE",),
                "图片6": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("文本", "JSON", "坐标JSON")
    FUNCTION = "analyze"
    CATEGORY = NODE_CATEGORY

    def analyze(
        self,
        模型,
        精度,
        设备,
        量化,
        注意力模式,
        任务类型,
        提示词,
        系统提示词,
        历史记录JSON,
        强制JSON,
        最大输出Token,
        温度,
        核采样,
        随机种子,
        图片1=None,
        图片2=None,
        图片3=None,
        图片4=None,
        图片5=None,
        图片6=None,
    ):
        pil_images = _collect_pil_images(图片1, 图片2, 图片3, 图片4, 图片5, 图片6)
        final_prompt = _combine_prompt(任务类型, 提示词, 强制JSON)
        bundle = _load_selected_vl_bundle(模型, 精度, 设备, 量化, 注意力模式)
        text = _run_generation(
            bundle,
            pil_images,
            final_prompt,
            系统提示词,
            历史记录JSON,
            "image",
            1.0,
            DEFAULT_IMAGE_MIN_PIXELS,
            DEFAULT_IMAGE_MAX_PIXELS,
            最大输出Token,
            温度,
            核采样,
            随机种子,
        )
        return (text, _extract_json(text), _extract_boxes(text))


class QwenVLSmitVideo:
    @classmethod
    def INPUT_TYPES(cls):
        models = _vl_model_choices()
        return {
            "required": {
                "模型": (models, {"default": models[0] if models else DEFAULT_MODEL}),
                "精度": (["自动", "bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "设备": (["自动", "CUDA", "CPU"], {"default": "自动"}),
                "量化": (["不量化", "4bit", "8bit"], {"default": "4bit"}),
                "注意力模式": (["自动", "SDPA", "Flash Attention 2", "Eager"], {"default": "自动"}),
                "视频帧1": ("IMAGE",),
                "任务类型": (list(TASK_PRESETS.keys()), {"default": "自定义"}),
                "提示词": ("STRING", {"default": "请描述这个视频。", "multiline": True}),
                "系统提示词": ("STRING", {"default": "你是一个有帮助的视频理解助手。", "multiline": True}),
                "历史记录JSON": ("STRING", {"default": "", "multiline": True}),
                "强制JSON": ("BOOLEAN", {"default": False}),
                "帧率": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 60.0, "step": 0.01}),
                "最大帧数": ("INT", {"default": 32, "min": 1, "max": 512, "step": 1}),
                "最大输出Token": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "温度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "核采样": ("FLOAT", {"default": 0.9, "min": 0.01, "max": 1.0, "step": 0.01}),
                "随机种子": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFF, "step": 1}),
            },
            "optional": {
                "视频帧2": ("IMAGE",),
                "视频帧3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("文本", "JSON", "坐标JSON")
    FUNCTION = "analyze"
    CATEGORY = NODE_CATEGORY

    def analyze(
        self,
        模型,
        精度,
        设备,
        量化,
        注意力模式,
        视频帧1,
        任务类型,
        提示词,
        系统提示词,
        历史记录JSON,
        强制JSON,
        帧率,
        最大帧数,
        最大输出Token,
        温度,
        核采样,
        随机种子,
        视频帧2=None,
        视频帧3=None,
    ):
        pil_frames = _select_frames(_collect_pil_images(视频帧1, 视频帧2, 视频帧3), 最大帧数)
        final_prompt = _combine_prompt(任务类型, 提示词, 强制JSON)
        bundle = _load_selected_vl_bundle(模型, 精度, 设备, 量化, 注意力模式)
        text = _run_generation(
            bundle,
            pil_frames,
            final_prompt,
            系统提示词,
            历史记录JSON,
            "video",
            帧率,
            DEFAULT_IMAGE_MIN_PIXELS,
            DEFAULT_VIDEO_MAX_PIXELS,
            最大输出Token,
            温度,
            核采样,
            随机种子,
        )
        return (text, _extract_json(text), _extract_boxes(text))


class QwenVLSmitPromptPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "任务类型": (list(TASK_PRESETS.keys()), {"default": "图像描述"}),
                "补充要求": ("STRING", {"default": "", "multiline": True}),
                "强制JSON": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("提示词",)
    FUNCTION = "build"
    CATEGORY = NODE_CATEGORY

    def build(self, 任务类型, 补充要求, 强制JSON):
        return (_combine_prompt(任务类型, 补充要求, 强制JSON),)


class QwenVLSmitUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"清理全部模型缓存": ("BOOLEAN", {"default": True})},
            "optional": {"触发": (ANY_TYPE,)},
        }

    RETURN_TYPES = (ANY_TYPE, "STRING")
    RETURN_NAMES = ("触发", "状态")
    FUNCTION = "unload"
    CATEGORY = NODE_CATEGORY

    def unload(self, 清理全部模型缓存, 触发=None):
        if 清理全部模型缓存:
            MODEL_CACHE.clear()
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if model_management is not None:
            try:
                model_management.soft_empty_cache()
            except Exception:
                pass
        return (触发, "QwenVL-Smit 模型缓存已清理。")


NODE_CLASS_MAPPINGS = {
    "QwenVLSmitImage": QwenVLSmitImage,
    "QwenVLSmitVideo": QwenVLSmitVideo,
    "QwenVLSmitPromptPreset": QwenVLSmitPromptPreset,
    "QwenVLSmitUnload": QwenVLSmitUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVLSmitImage": "QwenVL-Smit 智能问答",
    "QwenVLSmitVideo": "QwenVL-Smit 视频理解",
    "QwenVLSmitPromptPreset": "QwenVL-Smit 提示词预设",
    "QwenVLSmitUnload": "QwenVL-Smit 清理缓存",
}
