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
DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
LEGACY_QWENVL_DIR = os.path.join("LLM", "Qwen-VL")
DEFAULT_QWEN35_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_IMAGE_MIN_PIXELS = 3136
DEFAULT_IMAGE_MAX_PIXELS = 1003520
DEFAULT_VIDEO_MAX_PIXELS = 200704

MODEL_PRESETS = [
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-4B-Thinking",
    "Qwen/Qwen3-VL-4B-Instruct-FP8",
    "Qwen/Qwen3-VL-4B-Thinking-FP8",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-8B-Thinking",
    "Qwen/Qwen3-VL-8B-Instruct-FP8",
    "Qwen/Qwen3-VL-8B-Thinking-FP8",
]

QWEN35_MODEL_PRESETS = [
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B",
]

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
QWEN35_CACHE: Dict[str, "Qwen35ModelBundle"] = {}


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


@dataclass
class Qwen35ModelBundle:
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


def _primary_qwen35_models_dir() -> str:
    path = os.path.join(_models_dir(), "LLM", "Qwen3.5")
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


def _qwen35_model_roots() -> List[str]:
    base = _models_dir()
    roots = [
        os.path.join(base, "LLM", "Qwen3.5"),
        os.path.join(base, "LLM"),
        os.path.join(base, "llm"),
        os.path.join(base, "transformers"),
    ]
    if folder_paths is not None:
        for key in ["LLM", "llm", "transformers"]:
            try:
                roots.extend(folder_paths.get_folder_paths(key))
            except Exception:
                pass
    return [root for root in dict.fromkeys(roots) if os.path.isdir(root)]


def _local_model_label(root: str, model_dir: str) -> str:
    root_name = os.path.basename(os.path.normpath(root))
    parent_name = os.path.basename(os.path.dirname(os.path.normpath(root)))
    rel = os.path.relpath(model_dir, root).replace("\\", "/")
    if root_name == "Qwen-VL" and parent_name == "LLM":
        return f"LLM/Qwen-VL/{rel}"
    return f"{root_name}/{rel}"


def _iter_local_hf_models():
    for root in _comfy_model_roots():
        for current, dirs, _files in os.walk(root):
            if _looks_like_hf_model_dir(current):
                label = _local_model_label(root, current)
                if "qwen" in label.lower():
                    yield label, current
                dirs[:] = []


def _local_model_map() -> Dict[str, str]:
    return {label: path for label, path in _iter_local_hf_models()}


def _iter_local_qwen35_models():
    for root in _qwen35_model_roots():
        for current, dirs, _files in os.walk(root):
            if _looks_like_hf_model_dir(current):
                label = _local_model_label(root, current)
                lowered = label.lower()
                if "qwen3.5" in lowered or "qwen3_5" in lowered or "qwen-3.5" in lowered:
                    yield label, current
                dirs[:] = []


def _qwen35_model_map() -> Dict[str, str]:
    return {label: path for label, path in _iter_local_qwen35_models()}


def _list_comfy_qwen35_models() -> List[str]:
    return sorted(_qwen35_model_map().keys(), key=str.lower)


def _resolve_qwen35_comfy_model(model_name: str) -> str:
    model_name = (model_name or "").strip()
    if not model_name or model_name == "none":
        return ""
    return _qwen35_model_map().get(model_name, "")


def _first_qwen35_model_path() -> str:
    for _label, path in _iter_local_qwen35_models():
        return path
    return ""


def _list_comfy_qwen_vl_models() -> List[str]:
    return sorted(_local_model_map().keys(), key=str.lower)


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


def _find_qwen35_repo_in_local_models(repo_id: str) -> str:
    expected = _hf_repo_folder_name(repo_id).lower()
    for _label, path in _iter_local_qwen35_models():
        if os.path.basename(os.path.normpath(path)).lower() == expected:
            return path
    return ""


def _ensure_repo_in_comfy_models(repo_id: str) -> str:
    target = os.path.join(_primary_qwen_vl_models_dir(), _hf_repo_folder_name(repo_id))
    if _looks_like_hf_model_dir(target):
        print(f"[QwenVL-Smit] Using local model: {target}")
        return target

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required for automatic model downloads. "
            "Install requirements.txt or place the model under ComfyUI/models/LLM/Qwen-VL."
        ) from exc

    os.makedirs(target, exist_ok=True)
    print(f"[QwenVL-Smit] Downloading {repo_id} to {target}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=target,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md", "*.txt", ".git*", ".gitattributes"],
        resume_download=True,
    )
    return target


def _ensure_qwen35_repo_in_comfy_models(repo_id: str) -> str:
    target = os.path.join(_primary_qwen35_models_dir(), _hf_repo_folder_name(repo_id))
    if _looks_like_hf_model_dir(target):
        print(f"[QwenVL-Smit] Using local Qwen3.5 model: {target}")
        return target

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required for automatic model downloads. "
            "Install requirements.txt or place the model under ComfyUI/models/LLM/Qwen3.5."
        ) from exc

    os.makedirs(target, exist_ok=True)
    print(f"[QwenVL-Smit] Downloading {repo_id} to {target}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=target,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md", "*.txt", ".git*", ".gitattributes"],
        resume_download=True,
    )
    return target


def _resolve_selected_vl_model(model_name: str) -> str:
    model_name = (model_name or "").strip()
    local_path = _resolve_comfy_model_name(model_name)
    if local_path:
        return local_path
    if _is_repo_id(model_name):
        local_repo = _find_repo_in_local_models(model_name)
        if local_repo:
            return local_repo
        return _ensure_repo_in_comfy_models(model_name)
    return _ensure_repo_in_comfy_models(DEFAULT_MODEL)


def _resolve_selected_qwen35_model(model_name: str) -> str:
    model_name = (model_name or "").strip()
    local_path = _resolve_qwen35_comfy_model(model_name)
    if local_path:
        return local_path
    if _is_repo_id(model_name):
        local_repo = _find_qwen35_repo_in_local_models(model_name)
        if local_repo:
            return local_repo
        return _ensure_qwen35_repo_in_comfy_models(model_name)
    return _ensure_qwen35_repo_in_comfy_models(DEFAULT_QWEN35_MODEL)


def _vl_model_choices() -> List[str]:
    return list(dict.fromkeys(_list_comfy_qwen_vl_models() + MODEL_PRESETS))


def _qwen35_model_choices() -> List[str]:
    return list(dict.fromkeys(_list_comfy_qwen35_models() + QWEN35_MODEL_PRESETS))


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


def _load_selected_qwen35_bundle(模型, 精度, 设备, 量化, 注意力模式) -> Qwen35ModelBundle:
    return _load_qwen35_model(
        _resolve_selected_qwen35_model(模型),
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

    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
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


def _load_qwen35_model(
    model_id: str,
    cache_dir: str,
    dtype: str,
    device_map: str,
    quantization: str,
    attention_implementation: str,
    trust_remote_code: bool,
) -> Qwen35ModelBundle:
    _require_torch()
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except Exception as exc:
        raise RuntimeError(
            "Missing transformers. Install requirements.txt in the ComfyUI Python environment."
        ) from exc

    key = "qwen35:" + _cache_key(
        model_id,
        cache_dir,
        dtype,
        device_map,
        quantization,
        attention_implementation,
        trust_remote_code,
    )
    if key in QWEN35_CACHE:
        return QWEN35_CACHE[key]

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

    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.eval()

    bundle = Qwen35ModelBundle(
        model=model,
        processor=processor,
        model_id=model_id,
        dtype=dtype,
        device_map=device_map,
        quantization=quantization,
    )
    QWEN35_CACHE[key] = bundle
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

    if not images:
        raise ValueError("At least one IMAGE input is required.")

    if seed >= 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.extend(_parse_history(history_json))
    messages.append(
        {
            "role": "user",
            "content": _build_visual_content(images, prompt, mode, fps, min_pixels, max_pixels),
        }
    )

    text = bundle.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
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


def _run_qwen35_chat(
    bundle: Qwen35ModelBundle,
    prompt: str,
    system_prompt: str,
    history_json: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
) -> str:
    _require_torch()
    if seed >= 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.extend(_parse_history(history_json))
    messages.append({"role": "user", "content": (prompt or "").strip()})

    if hasattr(bundle.processor, "apply_chat_template"):
        text = bundle.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages]) + "\nassistant:"

    inputs = bundle.processor(text=[text], padding=True, return_tensors="pt")
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
        "repetition_penalty": repetition_penalty,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    tokenizer = getattr(bundle.processor, "tokenizer", None)
    if tokenizer is not None and tokenizer.pad_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    if tokenizer is not None and tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

    with torch.inference_mode():
        generated_ids = bundle.model.generate(**inputs, **generation_kwargs)

    input_len = inputs["input_ids"].shape[-1]
    output_ids = generated_ids[0, input_len:]
    if hasattr(bundle.processor, "decode"):
        return bundle.processor.decode(output_ids, skip_special_tokens=True).strip()
    if tokenizer is not None:
        return tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return str(output_ids)


class QwenVLSmitModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        models = _list_comfy_qwen_vl_models() + MODEL_PRESETS
        models = list(dict.fromkeys(models))
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


class Qwen35SmitModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        models = _list_comfy_qwen35_models() + QWEN35_MODEL_PRESETS
        models = list(dict.fromkeys(models))
        return {
            "required": {
                "模型": (models, {"default": models[0] if models else DEFAULT_QWEN35_MODEL}),
                "精度": (["自动", "bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "设备": (["自动", "CUDA", "CPU"], {"default": "自动"}),
                "量化": (["不量化", "4bit", "8bit"], {"default": "4bit"}),
                "注意力模式": (["自动", "SDPA", "Flash Attention 2", "Eager"], {"default": "自动"}),
            }
        }

    RETURN_TYPES = ("QWEN35_MODEL", "STRING")
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
        model_id = _resolve_selected_qwen35_model(模型)
        bundle = _load_qwen35_model(
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
            "type": "qwen3.5 multimodal model",
        }
        return (bundle, json.dumps(info, ensure_ascii=False, indent=2))


class Qwen35SmitChat:
    @classmethod
    def INPUT_TYPES(cls):
        models = _qwen35_model_choices()
        return {
            "required": {
                "模型": (models, {"default": models[0] if models else DEFAULT_QWEN35_MODEL}),
                "精度": (["自动", "bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "设备": (["自动", "CUDA", "CPU"], {"default": "自动"}),
                "量化": (["不量化", "4bit", "8bit"], {"default": "4bit"}),
                "注意力模式": (["自动", "SDPA", "Flash Attention 2", "Eager"], {"default": "自动"}),
                "提示词": ("STRING", {"default": "请给出简洁回答。", "multiline": True}),
                "系统提示词": ("STRING", {"default": "你是一个有帮助的助手。", "multiline": True}),
                "历史记录JSON": ("STRING", {"default": "", "multiline": True}),
                "强制JSON": ("BOOLEAN", {"default": False}),
                "最大输出Token": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "温度": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "核采样": ("FLOAT", {"default": 0.9, "min": 0.01, "max": 1.0, "step": 0.01}),
                "重复惩罚": ("FLOAT", {"default": 1.05, "min": 0.1, "max": 2.0, "step": 0.01}),
                "随机种子": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFF, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("文本", "JSON")
    FUNCTION = "chat"
    CATEGORY = NODE_CATEGORY

    def chat(
        self,
        模型,
        精度,
        设备,
        量化,
        注意力模式,
        提示词,
        系统提示词,
        历史记录JSON,
        强制JSON,
        最大输出Token,
        温度,
        核采样,
        重复惩罚,
        随机种子,
    ):
        final_prompt = (提示词 or "").strip()
        if 强制JSON and "json" not in final_prompt.lower():
            final_prompt += "\n\nReturn only valid JSON."
        bundle = _load_selected_qwen35_bundle(模型, 精度, 设备, 量化, 注意力模式)
        text = _run_qwen35_chat(
            bundle,
            final_prompt,
            系统提示词,
            历史记录JSON,
            最大输出Token,
            温度,
            核采样,
            重复惩罚,
            随机种子,
        )
        return (text, _extract_json(text))


class Qwen35SmitImage:
    @classmethod
    def INPUT_TYPES(cls):
        models = _qwen35_model_choices()
        return {
            "required": {
                "模型": (models, {"default": models[0] if models else DEFAULT_QWEN35_MODEL}),
                "精度": (["自动", "bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "设备": (["自动", "CUDA", "CPU"], {"default": "自动"}),
                "量化": (["不量化", "4bit", "8bit"], {"default": "4bit"}),
                "注意力模式": (["自动", "SDPA", "Flash Attention 2", "Eager"], {"default": "自动"}),
                "图片1": ("IMAGE",),
                "任务类型": (list(TASK_PRESETS.keys()), {"default": "自定义"}),
                "提示词": ("STRING", {"default": "请描述这张图片。", "multiline": True}),
                "系统提示词": ("STRING", {"default": "你是一个有帮助的多模态助手。", "multiline": True}),
                "历史记录JSON": ("STRING", {"default": "", "multiline": True}),
                "强制JSON": ("BOOLEAN", {"default": False}),
                "最大输出Token": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "温度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "核采样": ("FLOAT", {"default": 0.9, "min": 0.01, "max": 1.0, "step": 0.01}),
                "随机种子": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFF, "step": 1}),
            },
            "optional": {
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
        图片1,
        任务类型,
        提示词,
        系统提示词,
        历史记录JSON,
        强制JSON,
        最大输出Token,
        温度,
        核采样,
        随机种子,
        图片2=None,
        图片3=None,
        图片4=None,
        图片5=None,
        图片6=None,
    ):
        pil_images = _collect_pil_images(图片1, 图片2, 图片3, 图片4, 图片5, 图片6)
        final_prompt = _combine_prompt(任务类型, 提示词, 强制JSON)
        bundle = _load_selected_qwen35_bundle(模型, 精度, 设备, 量化, 注意力模式)
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


class Qwen35SmitVideo:
    @classmethod
    def INPUT_TYPES(cls):
        models = _qwen35_model_choices()
        return {
            "required": {
                "模型": (models, {"default": models[0] if models else DEFAULT_QWEN35_MODEL}),
                "精度": (["自动", "bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "设备": (["自动", "CUDA", "CPU"], {"default": "自动"}),
                "量化": (["不量化", "4bit", "8bit"], {"default": "4bit"}),
                "注意力模式": (["自动", "SDPA", "Flash Attention 2", "Eager"], {"default": "自动"}),
                "视频帧1": ("IMAGE",),
                "任务类型": (list(TASK_PRESETS.keys()), {"default": "自定义"}),
                "提示词": ("STRING", {"default": "请描述这个视频。", "multiline": True}),
                "系统提示词": ("STRING", {"default": "你是一个有帮助的多模态助手。", "multiline": True}),
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
        bundle = _load_selected_qwen35_bundle(模型, 精度, 设备, 量化, 注意力模式)
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
                "图片1": ("IMAGE",),
                "任务类型": (list(TASK_PRESETS.keys()), {"default": "自定义"}),
                "提示词": ("STRING", {"default": "请描述这张图片。", "multiline": True}),
                "系统提示词": ("STRING", {"default": "你是一个有帮助的视觉语言助手。", "multiline": True}),
                "历史记录JSON": ("STRING", {"default": "", "multiline": True}),
                "强制JSON": ("BOOLEAN", {"default": False}),
                "最大输出Token": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "温度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "核采样": ("FLOAT", {"default": 0.9, "min": 0.01, "max": 1.0, "step": 0.01}),
                "随机种子": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFF, "step": 1}),
            },
            "optional": {
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
        图片1,
        任务类型,
        提示词,
        系统提示词,
        历史记录JSON,
        强制JSON,
        最大输出Token,
        温度,
        核采样,
        随机种子,
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
        return {"required": {"清理全部模型缓存": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("状态",)
    FUNCTION = "unload"
    CATEGORY = NODE_CATEGORY

    def unload(self, 清理全部模型缓存):
        if 清理全部模型缓存:
            MODEL_CACHE.clear()
            QWEN35_CACHE.clear()
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if model_management is not None:
            try:
                model_management.soft_empty_cache()
            except Exception:
                pass
        return ("QwenVL-Smit 模型缓存已清理。",)


NODE_CLASS_MAPPINGS = {
    "QwenVLSmitImage": QwenVLSmitImage,
    "QwenVLSmitVideo": QwenVLSmitVideo,
    "Qwen35SmitChat": Qwen35SmitChat,
    "Qwen35SmitImage": Qwen35SmitImage,
    "Qwen35SmitVideo": Qwen35SmitVideo,
    "QwenVLSmitPromptPreset": QwenVLSmitPromptPreset,
    "QwenVLSmitUnload": QwenVLSmitUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVLSmitImage": "QwenVL-Smit 图片理解",
    "QwenVLSmitVideo": "QwenVL-Smit 视频理解",
    "Qwen35SmitChat": "Qwen3.5-Smit 文本对话",
    "Qwen35SmitImage": "Qwen3.5-Smit 图片理解",
    "Qwen35SmitVideo": "Qwen3.5-Smit 视频理解",
    "QwenVLSmitPromptPreset": "QwenVL-Smit 提示词预设",
    "QwenVLSmitUnload": "QwenVL-Smit 清理缓存",
}
