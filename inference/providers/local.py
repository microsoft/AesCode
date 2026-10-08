from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


ImageInput = Union[str, Path, "PILImage"]


@dataclass(frozen=True)
class Qwen3VLInferenceBundle:
    model: Any
    processor: Any
    model_name: str


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as err:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "torch is required for Qwen3-VL inference. Install torch before using --provider local."
        ) from err
    return torch


def _resolve_torch_dtype(torch_module: Any, dtype: str) -> Any:
    normalized = (dtype or "auto").strip().lower()
    if normalized == "auto":
        return "auto"
    mapping = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    resolved = mapping.get(normalized)
    if resolved is None:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return resolved


def build_qwen3_vl_messages(
    *,
    image: ImageInput,
    prompt: str,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    image_value = str(image) if isinstance(image, Path) else image
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_value},
                {"type": "text", "text": prompt},
            ],
        }
    )
    return messages


def load_qwen3_vl_model(
    *,
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    dtype: str = "auto",
    device_map: str = "auto",
    attn_implementation: Optional[str] = None,
) -> Qwen3VLInferenceBundle:
    torch = _import_torch()
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as err:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "transformers is required for Qwen3-VL inference. Install transformers before using --provider local."
        ) from err

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": _resolve_torch_dtype(torch, dtype),
        "device_map": device_map,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_name)
    return Qwen3VLInferenceBundle(model=model, processor=processor, model_name=model_name)


def call_qwen3_vl_inference(
    bundle: Qwen3VLInferenceBundle,
    messages: Sequence[Dict[str, Any]],
    *,
    max_new_tokens: int = 128,
) -> List[str]:
    inputs = bundle.processor.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(bundle.model.device)

    generated_ids = bundle.model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = bundle.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [text.strip() for text in output_text]