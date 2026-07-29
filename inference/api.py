from __future__ import annotations

from importlib import import_module
from typing import Any


def _generation() -> Any:
    return import_module("inference.batch")


def _providers() -> Any:
    return import_module("inference.providers")


def __getattr__(name: str) -> Any:
    if name in {
        "DEFAULT_SYSTEM_PROMPT",
        "DEFAULT_USER_TEMPLATE",
        "CustomInferenceBundle",
        "CustomInferenceProvider",
        "Qwen3VLInferenceBundle",
        "build_user_text",
        "build_qwen3_vl_messages",
        "call_custom_inference",
        "call_html_generation",
        "call_qwen3_vl_inference",
        "extract_html_document",
        "load_jsonl",
        "load_custom_model",
        "load_qwen3_vl_model",
        "main",
        "process_one",
    }:
        if name in {
            "CustomInferenceBundle",
            "CustomInferenceProvider",
            "Qwen3VLInferenceBundle",
            "build_qwen3_vl_messages",
            "call_custom_inference",
            "call_qwen3_vl_inference",
            "load_custom_model",
            "load_qwen3_vl_model",
        }:
            return getattr(_providers(), name)
        return getattr(_generation(), name)
    raise AttributeError(name)


def main() -> None:
    _generation().main()


def load_jsonl(*args: Any, **kwargs: Any) -> Any:
    return _generation().load_jsonl(*args, **kwargs)


def extract_html_document(*args: Any, **kwargs: Any) -> Any:
    return _generation().extract_html_document(*args, **kwargs)


def call_html_generation(*args: Any, **kwargs: Any) -> Any:
    return _providers().call_html_generation(*args, **kwargs)


def load_custom_model(*args: Any, **kwargs: Any) -> Any:
    return _providers().load_custom_model(*args, **kwargs)


def call_custom_inference(*args: Any, **kwargs: Any) -> Any:
    return _providers().call_custom_inference(*args, **kwargs)


def load_qwen3_vl_model(*args: Any, **kwargs: Any) -> Any:
    return _providers().load_qwen3_vl_model(*args, **kwargs)


def build_qwen3_vl_messages(*args: Any, **kwargs: Any) -> Any:
    return _providers().build_qwen3_vl_messages(*args, **kwargs)


def call_qwen3_vl_inference(*args: Any, **kwargs: Any) -> Any:
    return _providers().call_qwen3_vl_inference(*args, **kwargs)


def build_user_text(*args: Any, **kwargs: Any) -> Any:
    return _generation().build_user_text(*args, **kwargs)


def process_one(*args: Any, **kwargs: Any) -> Any:
    return _generation().process_one(*args, **kwargs)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_USER_TEMPLATE",
    "CustomInferenceBundle",
    "CustomInferenceProvider",
    "Qwen3VLInferenceBundle",
    "build_user_text",
    "build_qwen3_vl_messages",
    "call_custom_inference",
    "call_html_generation",
    "call_qwen3_vl_inference",
    "extract_html_document",
    "load_jsonl",
    "load_custom_model",
    "load_qwen3_vl_model",
    "main",
    "process_one",
]