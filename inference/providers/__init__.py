from inference.providers.azure import call_html_generation, extract_html_document
from inference.providers.custom import (
    CustomInferenceBundle,
    CustomInferenceProvider,
    call_custom_inference,
    load_custom_model,
)
from inference.providers.qwen import (
    Qwen3VLInferenceBundle,
    build_qwen3_vl_messages,
    call_qwen3_vl_inference,
    load_qwen3_vl_model,
)
from inference.providers.silicon import call_silicon_generation, load_silicon_api_key

__all__ = [
    "call_html_generation",
    "extract_html_document",
    "CustomInferenceBundle",
    "CustomInferenceProvider",
    "call_custom_inference",
    "load_custom_model",
    "Qwen3VLInferenceBundle",
    "build_qwen3_vl_messages",
    "call_qwen3_vl_inference",
    "load_qwen3_vl_model",
    "call_silicon_generation",
    "load_silicon_api_key",
]