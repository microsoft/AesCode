from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from inference.providers.qwen import ImageInput


@runtime_checkable
class CustomInferenceProvider(Protocol):
    def load(
        self,
        model_path: Path,
        *,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        config: Optional[Dict[str, Any]] = None,
    ) -> Any: ...

    def generate(
        self,
        runtime: Any,
        *,
        image: ImageInput,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 128,
    ) -> str: ...


@dataclass(frozen=True)
class CustomInferenceBundle:
    provider: CustomInferenceProvider
    runtime: Any
    provider_ref: str
    model_path: Path


def _load_provider_object(provider_ref: str) -> CustomInferenceProvider:
    module_name, separator, attr_name = provider_ref.partition(":")
    if not separator or not module_name or not attr_name:
        raise ValueError(
            "custom provider ref must use 'module.path:attribute' format, for example 'my_pkg.my_provider:provider'."
        )

    module = importlib.import_module(module_name)
    provider = getattr(module, attr_name)
    if isinstance(provider, type):
        provider = provider()
    if not isinstance(provider, CustomInferenceProvider):
        raise TypeError(
            f"Custom provider '{provider_ref}' must implement load() and generate() methods."
        )
    return provider


def _load_custom_config(config_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if config_path is None:
        return None
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_custom_model(
    *,
    provider_ref: str,
    model_path: Path,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    config_path: Optional[Path] = None,
) -> CustomInferenceBundle:
    provider = _load_provider_object(provider_ref)
    runtime = provider.load(
        model_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        config=_load_custom_config(config_path),
    )
    return CustomInferenceBundle(
        provider=provider,
        runtime=runtime,
        provider_ref=provider_ref,
        model_path=model_path,
    )


def call_custom_inference(
    bundle: CustomInferenceBundle,
    *,
    image: ImageInput,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 128,
) -> str:
    return bundle.provider.generate(
        bundle.runtime,
        image=image,
        prompt=prompt,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
    ).strip()