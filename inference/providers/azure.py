"""Azure OpenAI HTML generation client."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from openai import AzureOpenAI


def _extract_text_from_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"text", "output_text"}:
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def extract_html_document(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:html)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    doctype_index = stripped.lower().find("<!doctype html>")
    if doctype_index >= 0:
        return stripped[doctype_index:].strip()
    html_index = stripped.lower().find("<html")
    if html_index >= 0:
        return "<!DOCTYPE html>\n" + stripped[html_index:].strip()
    raise ValueError("Model output did not contain an HTML document.")


def call_html_generation(
    client: "AzureOpenAI",
    deployment: str,
    system_prompt: str,
    user_text: str,
    max_tokens: int,
    image_data_url: Optional[str] = None,
    extra_image_data_urls: Optional[List[str]] = None,
) -> str:
    """Call Azure OpenAI for HTML generation.

    Args:
        image_data_url: primary reference image (image #1).
        extra_image_data_urls: additional images to include AFTER the primary
            image and BEFORE the user text. Used by the refine pipeline to
            attach a low-res screenshot of the previous implementation as
            image #2.
    """
    errors: List[str] = []

    # Normalize image list (preserves [primary, extra1, extra2, ...] order).
    all_image_urls: List[str] = []
    if image_data_url:
        all_image_urls.append(image_data_url)
    if extra_image_data_urls:
        all_image_urls.extend([u for u in extra_image_data_urls if u])

    try:
        responses_api = getattr(client, "responses", None)
        if responses_api is not None:
            user_content: List[Any] = [{"type": "input_text", "text": user_text}]
            # Insert images in order, before the text.
            for url in reversed(all_image_urls):
                user_content.insert(0, {"type": "input_image", "image_url": url})
            response = responses_api.create(
                model=deployment,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": user_content},
                ],
                max_output_tokens=max_tokens,
            )
            output_text = (getattr(response, "output_text", None) or "").strip()
            if output_text:
                return output_text
            errors.append("responses.create returned empty content")
    except Exception as err:  # noqa: BLE001
        errors.append(f"responses.create failed: {err}")

    if all_image_urls:
        chat_user_content: Any = [
            {"type": "image_url", "image_url": {"url": url}}
            for url in all_image_urls
        ]
        chat_user_content.append({"type": "text", "text": user_text})
    else:
        chat_user_content = user_text

    base_kwargs = {
        "model": deployment,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chat_user_content},
        ],
    }

    try:
        response = client.chat.completions.create(**base_kwargs, max_completion_tokens=max_tokens)
        output_text = _extract_text_from_message_content(response.choices[0].message.content).strip()
        if output_text:
            return output_text
        errors.append("chat(max_completion_tokens) returned empty content")
    except Exception as err:  # noqa: BLE001
        errors.append(f"chat(max_completion_tokens) failed: {err}")

    try:
        response = client.chat.completions.create(**base_kwargs, max_tokens=max_tokens)
        output_text = _extract_text_from_message_content(response.choices[0].message.content).strip()
        if output_text:
            return output_text
        errors.append("chat(max_tokens) returned empty content")
    except Exception as err:  # noqa: BLE001
        errors.append(f"chat(max_tokens) failed: {err}")

    raise RuntimeError("All HTML generation routes failed. " + " | ".join(errors))