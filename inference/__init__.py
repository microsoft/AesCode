from inference.generate import extract_html_document, generate, to_data_url
from inference.prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_TEMPLATE, build_user_text

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_USER_TEMPLATE",
    "build_user_text",
    "extract_html_document",
    "generate",
    "to_data_url",
]
