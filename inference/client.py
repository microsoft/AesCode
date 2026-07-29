"""Backward-compatible Azure OpenAI client shim."""

from inference.providers.azure import call_html_generation, extract_html_document

__all__ = ["call_html_generation", "extract_html_document"]
