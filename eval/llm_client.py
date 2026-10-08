"""OpenAI-compatible client pool for the judge and the checklist author.

Everything in this benchmark that talks to a large model talks to it through the
chat-completions schema, so one client works for the OpenAI API, for any hosted
gateway that speaks the same schema, and for a local vLLM or SGLang server.

    export OPENAI_API_KEY=sk-...
    export OPENAI_BASE_URL=https://api.openai.com/v1

`OPENAI_BASE_URL` may be a comma-separated list. Calls are handed out round-robin
over the list, which is how a judging pass is spread across several deployments
of the same model without any of them hot-spotting. The model name is the same on
every base URL in the list; a pool of endpoints serving *different* models would
silently mix judges across the table, so it is not supported.

The judge model is chosen by `--judge-model` on the command line, falling back to
`$JUDGE_MODEL`. Pin it for a whole table: a provider alias that is repointed
mid-run changes the judge between rows, and the rows are then not comparable.
"""

from __future__ import annotations

import itertools
import os
import threading
import time
from typing import Any, List, Sequence

DEFAULT_MODEL = "gpt-5.5"
# Errors that no amount of waiting fixes: a bad request, a bad key, a model the
# endpoint does not serve. Retrying these just turns a clear failure into a slow
# one, and a judge that gives up records ok=False rather than scoring the page 0.
FATAL_HTTP = ("400 ", "401 ", "403 ", "404 ", "BadRequest")


def base_urls() -> List[str]:
    raw = os.environ.get("OPENAI_BASE_URL", "")
    return [u.strip() for u in raw.split(",") if u.strip()]


def judge_model(cli_value: str = "") -> str:
    return cli_value or os.environ.get("JUDGE_MODEL") or DEFAULT_MODEL


class ClientPool:
    """Round-robin pool of OpenAI-compatible clients, one per base URL."""

    def __init__(self, urls: Sequence[str] | None = None, timeout: float = 300.0):
        from openai import OpenAI

        urls = list(urls) if urls else base_urls()
        api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
        if urls:
            self.clients = [OpenAI(base_url=u, api_key=api_key, timeout=timeout)
                            for u in urls]
            self.labels = list(urls)
        else:
            # No base URL: the SDK's own default, i.e. the OpenAI API.
            self.clients = [OpenAI(api_key=api_key, timeout=timeout)]
            self.labels = ["default"]
        self._cycle = itertools.cycle(zip(self.clients, self.labels))
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.clients)

    def next(self):
        with self._lock:
            return next(self._cycle)


def chat(
    pool: ClientPool,
    model: str,
    messages: List[Any],
    *,
    max_tokens: int = 8000,
    timeout: float = 240.0,
    reasoning_effort: str | None = None,
    tries: int = 4,
) -> str:
    """One chat completion, retried on transient failures. Returns the text."""
    kwargs: dict = {}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    delay, last = 10.0, None
    for attempt in range(tries):
        client, _label = pool.next()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
                timeout=timeout,
                **kwargs,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            last = exc
            if any(code in str(exc) for code in FATAL_HTTP):
                break
            if attempt < tries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 120.0)
    raise RuntimeError(f"{type(last).__name__}: {str(last)[:200]}")


def wants_reasoning_effort(model: str) -> bool:
    """Reasoning models take an effort knob; sending it elsewhere is a 400."""
    return model.startswith(("gpt-5", "o1", "o3", "o4"))
