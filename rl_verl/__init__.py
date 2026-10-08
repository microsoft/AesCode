"""VERL-based GDPO RL module (generate-only).

This module provides the RL training pipeline using the verl framework
(Ray + FSDP + vLLM).

Key components:
  - rewards: Verifier-based reward backend (_extract_html, _run_verifier_fast, ...)
  - data_prep: Convert JSONL dataset to verl parquet format
  - reward_fn: Custom compute_score for verl's reward manager
  - design_coordinator: Cross-rollout DesignValidator (Swiss tournament)
  - multi_step_agent: SingleStepGenerateAgentLoop (generate-only)

Training: each step = 1 generate rollout → reward → gradient update.
"""

from __future__ import annotations

__all__ = [
    "compute_score",
    "prepare_dataset",
]


def __getattr__(name: str):
    if name == "compute_score":
        from rl_verl.reward_fn import compute_score
        return compute_score
    if name == "prepare_dataset":
        from rl_verl.data_prep import prepare_dataset
        return prepare_dataset

    raise AttributeError(f"module 'rl_verl' has no attribute {name!r}")
