#!/usr/bin/env python3
"""Checkpoint cleanup for VERL GDPO training.

Three-tier retention policy:
  - FULL (model + optim + extra_state + data.pt = ~52 GB / step):
    only the latest --keep-full-last steps (default: 2). Resume-ready.
  - INFERENCE-ONLY (model + extra_state + data.pt, optim stripped = ~20 GB / step):
    older steps where step % --keep-every == 0 (default: 5), AND every
    --milestone-every step (default: 50). Not resume-ready — you'd have to
    rewind to a FULL one — but suitable for downstream eval / SFT export.
  - DELETED: everything else.

Usage:
    # default policy
    python rl_verl/cleanup_checkpoints.py /path/to/output_dir
    # custom
    python rl_verl/cleanup_checkpoints.py /path/to/output_dir \
        --keep-full-last 2 --keep-every 5 --milestone-every 50
    # preview
    python rl_verl/cleanup_checkpoints.py /path/to/output_dir --dry-run
"""

import argparse
import re
import shutil
from pathlib import Path


# Files inside global_step_N/actor/ that are ONLY needed for resume.
# Stripping them leaves an inference-ready checkpoint (model weights + tokenizer
# metadata + extra_state). data.pt at the step root is dataloader state — also
# resume-only, but tiny (~8 KB) so we keep it for simplicity.
_OPTIM_PATTERN = re.compile(r"^optim_world_size_\d+_rank_\d+\.pt$")


def find_checkpoint_steps(ckpt_dir: Path) -> list[int]:
    """Find all global_step_N directories and return sorted step numbers."""
    steps = []
    for d in ckpt_dir.iterdir():
        if d.is_dir():
            m = re.match(r"global_step_(\d+)$", d.name)
            if m:
                steps.append(int(m.group(1)))
    return sorted(steps)


def classify_steps(
    steps: list[int],
    keep_full_last: int,
    keep_every: int,
    milestone_every: int,
) -> tuple[set[int], set[int], list[int]]:
    """Decide which steps go to which tier.

    Returns: (full_steps, strip_steps, delete_steps).
    """
    if not steps:
        return set(), set(), []

    full = set(steps[-keep_full_last:]) if keep_full_last > 0 else set()

    strip = set()
    for s in steps:
        if s in full:
            continue
        if s == 0:
            strip.add(s)  # always keep step 0 as inference baseline
            continue
        if keep_every > 0 and s % keep_every == 0:
            strip.add(s)
        elif milestone_every > 0 and s % milestone_every == 0:
            strip.add(s)

    delete = [s for s in steps if s not in full and s not in strip]
    return full, strip, delete


def strip_optim_from_step(step_dir: Path) -> int:
    """Delete optimizer shards inside global_step_N/actor/. Returns bytes freed."""
    actor = step_dir / "actor"
    if not actor.is_dir():
        return 0
    freed = 0
    for f in actor.iterdir():
        if f.is_file() and _OPTIM_PATTERN.match(f.name):
            try:
                freed += f.stat().st_size
            except OSError:
                pass
            f.unlink(missing_ok=True)
    return freed


def has_optim(step_dir: Path) -> bool:
    actor = step_dir / "actor"
    if not actor.is_dir():
        return False
    return any(_OPTIM_PATTERN.match(f.name) for f in actor.iterdir() if f.is_file())


def cleanup(
    ckpt_dir: str,
    keep_full_last: int = 2,
    keep_every: int = 5,
    milestone_every: int = 50,
    dry_run: bool = False,
):
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        print(f"Checkpoint dir does not exist: {ckpt_dir}")
        return

    steps = find_checkpoint_steps(ckpt_path)
    if not steps:
        print("No checkpoints found.")
        return

    full, strip, delete = classify_steps(
        steps, keep_full_last, keep_every, milestone_every
    )

    print(f"Found {len(steps)} checkpoints: {steps[0]}..{steps[-1]}")
    print(f"  FULL  (model+optim, resume-ready)  [{len(full)}]: {sorted(full)}")
    print(f"  STRIP (model only, inference-only) [{len(strip)}]: {sorted(strip)}")
    print(f"  DELETE                             [{len(delete)}]: {delete}")

    if dry_run:
        print("(dry run — no files modified)")
        return

    total_freed_bytes = 0

    # 1. Strip optimizer shards from inference-only steps that still have them.
    for s in sorted(strip):
        d = ckpt_path / f"global_step_{s}"
        if has_optim(d):
            freed = strip_optim_from_step(d)
            total_freed_bytes += freed
            print(f"  Stripped optim from step {s} (freed {freed / 1e9:.1f} GB)")

    # 2. Fully remove deleted steps.
    for s in delete:
        d = ckpt_path / f"global_step_{s}"
        try:
            size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
        except OSError:
            size = 0
        total_freed_bytes += size
        print(f"  Removing {d} ({size / 1e9:.1f} GB)")
        shutil.rmtree(d, ignore_errors=True)

    print(
        f"Cleanup done. Freed {total_freed_bytes / 1e9:.1f} GB "
        f"(stripped {len(strip)}, deleted {len(delete)})."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup VERL GDPO checkpoints")
    parser.add_argument("ckpt_dir", help="Path to OUTPUT_DIR containing global_step_N/")
    parser.add_argument(
        "--keep-full-last",
        type=int,
        default=2,
        help="Number of most-recent steps to keep FULL (resume-ready). Default: 2",
    )
    parser.add_argument(
        "--keep-every",
        type=int,
        default=5,
        help="Beyond the latest FULL steps, keep inference-only at every Nth step. "
        "Default: 5 (i.e. steps 5,10,15,...). Set 0 to disable.",
    )
    parser.add_argument(
        "--milestone-every",
        type=int,
        default=50,
        help="Additionally keep inference-only at every Mth step as a milestone. "
        "Default: 50. Set 0 to disable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show classification without modifying any files",
    )
    # Back-compat: --keep-last is the old API; treat it as --keep-full-last.
    parser.add_argument(
        "--keep-last",
        type=int,
        default=None,
        help="DEPRECATED alias for --keep-full-last (kept for back-compat).",
    )
    args = parser.parse_args()
    keep_full = args.keep_full_last if args.keep_last is None else args.keep_last
    cleanup(
        args.ckpt_dir,
        keep_full_last=keep_full,
        keep_every=args.keep_every,
        milestone_every=args.milestone_every,
        dry_run=args.dry_run,
    )
