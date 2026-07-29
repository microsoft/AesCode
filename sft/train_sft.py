from __future__ import annotations

import argparse
import glob
import inspect
import logging
import os
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, set_seed
from trl import SFTConfig, SFTTrainer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sft.dataset_loader import PackingCollator, PackingDataset, SFTDataCollator, SFTDataset

logger = logging.getLogger(__name__)


class SaveEveryNEpochsCallback(transformers.TrainerCallback):
    """Only save checkpoints at multiples of N epochs.

    Instead of saving every epoch then deleting unwanted ones, we simply
    suppress the save by setting control.should_save = False on non-target
    epochs.  Zero IO waste, zero race conditions.
    """

    def __init__(self, n: int):
        self.n = n

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.n <= 0 or state.epoch is None:
            return
        epoch = int(round(state.epoch))
        if epoch % self.n != 0:
            control.should_save = False


def _find_latest_checkpoint(phase_dir: str) -> str | None:
    """Return path to the latest checkpoint-* dir, or None."""
    ckpts = glob.glob(os.path.join(phase_dir, "checkpoint-*"))
    if not ckpts:
        return None
    # Sort by step number (checkpoint-100, checkpoint-200, ...)
    def _step(p):
        try:
            return int(os.path.basename(p).rsplit("-", 1)[-1])
        except ValueError:
            return -1
    latest = max(ckpts, key=_step)
    return latest if _step(latest) > 0 else None



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UI2Code Stage-1 SFT")
    parser.add_argument("--data_path", type=str, default=None,
                        help="JSONL file or directory (legacy mode)")
    parser.add_argument("--spec_jsonl", type=str, default=None,
                        help="Spec JSONL (pipeline-dirs mode)")
    parser.add_argument("--spec_jsonl_generate", type=str, default=None,
                        help="Spec JSONL for generate samples (dual-jsonl mode)")
    parser.add_argument("--spec_jsonl_refine", type=str, default=None,
                        help="Spec JSONL for refine samples (dual-jsonl mode)")
    parser.add_argument("--refine_dir", type=str, default=None,
                        help="Directory with refined/clean HTML files")
    parser.add_argument("--draft_dir", type=str, default=None,
                        help="Directory with draft/buggy HTML files")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Directory with reference images")
    parser.add_argument("--cot_dir", type=str, default=None,
                        help="(Legacy) Single COT directory with generate/ and repair/ subdirs")
    parser.add_argument("--cot_generate_dir", type=str, default=None,
                        help="Directory with flat COT JSON files for generate task")
    parser.add_argument("--cot_repair_dir", type=str, default=None,
                        help="Directory with flat COT JSON files for repair task")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--phase1_max_length", type=int, default=0)
    parser.add_argument("--phase2_max_length", type=int, default=0)
    parser.add_argument("--phase1_task", type=str, default="generate",
                        choices=["generate", "repair"],
                        help="Phase 1 single-task mode: 'generate' or 'repair'.")
    parser.add_argument("--phase2_repair_max_buggy_chars", type=int, default=30000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--phase1_epochs", type=int, default=1)
    parser.add_argument("--phase2_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--image_min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--image_max_pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="If > 0, override num_train_epochs and stop after this many steps.")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=0)
    parser.add_argument("--save_every_n_epochs", type=int, default=0,
                        help="If > 0, only keep checkpoints at epoch multiples of N (delete others after save).")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--packing", action="store_true", default=False,
                        help="Enable sequence packing (requires flash_attention_2)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.save_steps <= 0:
        raise ValueError("--save_steps must be a positive integer.")
    if args.phase1_epochs < 0 or args.phase2_epochs < 0:
        raise ValueError("--phase1_epochs/--phase2_epochs must be >= 0.")
    if args.phase1_max_length < 0 or args.phase2_max_length < 0:
        raise ValueError("--phase1_max_length/--phase2_max_length must be >= 0.")
    if args.phase2_repair_max_buggy_chars < 0:
        raise ValueError("--phase2_repair_max_buggy_chars must be >= 0.")

    set_seed(args.seed)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    transformers.utils.logging.set_verbosity_info()

    logger.info("Loading model: %s", args.model_name_or_path)
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    # trl 0.12.x SFTTrainer expects processing_class.model_max_length;
    # Qwen3VLProcessor doesn't expose it, so proxy from the tokenizer.
    if not hasattr(processor, "model_max_length"):
        processor.model_max_length = getattr(
            processor.tokenizer, "model_max_length", 32768
        )
    if not hasattr(processor, "padding_side"):
        processor.padding_side = getattr(
            processor.tokenizer, "padding_side", "right"
        )

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": None,
        "low_cpu_mem_usage": True,
    }
    if args.packing:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Packing enabled: using flash_attention_2")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # ── Determine data loading mode ──────────────────────────────────────
    dual_jsonl_mode = args.spec_jsonl_generate is not None or args.spec_jsonl_refine is not None
    if dual_jsonl_mode:
        # Dual-jsonl mode: separate generate/refine spec files
        load_kw = {
            "spec_jsonl_generate": args.spec_jsonl_generate,
            "spec_jsonl_refine": args.spec_jsonl_refine,
            "refine_dir": args.refine_dir,
            "draft_dir": args.draft_dir,
            "image_dir": args.image_dir,
            "cot_dir": args.cot_dir,
            "cot_generate_dir": args.cot_generate_dir,
            "cot_repair_dir": args.cot_repair_dir,
        }
        logger.info("Using dual-jsonl mode: gen=%s ref=%s refine=%s draft=%s image=%s",
                     args.spec_jsonl_generate, args.spec_jsonl_refine,
                     args.refine_dir, args.draft_dir, args.image_dir)
    elif args.spec_jsonl:
        # Pipeline-dirs mode: lazy loading from spec + HTML directories
        if not os.path.isfile(args.spec_jsonl):
            raise ValueError(f"--spec_jsonl not found: {args.spec_jsonl}")
        load_kw = {
            "spec_jsonl": args.spec_jsonl,
            "refine_dir": args.refine_dir,
            "draft_dir": args.draft_dir,
            "image_dir": args.image_dir,
            "cot_dir": args.cot_dir,
            "cot_generate_dir": args.cot_generate_dir,
            "cot_repair_dir": args.cot_repair_dir,
        }
        logger.info("Using pipeline-dirs mode: spec=%s refine=%s draft=%s image=%s cot_gen=%s cot_rep=%s",
                     args.spec_jsonl, args.refine_dir, args.draft_dir, args.image_dir,
                     args.cot_generate_dir, args.cot_repair_dir)
    elif args.data_path:
        if os.path.isdir(args.data_path):
            load_kw = {"dir_path": args.data_path}
        elif os.path.isfile(args.data_path) and args.data_path.endswith(".jsonl"):
            load_kw = {"jsonl_path": args.data_path}
        else:
            raise ValueError(
                f"--data_path must be a directory or a .jsonl file, got: {args.data_path}"
            )
    else:
        raise ValueError("Provide --spec_jsonl_generate/--spec_jsonl_refine (dual mode), --spec_jsonl (pipeline mode), or --data_path (legacy mode).")

    phase1_max_length = args.phase1_max_length if args.phase1_max_length > 0 else args.max_length
    phase2_max_length = args.phase2_max_length if args.phase2_max_length > 0 else args.max_length
    ds_common = {
        "processor": processor,
        "image_min_pixels": args.image_min_pixels,
        "image_max_pixels": args.image_max_pixels,
    }
    data_collator = SFTDataCollator(pad_token_id=processor.tokenizer.pad_token_id)

    # Packing config
    packing_kwargs = {}
    if args.packing:
        packing_kwargs = {
            "image_token_id": model.config.image_token_id,
            "vision_start_token_id": model.config.vision_start_token_id,
            "spatial_merge_size": model.config.vision_config.spatial_merge_size,
            "pad_token_id": processor.tokenizer.pad_token_id,
        }
        data_collator = PackingCollator()

    def make_sft_config(output_dir: str, num_epochs: int) -> SFTConfig:
        cfg_kwargs = {
            "output_dir": output_dir,
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "lr_scheduler_type": args.lr_scheduler_type,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "bf16": True,
            "fp16": False,
            "logging_steps": args.logging_steps,
            "save_strategy": "epoch",
            "save_total_limit": None if args.save_every_n_epochs > 0 else args.save_total_limit,
            "max_steps": args.max_steps,
            "dataloader_num_workers": args.dataloader_num_workers,
            "dataloader_pin_memory": True,
            "dataloader_drop_last": True,
            "report_to": args.report_to,
            "seed": args.seed,
            "remove_unused_columns": False,
            "deepspeed": args.deepspeed,
            "ddp_timeout": 7200,
        }
        cfg_params = inspect.signature(SFTConfig.__init__).parameters
        if "dataset_kwargs" in cfg_params:
            cfg_kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}
        if "max_seq_length" in cfg_params:
            cfg_kwargs["max_seq_length"] = None
        return SFTConfig(**cfg_kwargs)

    def make_sft_trainer(dataset: SFTDataset, cfg: SFTConfig) -> SFTTrainer:
        trainer_kwargs = {
            "model": model,
            "args": cfg,
            "train_dataset": dataset,
            "data_collator": data_collator,
            "callbacks": [SaveEveryNEpochsCallback(args.save_every_n_epochs)] if args.save_every_n_epochs > 0 else [],
        }
        trainer_params = inspect.signature(SFTTrainer.__init__).parameters
        if "processing_class" in trainer_params:
            trainer_kwargs["processing_class"] = processor
        elif "tokenizer" in trainer_params:
            trainer_kwargs["tokenizer"] = processor.tokenizer
        return SFTTrainer(**trainer_kwargs)

    if args.phase1_epochs > 0:
        p1_task = args.phase1_task
        p1_subdir = f"phase1-{p1_task}"
        logger.info("=== Phase 1: %s-only (%d epoch(s)) ===", p1_task, args.phase1_epochs)
        phase1_dataset = SFTDataset(
            **ds_common,
            **load_kw,
            max_length=phase1_max_length,
            generate_ratio=1.0 if p1_task == "generate" else 0.0,
            repair_max_buggy_chars=args.phase2_repair_max_buggy_chars,
        )
        if args.packing:
            phase1_dataset = PackingDataset(phase1_dataset, phase1_max_length, **packing_kwargs)
        phase1_task_dist = {}
        _p1_base = phase1_dataset.base if args.packing else phase1_dataset
        for sample in _p1_base.samples:
            phase1_task_dist[sample["task"]] = phase1_task_dist.get(sample["task"], 0) + 1
        expected_task = p1_task
        unexpected_task = "repair" if p1_task == "generate" else "generate"
        assert phase1_task_dist.get(unexpected_task, 0) == 0, (
            f"Phase-1 ({p1_task}-only) contains {unexpected_task} samples: {phase1_task_dist}"
        )

        p1_output_dir = os.path.join(args.output_dir, p1_subdir)
        phase1_trainer = make_sft_trainer(
            phase1_dataset,
            make_sft_config(
                output_dir=p1_output_dir,
                num_epochs=args.phase1_epochs,
            ),
        )
        p1_resume = _find_latest_checkpoint(p1_output_dir)
        if p1_resume:
            logger.info("Resuming Phase 1 from %s", p1_resume)
        phase1_result = phase1_trainer.train(resume_from_checkpoint=p1_resume)
        phase1_trainer.log_metrics("train_phase1", phase1_result.metrics)
        phase1_trainer.save_metrics("train_phase1", phase1_result.metrics)

        phase1_final_ckpt = os.path.join(p1_output_dir, "final")
        logger.info("GPU memory before Phase-1 save:\n%s", torch.cuda.memory_summary())
        logger.info("Saving Phase-1 final weights to %s", phase1_final_ckpt)
        phase1_trainer.save_model(phase1_final_ckpt)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        del phase1_trainer
        torch.cuda.empty_cache()

        logger.info("Reloading model from %s for Phase-2 ...", phase1_final_ckpt)
        reload_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "device_map": None,
        }
        if args.packing:
            reload_kwargs["attn_implementation"] = "flash_attention_2"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            phase1_final_ckpt,
            **reload_kwargs,
        )
        model.config.use_cache = False
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        logger.info("=== Skipping Phase 1 (phase1_epochs=0) ===")
        # Try to load Phase 1 checkpoint from a previous run (separate-process mode)
        p1_task = args.phase1_task
        p1_subdir = f"phase1-{p1_task}"
        phase1_final_ckpt = os.path.join(args.output_dir, p1_subdir, "final")
        if os.path.isdir(phase1_final_ckpt):
            logger.info("Found Phase-1 checkpoint at %s, reloading model ...", phase1_final_ckpt)
            reload_kwargs = {
                "torch_dtype": torch.bfloat16,
                "trust_remote_code": True,
                "device_map": None,
            }
            if args.packing:
                reload_kwargs["attn_implementation"] = "flash_attention_2"
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                phase1_final_ckpt,
                **reload_kwargs,
            )
            model.config.use_cache = False
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            logger.info("No Phase-1 checkpoint found at %s, using base model for Phase 2.", phase1_final_ckpt)

    if args.phase2_epochs > 0:
        logger.info("=== Phase 2: repair+generate (%d epoch(s)) ===", args.phase2_epochs)
        phase2_dataset = SFTDataset(
            **ds_common,
            **load_kw,
            max_length=phase2_max_length,
            duplicate_tasks=True,
            repair_max_buggy_chars=args.phase2_repair_max_buggy_chars,
        )
        if args.packing:
            phase2_dataset = PackingDataset(phase2_dataset, phase2_max_length, **packing_kwargs)
        phase2_task_dist = {}
        _p2_base = phase2_dataset.base if args.packing else phase2_dataset
        for sample in _p2_base.samples:
            phase2_task_dist[sample["task"]] = phase2_task_dist.get(sample["task"], 0) + 1
        assert phase2_task_dist.get("repair", 0) > 0 and phase2_task_dist.get("generate", 0) > 0, (
            f"Phase-2 task split unexpected: {phase2_task_dist}"
        )

        p2_output_dir = os.path.join(args.output_dir, "phase2")
        phase2_trainer = make_sft_trainer(
            phase2_dataset,
            make_sft_config(
                output_dir=p2_output_dir,
                num_epochs=args.phase2_epochs,
            ),
        )
        p2_resume = _find_latest_checkpoint(p2_output_dir)
        if p2_resume:
            logger.info("Resuming Phase 2 from %s", p2_resume)
        phase2_result = phase2_trainer.train(resume_from_checkpoint=p2_resume)
        phase2_trainer.log_metrics("train_phase2", phase2_result.metrics)
        phase2_trainer.save_metrics("train_phase2", phase2_result.metrics)

        final_ckpt_dir = Path(args.output_dir) / "phase2" / "final"
        logger.info("GPU memory before Phase-2 save:\n%s", torch.cuda.memory_summary())
        phase2_trainer.save_model(str(final_ckpt_dir))
        phase2_trainer.save_state()
        logger.info("Training complete. Final checkpoint: %s", final_ckpt_dir)
    else:
        logger.info("=== Skipping Phase 2 (phase2_epochs=0) ===")
        logger.info("Training complete (Phase 1 only).")


if __name__ == "__main__":
    main()