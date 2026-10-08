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


def _find_latest_checkpoint(run_dir: str) -> str | None:
    """Return path to the latest checkpoint-* dir, or None."""
    ckpts = glob.glob(os.path.join(run_dir, "checkpoint-*"))
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
    parser = argparse.ArgumentParser(description="AesCode supervised fine-tuning")
    parser.add_argument("--data_path", type=str, default=None,
                        help="JSONL file or directory (legacy mode)")
    parser.add_argument("--spec_jsonl", type=str, default=None,
                        help="Spec JSONL (pipeline-dirs mode)")
    parser.add_argument("--clean_html_dir", type=str, default=None,
                        help="Directory with the ground-truth (clean) HTML files")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Directory with reference images")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
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
    if args.num_train_epochs <= 0:
        raise ValueError("--num_train_epochs must be a positive integer.")
    if args.max_length <= 0:
        raise ValueError("--max_length must be a positive integer.")

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
    if args.spec_jsonl:
        # Pipeline-dirs mode: lazy loading from spec + HTML directories
        if not os.path.isfile(args.spec_jsonl):
            raise ValueError(f"--spec_jsonl not found: {args.spec_jsonl}")
        load_kw = {
            "spec_jsonl": args.spec_jsonl,
            "clean_html_dir": args.clean_html_dir,
            "image_dir": args.image_dir,
        }
        logger.info("Using pipeline-dirs mode: spec=%s clean_html=%s image=%s",
                     args.spec_jsonl, args.clean_html_dir, args.image_dir)
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
        raise ValueError("Provide --spec_jsonl (pipeline mode) or --data_path (legacy mode).")

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

    # NOTE: the "phase1-generate" sub-directory name is kept because the RL
    # scripts under rl_verl/ default to this path for the cold-start checkpoint.
    train_output_dir = os.path.join(args.output_dir, "phase1-generate")
    logger.info("=== SFT: generate-only (%d epoch(s)) ===", args.num_train_epochs)
    train_dataset = SFTDataset(
        **ds_common,
        **load_kw,
        max_length=args.max_length,
    )
    if args.packing:
        train_dataset = PackingDataset(train_dataset, args.max_length, **packing_kwargs)

    trainer = make_sft_trainer(
        train_dataset,
        make_sft_config(
            output_dir=train_output_dir,
            num_epochs=args.num_train_epochs,
        ),
    )
    resume = _find_latest_checkpoint(train_output_dir)
    if resume:
        logger.info("Resuming from %s", resume)
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)

    final_ckpt_dir = Path(train_output_dir) / "final"
    logger.info("GPU memory before save:\n%s", torch.cuda.memory_summary())
    logger.info("Saving final weights to %s", final_ckpt_dir)
    trainer.save_model(str(final_ckpt_dir))
    trainer.save_state()
    logger.info("Training complete. Final checkpoint: %s", final_ckpt_dir)


if __name__ == "__main__":
    main()