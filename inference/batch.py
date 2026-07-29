"""HTML generation batch: per-sample processing, JSONL loading, and CLI entry."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from openai import AzureOpenAI

    from inference.providers.custom import CustomInferenceBundle
    from inference.providers.qwen import Qwen3VLInferenceBundle

from inference.image_utils import load_api_key, resolve_image_path, image_to_data_url
from inference.providers import call_html_generation, extract_html_document
from inference.prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_TEMPLATE, build_user_text


DEFAULT_IMAGE_DIR = Path("data/testset")
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "inference_runs"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def parse_sample_ids(sample_ids: str) -> List[str]:
    return [item.strip() for item in sample_ids.split(",") if item.strip()]


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip())
    text = re.sub(r"-+", "-", text).strip("-._")
    return text or "default"


def _optional_slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip())
    return re.sub(r"-+", "-", text).strip("-._")


def _provider_model_label(args: argparse.Namespace) -> str:
    if args.provider == "azure":
        return _slugify(args.deployment).replace("-", "")
    if args.provider == "qwen":
        return _slugify(args.model_name)
    if args.provider == "silicon":
        return _slugify(args.silicon_model)
    custom_label = args.custom_model_path.name if args.custom_model_path else args.custom_provider_ref
    return _slugify(custom_label)


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()

    base_name = f"{_slugify(args.provider)}_{_provider_model_label(args)}"
    run_label = _optional_slugify(args.run_name or "")
    dir_name = f"{base_name}__{run_label}" if run_label else base_name
    return args.output_root.resolve() / dir_name


def _build_run_metadata(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    rows: List[Dict[str, Any]],
    system_prompt: str,
    user_template: str,
) -> Dict[str, Any]:
    return {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "provider": args.provider,
        "model_label": _provider_model_label(args),
        "input_jsonl": str(args.input_jsonl.resolve()),
        "output_dir": str(output_dir),
        "output_root": str(args.output_root.resolve()) if args.output_root else None,
        "run_name": _optional_slugify(args.run_name or ""),
        "sample_count": len(rows),
        "sample_ids": [str(row.get("id") or "") for row in rows],
        "image_dir": str(args.image_dir.resolve()) if args.image_dir else None,
        "system_prompt_source": str(args.system_prompt_file.resolve()) if args.system_prompt_file else "builtin",
        "user_template_source": str(args.user_template_file.resolve()) if args.user_template_file else "builtin",
        "max_completion_tokens": args.max_completion_tokens,
        "retries": args.retries,
        "retry_sleep": args.retry_sleep,
        "resume": bool(args.resume),
        "system_prompt_preview": system_prompt[:400],
        "user_template_preview": user_template[:400],
    }


def process_one(
    sample: Dict[str, Any],
    *,
    client: "AzureOpenAI",
    deployment: str,
    system_prompt: str,
    user_template: str,
    output_dir: Path,
    image_dir: Optional[Path] = None,
    root_dir: Optional[Path] = None,
    max_tokens: int,
    retries: int,
    retry_sleep: float,
) -> Dict[str, Any]:
    sample_id = str(sample.get("id") or "")
    if not sample_id:
        raise ValueError("Refined sample is missing id.")

    image_data_url: Optional[str] = None
    if image_dir and root_dir:
        try:
            img_path = resolve_image_path(sample, root_dir=root_dir, image_dir=image_dir)
            if img_path.exists():
                image_data_url = image_to_data_url(img_path)
        except Exception:  # noqa: BLE001
            pass  # no image available; generation continues without it

    user_text = build_user_text(sample, user_template, has_image=image_data_url is not None)
    output_path = output_dir / f"{sample_id}.html"
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            raw_html = call_html_generation(
                client=client,
                deployment=deployment,
                system_prompt=system_prompt,
                user_text=user_text,
                max_tokens=max_tokens,
                image_data_url=image_data_url,
            )
            html = extract_html_document(raw_html)
            output_path.write_text(html + "\n", encoding="utf-8")
            return {
                "sample_id": sample_id,
                "html_path": str(output_path),
                "status": "success",
            }
        except Exception as err:  # noqa: BLE001
            last_error = err
            if attempt < retries:
                time.sleep(retry_sleep * attempt)

    return {
        "sample_id": sample_id,
        "html_path": str(output_path),
        "status": "error",
        "error": str(last_error) if last_error else "unknown_error",
    }


def process_one_qwen(
    sample: Dict[str, Any],
    *,
    bundle: "Qwen3VLInferenceBundle",
    system_prompt: str,
    user_template: str,
    output_dir: Path,
    image_dir: Path,
    root_dir: Path,
    max_tokens: int,
    retries: int,
    retry_sleep: float,
) -> Dict[str, Any]:
    sample_id = str(sample.get("id") or "")
    if not sample_id:
        raise ValueError("Refined sample is missing id.")

    image_path = resolve_image_path(sample, root_dir=root_dir, image_dir=image_dir)
    user_text = build_user_text(sample, user_template, has_image=True)
    output_path = output_dir / f"{sample_id}.html"
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            from inference.providers.qwen import build_qwen3_vl_messages, call_qwen3_vl_inference

            messages = build_qwen3_vl_messages(
                image=image_path,
                prompt=user_text,
                system_prompt=system_prompt,
            )
            output_texts = call_qwen3_vl_inference(
                bundle,
                messages,
                max_new_tokens=max_tokens,
            )
            raw_html = output_texts[0] if output_texts else ""
            html = extract_html_document(raw_html)
            output_path.write_text(html + "\n", encoding="utf-8")
            return {
                "sample_id": sample_id,
                "html_path": str(output_path),
                "status": "success",
                "provider": "qwen",
            }
        except Exception as err:  # noqa: BLE001
            last_error = err
            if attempt < retries:
                time.sleep(retry_sleep * attempt)

    return {
        "sample_id": sample_id,
        "html_path": str(output_path),
        "status": "error",
        "provider": "qwen",
        "error": str(last_error) if last_error else "unknown_error",
    }


def process_one_custom(
    sample: Dict[str, Any],
    *,
    bundle: "CustomInferenceBundle",
    system_prompt: str,
    user_template: str,
    output_dir: Path,
    image_dir: Path,
    root_dir: Path,
    max_tokens: int,
    retries: int,
    retry_sleep: float,
) -> Dict[str, Any]:
    sample_id = str(sample.get("id") or "")
    if not sample_id:
        raise ValueError("Refined sample is missing id.")

    image_path = resolve_image_path(sample, root_dir=root_dir, image_dir=image_dir)
    user_text = build_user_text(sample, user_template, has_image=True)
    output_path = output_dir / f"{sample_id}.html"
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            from inference.providers.custom import call_custom_inference

            raw_html = call_custom_inference(
                bundle,
                image=image_path,
                prompt=user_text,
                system_prompt=system_prompt,
                max_new_tokens=max_tokens,
            )
            html = extract_html_document(raw_html)
            output_path.write_text(html + "\n", encoding="utf-8")
            return {
                "sample_id": sample_id,
                "html_path": str(output_path),
                "status": "success",
                "provider": "custom",
            }
        except Exception as err:  # noqa: BLE001
            last_error = err
            if attempt < retries:
                time.sleep(retry_sleep * attempt)

    return {
        "sample_id": sample_id,
        "html_path": str(output_path),
        "status": "error",
        "provider": "custom",
        "error": str(last_error) if last_error else "unknown_error",
    }


def process_one_silicon(
    sample: Dict[str, Any],
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_template: str,
    output_dir: Path,
    image_dir: Optional[Path] = None,
    root_dir: Optional[Path] = None,
    max_tokens: int,
    retries: int,
    retry_sleep: float,
) -> Dict[str, Any]:
    from inference.providers.silicon import call_silicon_generation

    sample_id = str(sample.get("id") or "")
    if not sample_id:
        raise ValueError("Sample is missing id.")

    image_data_url: Optional[str] = None
    if image_dir and root_dir:
        try:
            img_path = resolve_image_path(sample, root_dir=root_dir, image_dir=image_dir)
            if img_path.exists():
                image_data_url = image_to_data_url(img_path)
        except Exception:  # noqa: BLE001
            pass

    user_text = build_user_text(sample, user_template, has_image=image_data_url is not None)
    output_path = output_dir / f"{sample_id}.html"
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            raw_html = call_silicon_generation(
                api_key=api_key,
                base_url=base_url,
                model=model,
                system_prompt=system_prompt,
                user_text=user_text,
                max_tokens=max_tokens,
                image_data_url=image_data_url,
            )
            html = extract_html_document(raw_html)
            output_path.write_text(html + "\n", encoding="utf-8")
            return {
                "sample_id": sample_id,
                "html_path": str(output_path),
                "status": "success",
                "provider": "silicon",
            }
        except Exception as err:  # noqa: BLE001
            last_error = err
            if attempt < retries:
                time.sleep(retry_sleep * attempt)

    return {
        "sample_id": sample_id,
        "html_path": str(output_path),
        "status": "error",
        "provider": "silicon",
        "error": str(last_error) if last_error else "unknown_error",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate HTML files from refined sample JSONL.")
    parser.add_argument("--input-jsonl", type=Path, default=Path("data/spec.jsonl"),
        help="Spec JSONL to generate HTML from. Default: data/spec.jsonl")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit output directory for HTML files. Overrides the structured run-directory layout.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for structured inference runs when --output-dir is not provided.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional short suffix appended to the default output directory name when --output-dir is not provided.",
    )
    parser.add_argument("--api-key-path", type=Path, default=None)
    parser.add_argument(
        "--provider",
        type=str,
        choices=["azure", "qwen", "custom", "silicon"],
        default=os.getenv("GENERATION_PROVIDER", "azure"),
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.getenv("AZURE_OPENAI_ENDPOINT", "https://agl-dev.cognitiveservices.azure.com/"),
    )
    parser.add_argument(
        "--api-version",
        type=str,
        default=os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
    )
    parser.add_argument(
        "--deployment",
        type=str,
        default=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4"),
    )
    parser.add_argument(
        "--silicon-api-key",
        type=str,
        default=os.getenv("SILICON_API_KEY", ""),
        help="Silicon Flow API key. Falls back to SILICON_API_KEY env var.",
    )
    parser.add_argument(
        "--silicon-base-url",
        type=str,
        default=os.getenv("SILICON_BASE_URL", "https://api.siliconflow.cn/v1"),
        help="Silicon Flow base URL. Default: https://api.siliconflow.cn/v1",
    )
    parser.add_argument(
        "--silicon-model",
        type=str,
        default=os.getenv("SILICON_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
        help="Model name passed to Silicon Flow API. Default: Qwen/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Reference image directory. Default: data/testset",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct"),
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default=os.getenv("QWEN_DEVICE_MAP", "auto"),
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default=os.getenv("QWEN_TORCH_DTYPE", "auto"),
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=os.getenv("QWEN_ATTN_IMPLEMENTATION", ""),
        help="Optional attention implementation, for example flash_attention_2.",
    )
    parser.add_argument(
        "--custom-provider-ref",
        type=str,
        default=os.getenv("CUSTOM_PROVIDER_REF", ""),
        help="Python ref in module:attribute format for a custom local provider.",
    )
    parser.add_argument(
        "--custom-model-path",
        type=Path,
        default=Path(os.getenv("CUSTOM_MODEL_PATH", "")) if os.getenv("CUSTOM_MODEL_PATH") else None,
        help="Path to a locally saved custom model checkpoint or model directory.",
    )
    parser.add_argument(
        "--custom-config-json",
        type=Path,
        default=Path(os.getenv("CUSTOM_CONFIG_JSON", "")) if os.getenv("CUSTOM_CONFIG_JSON") else None,
        help="Optional JSON config passed to the custom provider load() hook.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0,
        help="Maximum number of samples to process (0 = all).")
    parser.add_argument("--random", action="store_true",
        help="Shuffle samples before applying --limit.")
    parser.add_argument("--sample-ids", type=str, default="")
    parser.add_argument("--max-completion-tokens", type=int, default=12000)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--failed-log", type=Path, default=None)
    parser.add_argument("--system-prompt-file", type=Path, default=None)
    parser.add_argument("--user-template-file", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl.resolve())
    selected_ids = set(parse_sample_ids(args.sample_ids))
    if selected_ids:
        rows = [row for row in rows if str(row.get("id") or "") in selected_ids]
    if args.random:
        import random as _random
        _random.shuffle(rows)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        rows = [row for row in rows if not (output_dir / f"{row.get('id', '')}.html").exists()]

    if not rows:
        print("No pending samples.")
        return

    system_prompt = (
        args.system_prompt_file.read_text(encoding="utf-8").strip()
        if args.system_prompt_file
        else DEFAULT_SYSTEM_PROMPT
    )
    user_template = (
        args.user_template_file.read_text(encoding="utf-8").strip()
        if args.user_template_file
        else DEFAULT_USER_TEMPLATE
    )

    run_metadata = _build_run_metadata(
        args,
        output_dir=output_dir,
        rows=rows,
        system_prompt=system_prompt,
        user_template=user_template,
    )
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"output_dir={output_dir}", flush=True)
    print(f"run_metadata={metadata_path}", flush=True)

    results: List[Dict[str, Any]] = []
    if args.provider == "azure":
        api_key = load_api_key(args.api_key_path.resolve() if args.api_key_path else None)
        from openai import AzureOpenAI  # lazy: only import when actually running

        client = AzureOpenAI(
            api_version=args.api_version,
            azure_endpoint=args.endpoint,
            api_key=api_key,
        )

        image_dir = args.image_dir.resolve() if args.image_dir else None
        root_dir = Path(__file__).resolve().parents[1]

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(
                    process_one,
                    row,
                    client=client,
                    deployment=args.deployment,
                    system_prompt=system_prompt,
                    user_template=user_template,
                    output_dir=output_dir,
                    image_dir=image_dir,
                    root_dir=root_dir,
                    max_tokens=args.max_completion_tokens,
                    retries=args.retries,
                    retry_sleep=args.retry_sleep,
                ): str(row.get("id") or "unknown")
                for row in rows
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                sample_id = result["sample_id"]
                if result["status"] == "success":
                    print(f"[ok] {sample_id} -> {result['html_path']}", flush=True)
                else:
                    print(f"[error] {sample_id} -> {result['error']}", flush=True)
    elif args.provider == "qwen":
        from inference.providers.qwen import load_qwen3_vl_model

        bundle = load_qwen3_vl_model(
            model_name=args.model_name,
            dtype=args.torch_dtype,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation or None,
        )
        image_dir = args.image_dir.resolve()
        root_dir = Path(__file__).resolve().parents[1]

        if args.workers != 1:
            print("[warn] qwen provider runs sequentially; ignoring --workers and using 1 worker.", flush=True)

        for row in rows:
            result = process_one_qwen(
                row,
                bundle=bundle,
                system_prompt=system_prompt,
                user_template=user_template,
                output_dir=output_dir,
                image_dir=image_dir,
                root_dir=root_dir,
                max_tokens=args.max_completion_tokens,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
            )
            results.append(result)
            sample_id = result["sample_id"]
            if result["status"] == "success":
                print(f"[ok] {sample_id} -> {result['html_path']}", flush=True)
            else:
                print(f"[error] {sample_id} -> {result['error']}", flush=True)
    elif args.provider == "custom":
        from inference.providers.custom import load_custom_model

        if not args.custom_provider_ref:
            raise ValueError("--custom-provider-ref is required when --provider custom")
        if args.custom_model_path is None:
            raise ValueError("--custom-model-path is required when --provider custom")

        bundle = load_custom_model(
            provider_ref=args.custom_provider_ref,
            model_path=args.custom_model_path.resolve(),
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            config_path=args.custom_config_json.resolve() if args.custom_config_json else None,
        )
        image_dir = args.image_dir.resolve()
        root_dir = Path(__file__).resolve().parents[1]

        if args.workers != 1:
            print("[warn] custom provider runs sequentially; ignoring --workers and using 1 worker.", flush=True)

        for row in rows:
            result = process_one_custom(
                row,
                bundle=bundle,
                system_prompt=system_prompt,
                user_template=user_template,
                output_dir=output_dir,
                image_dir=image_dir,
                root_dir=root_dir,
                max_tokens=args.max_completion_tokens,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
            )
            results.append(result)
            sample_id = result["sample_id"]
            if result["status"] == "success":
                print(f"[ok] {sample_id} -> {result['html_path']}", flush=True)
            else:
                print(f"[error] {sample_id} -> {result['error']}", flush=True)
    elif args.provider == "silicon":
        from inference.providers.silicon import load_silicon_api_key

        api_key = args.silicon_api_key.strip() or load_silicon_api_key()
        base_url = args.silicon_base_url
        model = args.silicon_model
        image_dir = args.image_dir.resolve() if args.image_dir else None
        root_dir = Path(__file__).resolve().parents[1]

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(
                    process_one_silicon,
                    row,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    system_prompt=system_prompt,
                    user_template=user_template,
                    output_dir=output_dir,
                    image_dir=image_dir,
                    root_dir=root_dir,
                    max_tokens=args.max_completion_tokens,
                    retries=args.retries,
                    retry_sleep=args.retry_sleep,
                ): str(row.get("id") or "unknown")
                for row in rows
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                sample_id = result["sample_id"]
                if result["status"] == "success":
                    print(f"[ok] {sample_id} -> {result['html_path']}", flush=True)
                else:
                    print(f"[error] {sample_id} -> {result['error']}", flush=True)

    if args.failed_log:
        failed_rows = [result for result in results if result["status"] != "success"]
        args.failed_log.resolve().write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in failed_rows)
            + ("\n" if failed_rows else ""),
            encoding="utf-8",
        )

    manifest_path = output_dir / "outputs_manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in results) + ("\n" if results else ""),
        encoding="utf-8",
    )

    success_count = sum(1 for result in results if result["status"] == "success")
    failed_count = len(results) - success_count
    print(f"outputs_manifest={manifest_path}", flush=True)
    print(f"completed={len(results)} success={success_count} failed={failed_count}")


if __name__ == "__main__":
    main()
