"""Run Qwen2.5-14B-Instruct-1M on OCRBench v2 English KIE document images.

Requires CUDA. Output is an OCRBench-compatible JSON list with a 'predict' string.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
import time

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct-1M"
MODEL_REVISION = "620fad32de7bdd2293b3d99b39eba2fe63e97438"
TASK_TYPE = "key information extraction en"
SUFFIX = (
    "\nReturn only a JSON object mapping each requested key to its extracted string value. "
    "Do not invent values or include commentary, reasoning, or Markdown."
)


def resolve_text(dataset_root: Path, text_path: str) -> Path:
    """Find OCR text in data/ocr/ using the sample ID."""
    if not isinstance(text_path, str) or not text_path.strip():
        raise ValueError("Missing text_path in benchmark record")
    
    raw = Path(text_path)

    if raw.is_absolute():
        candidates = [
            Path("data/ocr") / f"{raw.stem}.txt",
            dataset_root / "data/ocr" / f"{raw.stem}.txt",
            dataset_root.parent / "data/ocr" / f"{raw.stem}.txt",
        ]
    else:
        candidates = [
            dataset_root / raw, 
            dataset_root.parent / raw]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
        
    raise FileNotFoundError(f"Text not found: {text_path!r}; tried {candidates}")


def record_id(item):
    """Avoid the assumption that benchmark sample IDs are globally unique."""
    return json.dumps(
        [item.get("dataset_name"), item.get("type"), item.get("id"),
         item.get("text_path"), item.get("question")],
        ensure_ascii=False,
    )


def strip_reasoning(output: str) -> str:
    output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL)
    if "</think>" in output:
        output = output.rsplit("</think>", 1)[-1]
    return output.strip()


def write_json_atomic(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as handle:
            tmp_path = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def synchronize_cuda(torch):
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)


def memory_stats(torch):
    if not torch.cuda.is_available():
        return []
    return [int(torch.cuda.max_memory_allocated(i)) for i in range(torch.cuda.device_count())]


def generate_one(processor, model, torch, text_path: Path, question: str,
                 max_new_tokens: int, enable_thinking: bool):

    text = text_path.read_text(encoding="utf-8-sig")

    messages = [{"role": "user", "content": text + "\n\n" + question + SUFFIX}]

    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    synchronize_cuda(torch)
    start = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=True,
        return_dict=True, 
        return_tensors="pt", 
        enable_thinking=enable_thinking,
    ).to(model.device)

    with torch.inference_mode():
        tokens = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    synchronize_cuda(torch)
    elapsed = time.perf_counter() - start
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    generated_tokens = tokens[0][prompt_tokens:]

    prediction = strip_reasoning(processor.decode(generated_tokens, skip_special_tokens=True))

    return prediction, elapsed, memory_stats(torch), int(generated_tokens.numel())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="English KIE subset JSON")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Extracted OCRBench_v2 directory")
    parser.add_argument("--output", type=Path, required=True, help="Predictions JSON")
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face model ID or local path")
    parser.add_argument("--revision", default=MODEL_REVISION, help="Pinned Hugging Face checkpoint revision")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking (off for baseline)")
    parser.add_argument("--retry-failures", action="store_true", help="Re-attempt previous failed samples")
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.output.resolve() == args.input.resolve():
        parser.error("--output must not overwrite input benchmark annotations")
    with args.input.open(encoding="utf-8") as handle:
        examples = json.load(handle)

    if not isinstance(examples, list) or not examples:
        raise ValueError("Input must be a nonempty JSON list")
    
    if any(
        not isinstance(item, dict) or item.get("type") != TASK_TYPE for item in examples):
        raise ValueError("Input must contain only OCRBench v2 English KIE records")
    identities = [record_id(item) for item in examples]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate sample identities in benchmark input")

    metadata_path = args.output.with_suffix(".run_metadata.json")
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as handle:
            prior_settings = json.load(handle)
        for name, new_value in {
            "model": args.model,
            "revision": args.revision,
            "max_new_tokens": args.max_new_tokens,
            "enable_thinking": args.enable_thinking,
        }.items():
            if prior_settings.get(name) != new_value:
                raise ValueError(f"Existing predictions use different {name}; select a fresh output path.")

    existing = {}
    if args.output.exists():
        with args.output.open(encoding="utf-8") as handle:
            previous = json.load(handle)
        if not isinstance(previous, list):
            raise ValueError("Existing output must be a JSON list")
        existing = {record_id(row): row for row in previous}
        unfamiliar = set(existing) - set(identities)
        if unfamiliar:
            raise ValueError("Output contains other samples: use a new output path")

    pending = [item for item in examples if record_id(item) not in existing or
               (args.retry_failures and existing[record_id(item)].get("error"))]
    if not pending:
        print(f"Already finished {len(existing)} samples; nothing to generate")
        return

    # Lazy imports keep unit tests and dataset selection runnable without model weights.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. Request a GPU node via SLURM before loading Qwen2.5-14B-Instruct-1M.")
    start_load = time.perf_counter()
    print(f"Loading {args.model} on {torch.cuda.device_count()} CUDA device(s)...", flush=True)
    
    processor = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, device_map="auto", dtype="auto"
    ).eval()
    
    load_sec = time.perf_counter() - start_load
    print(f"Model loaded in {load_sec:.1f}s", flush=True)

    results = dict(existing)
    run_metadata = {
        "model": args.model, "revision": args.revision, "task": TASK_TYPE,
        "benchmark_input": str(args.input), "dataset_root": str(args.dataset_root),
        "max_new_tokens": args.max_new_tokens, "enable_thinking": args.enable_thinking,
        "do_sample": False, "model_load_seconds": round(load_sec, 3),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    write_json_atomic(metadata_path, run_metadata)

    for index, item in enumerate(examples, start=1):
        key = record_id(item)
        if key in existing and (not args.retry_failures or not existing[key].get("error")):
            continue
        row = dict(item)
        try:
            text_path = resolve_text(args.dataset_root, item["text_path"])
            prediction, duration, mem, n_tokens = generate_one(
                processor, 
                model, 
                torch, 
                text_path, 
                str(item["question"]),
                args.max_new_tokens, args.enable_thinking
            )
            row.update(predict=prediction, inference_seconds=round(duration, 4),
                       peak_allocated_gpu_bytes=mem, output_tokens=n_tokens)
            row.pop("error", None)
            status = "OK"
        except Exception as error:
            row.update(predict="", error=f"{type(error).__name__}: {error}")
            status = "ERROR"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results[key] = row
        # Keep the benchmark's JSON list structure; save atomically after each sample.
        write_json_atomic(args.output, [results[record_id(x)] for x in examples if record_id(x) in results])
        print(f"[{index}/{len(examples)}] {status} {item.get('id')} {str(row.get('predict', ''))[:100]!r}",
              flush=True)
    print(f"Predictions saved: {args.output}")


if __name__ == "__main__":
    main()
