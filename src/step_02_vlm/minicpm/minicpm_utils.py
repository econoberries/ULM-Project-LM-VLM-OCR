"""
MiniCPM-V-4.6 settings, model loading, inference, timing and GPU memory.
Used by run_one_record.py and run.py; summarize results with src/step_04_evaluation/summarize_all.py.

References
1. HF-CARD, MiniCPM-V-4.6 model card: https://huggingface.co/openbmb/MiniCPM-V-4.6/blob/36f34a661a4bd35d0dc2294cb044d2584646c7d3/README.md
2. HF-GEN, Generation defaults: https://huggingface.co/openbmb/MiniCPM-V-4.6/blob/36f34a661a4bd35d0dc2294cb044d2584646c7d3/generation_config.json
3. MCPV-DOC, MiniCPM-V-4.6 guide: https://github.com/OpenBMB/MiniCPM-V/blob/dfe5db0cabf054c4615c796306a51f9b0ee399ca/docs/minicpm_v4dot6_en.md
4. MCPV-EVAL, OpenBMB evaluation wrapper: https://github.com/OpenBMB/MiniCPM-V/blob/dfe5db0cabf054c4615c796306a51f9b0ee399ca/eval_mm/vlmevalkit/vlmeval/vlm/minicpm_v.py
5. OCRB, OCRBench inference example: https://github.com/Yuliang-Liu/MultimodalOCR/blob/4e363b3f499811b43d792ef0bb33932ef05eca27/OCRBench/example.py
6. OCRB2, OCRBench v2 evaluator: https://github.com/Yuliang-Liu/MultimodalOCR/blob/4e363b3f499811b43d792ef0bb33932ef05eca27/OCRBench_v2/eval_scripts/eval.py
7. TF, Transformers 5.19 processor API: https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/processing_utils.py
8. QWEN, Qwen3.5 inference script: src/step_02_vlm/qwen35/run.py, branch feature/qwen35-english-kie, commit 969c420
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed

logger = logging.getLogger(__name__)

# results/minicpm/ in the repository root
DEFAULT_OUTPUT_FOLDER = Path(__file__).resolve().parents[3] / "results" / "minicpm"

# ---------------------------------------------------------------------------
# Model and inference settings
# ---------------------------------------------------------------------------
# HF-CARD, line 125
MODEL_ID = "openbmb/MiniCPM-V-4.6"
# https://huggingface.co/openbmb/MiniCPM-V-4.6/commits/main
MODEL_REVISION = "36f34a661a4bd35d0dc2294cb044d2584646c7d3"

# HF-CARD, lines 155-161, 215-216; MCPV-DOC, lines 141-147
# Image settings untuned
DOWNSAMPLE_MODE = "16x"
MAX_SLICE_NUMS = 36

# QWEN, line 111
# Same limit as the other pipelines (HF-CARD, line 164 uses 512)
MAX_NEW_TOKENS = 384

# QWEN, lines 17-20
# Same instruction as the other pipelines, appended to the OCRBench question
SUFFIX = (
    "\nReturn only a JSON object mapping each requested key to its extracted string value. "
    "Do not invent values or include commentary, reasoning, or Markdown."
)

# Fixed seed
SEED = 0


def run_metadata(model, model_path, revision, **fields):
    """Run settings, saved as <output>.run_metadata.json."""
    return {
        "model_id": model_path,
        "revision": revision,
        **fields,
        "device": str(model.device),
        "dtype": str(model.dtype),
        "inference_settings": {
            "downsample_mode": DOWNSAMPLE_MODE,
            "max_slice_nums": MAX_SLICE_NUMS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "prompt_suffix": SUFFIX,
            "do_sample": False,
            "num_beams": 1,
            "enable_thinking": False,
            "seed": SEED,
        },
    }


def load_model(model_path=MODEL_ID, revision=MODEL_REVISION):
    """Load the pinned processor and model."""
    set_seed(SEED)

    # HF-CARD, lines 123-130
    processor = AutoProcessor.from_pretrained(model_path, revision=revision)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, revision=revision, dtype="auto", device_map="auto"
    )
    model.eval()
    logger.info(f"Loaded {model_path}@{revision[:7]} on {model.device} with dtype {model.dtype}")
    return model, processor


def run_inference(model, processor, image_path, question, max_new_tokens=MAX_NEW_TOKENS):
    """One image + question -> (raw answer, number of generated tokens)."""

    # MCPV-EVAL, line 695
    image = Image.open(image_path).convert("RGB")

    # HF-CARD, lines 145-153; QWEN, line 85
    # Image first, then OCRBench question + SUFFIX
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question + SUFFIX},
            ],
        }
    ]

    # HF-CARD, lines 157-162; TF, lines 2012, 2065-2073
    # Image options go in processor_kwargs
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
        processor_kwargs={"downsample_mode": DOWNSAMPLE_MODE, "max_slice_nums": MAX_SLICE_NUMS},
    ).to(model.device)

    # HF-CARD, lines 164, 221; HF-GEN, line 3; MCPV-EVAL, lines 505-506
    generated_ids = model.generate(
        **inputs,
        downsample_mode=DOWNSAMPLE_MODE,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )

    # HF-CARD, lines 165-170
    # Decode only the new tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    if len(generated_ids_trimmed[0]) >= max_new_tokens:
        logger.warning(f"Answer reached max_new_tokens={max_new_tokens} and may be truncated.")

    # OCRB2, lines 241-245
    # eval.py parses it
    return output_text[0], len(generated_ids_trimmed[0])


# ---------------------------------------------------------------------------
# Timing and GPU memory, defined as in the Qwen3.5 run
# ---------------------------------------------------------------------------
# QWEN, lines 69-72
def _synchronize_cuda():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


# QWEN, lines 87-90
def start_sample_timer():
    """Reset GPU peak memory and start the clock."""
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    _synchronize_cuda()
    return time.perf_counter()


# QWEN, lines 75-78, 97-98, 197
def stop_sample_timer(start):
    """Return (inference_seconds, peak_allocated_gpu_bytes); empty GPU list on CPU."""
    _synchronize_cuda()
    seconds = round(time.perf_counter() - start, 4)
    peak = [int(torch.cuda.max_memory_allocated(i)) for i in range(torch.cuda.device_count())]
    return seconds, peak


# QWEN, lines 180-182
def environment_info():
    return {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
