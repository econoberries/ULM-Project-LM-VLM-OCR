# Import libraries
import os
import csv
import time
import platform
import statistics
import threading
import psutil
from datetime import datetime
from paddleocr import PaddleOCR
import json
import statistics
import argparse

"""
uv run python scripts/ocr.py --input_dir data/OCRBench_v2/EN_part --output_dir data/ocr_output
"""


def get_process_memory_mb(process):
    #Return current process RSS memory in MB
    return process.memory_info().rss / (1024 ** 2)


def arrange_text(results, line_tolerance_ratio=0.01, char_width_ratio=0.01):
    items = []

    for item in results:
        bbox = item.get("bbox")
        if not bbox or len(bbox) < 4:
            continue

        x1, y1, x2, y2 = bbox[:4]
        items.append({
            "text": item["text"].strip(),
            "x": x1,
            "y": (y1 + y2) / 2,
            "height": y2 - y1,
        })

    if not items:
        return ""

    # Estimate document scale from the detected text.
    min_x = min(item["x"] for item in items)
    max_x = max(item["x"] for item in items)
    avg_height = sum(item["height"] for item in items) / len(items)

    y_tolerance = max(1, avg_height * 0.5)
    char_width = max(1, avg_height * 0.5)

    # Group detections by vertical center.
    items.sort(key=lambda item: item["y"])
    lines = []

    for item in items:
        matching_line = next(
            (
                line for line in reversed(lines)
                if abs(item["y"] - line["y"]) <= y_tolerance
            ),
            None,
        )

        if matching_line:
            matching_line["items"].append(item)
            matching_line["y"] = sum(
                entry["y"] for entry in matching_line["items"]
            ) / len(matching_line["items"])
        else:
            lines.append({"y": item["y"], "items": [item]})

    # Recreate approximate horizontal positions.
    output = []

    for line in lines:
        line["items"].sort(key=lambda item: item["x"])
        text_line = ""
        previous_x = min_x

        for item in line["items"]:
            spaces = max(0, round((item["x"] - previous_x) / char_width))
            text_line += " " * spaces + item["text"]
            previous_x = item["x"] + len(item["text"]) * char_width

        output.append(text_line.rstrip())

    return "\n".join(output)


def image_to_text(image_path, output_txt, ocr):
    # Run OCR and save recognized text plus bounding boxes.

    start_time = time.perf_counter()

    result = ocr.predict(image_path)

    all_text = []
    text_items = []

    for res in result:
        data = getattr(res, "json", None)

        if callable(data):
            data = data()

        if not isinstance(data, dict):
            continue

        data = data.get("res", data)

        texts = data.get("rec_texts", [])
        boxes = data.get("rec_boxes")
        polygons = data.get("rec_polys")
        scores = data.get("rec_scores", [])

        # Convert arrays (including NumPy arrays) to Python lists.
        if boxes is not None and hasattr(boxes, "tolist"):
            boxes = boxes.tolist()

        if polygons is not None and hasattr(polygons, "tolist"):
            polygons = polygons.tolist()

        for i, text in enumerate(texts):
            all_text.append(text)

            item = {
                "text": text,
                "confidence": (
                    float(scores[i])
                    if i < len(scores) else None
                ),
            }

            if boxes is not None and i < len(boxes):
                # [x_min, y_min, x_max, y_max]
                item["bbox"] = boxes[i]

            elif polygons is not None and i < len(polygons):
                # Four corner points, preserving rotated boxes.
                item["polygon"] = polygons[i]

            else:
                item["bbox"] = None

            text_items.append(item)

    # Save the original plain-text output.
    #with open(output_txt, "w", encoding="utf-8") as f:
     #   f.write("\n".join(all_text))

    # Arrange text
    formatted_text = arrange_text(text_items)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(formatted_text)

    # Save structured OCR output as JSON.
    output_json = os.path.splitext(output_txt)[0] + ".json"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image": os.path.basename(image_path),
                "results": text_items, 
                "formatted_text": formatted_text,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    elapsed = time.perf_counter() - start_time

    return {
        "inference_and_save_seconds": elapsed,
        "characters_extracted": sum(len(t) for t in all_text),
        "text_output_bytes": os.path.getsize(output_txt),
    }


def main():

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Path to the image data folder")
    parser.add_argument("--output_dir", required=True, help="Path to the output folder")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = os.path.join(args.output_dir, f"extracted_text_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    metrics_dir = os.path.join(output_dir, "benchmark_results")
    os.makedirs(metrics_dir, exist_ok=True)

    # Find all subfolders starting with "kie"
    kie_folders = [
        "kie_funsd",
        "kie_poie",
        "kie_sroie",
        "funsd_test_images",
        "receipt_test_images",
        "sroie_test_images",
        "poie_test_images",
        "poie_test_images_new",
    ]

    # Collect images from all matching folders
    image_files = sorted(
        (
            folder,
            filename,
        )
        for folder in kie_folders
        for filename in os.listdir(os.path.join(input_dir, folder))
        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
        and os.path.isfile(os.path.join(input_dir, folder, filename))
    )

    if not image_files:
        print(f"No images found in kie* subfolders of {input_dir}")
        return

    print(f"Found {len(kie_folders)} matching folders.")
    print(f"Found {len(image_files)} images.")

    process = psutil.Process(os.getpid())

    print(f"Found {len(image_files)} images.")
    print("Initializing PaddleOCR...")

    init_start = time.perf_counter()

    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    init_seconds = time.perf_counter() - init_start

    print(f"Model initialization: {init_seconds:.2f} seconds")

    # Monitor peak process memory and CPU usage.
    peak_memory_mb = get_process_memory_mb(process)
    stop_monitor = threading.Event()

    def monitor_resources():
        nonlocal peak_memory_mb

        while not stop_monitor.wait(0.2):
            try:
                peak_memory_mb = max(
                    peak_memory_mb,
                    get_process_memory_mb(process),
                )
            except psutil.Error:
                pass

    monitor = threading.Thread(
        target=monitor_resources,
        daemon=True,
    )
    monitor.start()

    records = []
    successful_times = []
    total_chars = 0
    total_output_bytes = 0
    errors = 0

    # Includes OCR and text-file saving, excludes model initialization.
    batch_start = time.perf_counter()
    process.cpu_percent(interval=None)

    try:
        for index, (folder, filename) in enumerate(image_files, start=1):
            image_path = os.path.join(input_dir, folder, filename)


            output_filename = (f"{folder}_{os.path.splitext(filename)[0]}.txt")
            output_txt = os.path.join(
                output_dir,
                os.path.splitext(filename)[0] + ".txt",
            )

            memory_before_mb = get_process_memory_mb(process)

            print(
                f"[{index}/{len(image_files)}] "
                f"Processing {folder}/{filename}..."
            )

            try:
                metrics = image_to_text(image_path, output_txt, ocr)

                elapsed = metrics["inference_and_save_seconds"]
                successful_times.append(elapsed)
                total_chars += metrics["characters_extracted"]
                total_output_bytes += metrics["text_output_bytes"]

                records.append({
                    "filename": filename,
                    "status": "success",
                    **metrics,
                    "memory_before_mb": round(memory_before_mb, 2),
                    "error": "",
                })

                print(f"  Completed in {elapsed:.2f} seconds")

            except Exception as e:
                errors += 1

                records.append({
                    "filename": filename,
                    "status": "error",
                    "inference_and_save_seconds": "",
                    "characters_extracted": "",
                    "text_output_bytes": "",
                    "memory_before_mb": round(memory_before_mb, 2),
                    "error": str(e),
                })

                print(f"  Error: {e}")

    finally:
        batch_seconds = time.perf_counter() - batch_start
        cpu_percent = process.cpu_percent(interval=None)

        stop_monitor.set()
        monitor.join()

    peak_memory_mb = max(
        peak_memory_mb,
        get_process_memory_mb(process),
    )

    processed = len(successful_times)

    avg_seconds = (
        statistics.mean(successful_times) if processed else 0
    )
    median_seconds = (
        statistics.median(successful_times) if processed else 0
    )
    throughput = (
        processed / batch_seconds if batch_seconds > 0 else 0
    )

    # Save per-image measurements.
    csv_path = os.path.join(metrics_dir, "per_image_metrics.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "folder",
            "filename",
            "status",
            "inference_and_save_seconds",
            "characters_extracted",
            "text_output_bytes",
            "memory_before_mb",
            "error",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Save overall experiment summary.
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "total_images": len(image_files),
        "successful_images": processed,
        "failed_images": errors,
        "model_initialization_seconds": round(init_seconds, 4),
        "batch_processing_seconds": round(batch_seconds, 4),
        "total_runtime_including_initialization_seconds": round(
            init_seconds + batch_seconds, 4
        ),
        "average_seconds_per_image": round(avg_seconds, 4),
        "median_seconds_per_image": round(median_seconds, 4),
        "throughput_images_per_second": round(throughput, 4),
        "process_cpu_percent_at_end": round(cpu_percent, 2),
        "peak_process_rss_mb": round(peak_memory_mb, 2),
        "total_characters_extracted": total_chars,
        "total_text_output_bytes": total_output_bytes,
    }

    summary_path = os.path.join(metrics_dir, "summary.csv")

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(summary.items())

    print("\n--- Benchmark Summary ---")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nPer-image metrics: {csv_path}")
    print(f"Summary metrics:   {summary_path}")


if __name__ == "__main__":
    main()
