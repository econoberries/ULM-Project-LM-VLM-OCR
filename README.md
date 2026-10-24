# Understanding LM, VLM and OCR for Document Extraction

A reproducible research project comparing **direct vision-language models (VLMs)** with **OCR + language-model (LM) pipelines** for document key information extraction (KIE).

## Research goal

Compare extraction accuracy, inference time, GPU memory usage, and structured-output reliability across approaches using the **OCRBench v2 English KIE** benchmark.

## Project workflow

1. Prepare one shared benchmark selection for every method.
2. Run each direct VLM or OCR-to-LM pipeline.
3. Score predictions using the same official OCRBench v2 evaluation procedure.
4. Summarize each method with a shared `summarize.py`.
5. Compare results and document experimental conditions.

## Repository setup

This is the **new, clean repository**. Scripts, SLURM batch files, tests, and documentation will be introduced step by step after reviewing the existing implementations.

**Data policy:** Do not commit the downloaded OCRBench images or raw dataset, pretrained model weights, Python environments, or large intermediate OCR outputs.
