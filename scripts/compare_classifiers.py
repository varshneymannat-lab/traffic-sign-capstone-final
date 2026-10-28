from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data.sign_datamodule import TrafficSignDataModule
from src.utils.video_pipeline import load_classifier


@torch.inference_mode()
def evaluate(model, loader, classes):
    actual, predicted = [], []
    for batch in loader:
        predicted.extend(model(batch["image"]).argmax(1).tolist())
        actual.extend(batch["sign_label"].tolist())
    labels = list(range(len(classes)))
    return {
        "samples": len(actual),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(precision_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "per_class": classification_report(actual, predicted, labels=labels, target_names=classes, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
        "predictions": [{"actual": classes[a], "predicted": classes[p]} for a, p in zip(actual, predicted)],
    }


@torch.inference_mode()
def latency(model, image_size, repeats):
    example = torch.zeros(1, 3, image_size, image_size)
    for _ in range(10):
        model(example)
    start = time.perf_counter()
    for _ in range(repeats):
        model(example)
    return 1000 * (time.perf_counter() - start) / repeats


def save_confusion(result, classes, destination, title):
    matrix = result["confusion_matrix"]
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.imshow(matrix, cmap="Greens")
    axis.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes, xlabel="Predicted category", ylabel="Actual category", title=title)
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    maximum = max(max(row) for row in matrix)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            axis.text(j, i, str(value), ha="center", va="center", color="white" if maximum and value > maximum / 2 else "black")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Compare three classifiers using the same fixed data split.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", default=["stop", "no_entry", "yield", "speed_limit_30", "speed_limit_50"])
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/classifier_comparison")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--latency-repeats", type=int, default=30)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args()
    training_device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    if min(args.epochs, args.batch_size, args.threads, args.latency_repeats, args.image_size) < 1:
        parser.error("Epochs, batch size, threads, and repeats must be positive.")
    data_dir = args.data_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new output directory to preserve previous experiments.")
    annotations_path = data_dir / "processed/annotations_split.csv"
    annotations = pd.read_csv(annotations_path)
    required = {"image_path", "sign_label", "split"}
    if not required.issubset(annotations.columns):
        parser.error("Prepared annotations must include image_path, sign_label, and split.")
    if not set(annotations.sign_label).issubset(args.classes):
        parser.error("Annotations include categories outside the requested classes.")
    for split in ["train", "val", "test"]:
        if set(annotations[annotations.split == split].sign_label) != set(args.classes):
            parser.error(f"Every category must appear in the {split} group.")
    if annotations.groupby("image_path").split.nunique().max() > 1:
        parser.error("An image appears in multiple groups. Split images before object rows.")
    if "group_id" in annotations and annotations.groupby("group_id").split.nunique().max() > 1:
        parser.error("Related images appear in multiple groups.")
    output.mkdir(parents=True, exist_ok=True)
    annotations.to_csv(output / "fixed_annotations_split.csv", index=False)
    torch.set_num_threads(args.threads)
    variants = {"resnet18": "sign_resnet", "mobilenet_v3_small": "sign_mobilenet", "efficientnet_b0": "sign_efficientnet"}
    all_results = []
    common = {
        "epochs": args.epochs, "batch_size": args.batch_size, "training_seed": args.seed,
        "classes": args.classes,
        "pretrained": not args.from_scratch, "image_size": args.image_size,
        "learning_rate": 0.0003, "weight_decay": 0.0001, "optimizer": "AdamW",
        "checkpoint_selection": "Highest validation macro-F1 within each run",
        "augmentation": "Resize, ColorJitter, RandomRotation(8 degrees); no horizontal flipping",
        "annotation_sha256": hashlib.sha256(annotations_path.read_bytes()).hexdigest(),
        "split_counts": annotations.groupby(["split", "sign_label"]).size().unstack(fill_value=0).to_dict(),
        "hardware": {"platform": platform.platform(), "training_device": training_device, "evaluation_device": "cpu", "torch_threads": args.threads},
        "selection_rule": "Highest validation macro-F1, then lower CPU latency for an exact tie. Test scores do not select the model.",
        "latency_scope": "Single-image forward pass after warm-up, excluding image loading and preprocessing. Not full video speed.",
    }
    for architecture, config in variants.items():
        run = output / architecture
        command = [sys.executable, "-m", "src.train", f"model={config}",
                   f"paths.data_dir={json.dumps(str(data_dir))}",
                   f"paths.model_dir={json.dumps(str(run / 'models'))}",
                   f"paths.log_dir={json.dumps(str(run / 'logs'))}",
                   f"paths.output_dir={json.dumps(str(run / 'training'))}",
                   f"data.sign_classes={json.dumps(args.classes)}",
                   f"data.image_size={args.image_size}",
                   f"model.pretrained={str(not args.from_scratch).lower()}",
                   f"trainer.max_epochs={args.epochs}", f"trainer.accelerator={training_device}",
                   f"data.batch_size={args.batch_size}", "data.num_workers=0",
                   f"seed={args.seed}", "test_after_training=false", "trainer.log_every_n_steps=1",
                   "+trainer.enable_progress_bar=false", "callbacks.model_checkpoint.monitor=val/f1", "callbacks.model_checkpoint.mode=max"]
        start = time.perf_counter()
        run.mkdir(parents=True, exist_ok=True)
        print(f"Training {architecture}: {args.epochs} epochs at {args.image_size}px", flush=True)
        with (run / "training.log").open("w") as log:
            subprocess.run(command, cwd=PROJECT, check=True, stdout=log, stderr=subprocess.STDOUT,
                           env={**os.environ, "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads)})
        training_seconds = time.perf_counter() - start
        checkpoint = run / "models/checkpoints/best.ckpt"
        model = load_classifier(checkpoint, args.classes, "cpu")
        data = TrafficSignDataModule(str(data_dir), str(annotations_path), args.image_size, args.batch_size, 0, args.classes)
        data.setup()
        validation = evaluate(model, data.val_dataloader(), args.classes)
        test = evaluate(model, data.test_dataloader(), args.classes)
        result = {"architecture": architecture, "validation": validation, "test": test,
                  "parameter_count": sum(p.numel() for p in model.parameters()),
                  "checkpoint_bytes": checkpoint.stat().st_size,
                  "training_seconds": training_seconds,
                  "cpu_latency_ms": latency(model, args.image_size, args.latency_repeats),
                  "settings": common}
        (run / "evaluation.json").write_text(json.dumps(result, indent=2))
        save_confusion(test, args.classes, run / "test_confusion_matrix.png", architecture)
        all_results.append(result)
        print(f"Finished {architecture}; validation macro-F1={validation['macro_f1']:.4f}", flush=True)
    selected = max(all_results, key=lambda item: (item["validation"]["macro_f1"], -item["cpu_latency_ms"]))["architecture"]
    (output / "comparison.json").write_text(json.dumps({"selected_architecture": selected, "experiments": all_results, "settings": common}, indent=2))
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["architecture", "validation_macro_f1", "test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1", "parameter_count", "cpu_latency_ms", "training_seconds"])
        writer.writeheader()
        for item in all_results:
            writer.writerow({"architecture": item["architecture"], "validation_macro_f1": item["validation"]["macro_f1"],
                             **{f"test_{key}": item["test"][key] for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]},
                             **{key: item[key] for key in ["parameter_count", "cpu_latency_ms", "training_seconds"]}})
    print(f"Selected using validation results: {selected}")
    print(f"Comparison saved to {output}")


if __name__ == "__main__":
    main()
