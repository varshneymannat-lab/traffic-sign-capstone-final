from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import markdown
import numpy as np
from PIL import Image


PROPOSED = {14: "Stop", 17: "No entry", 13: "Yield", 1: "Speed limit 30", 2: "Speed limit 50"}
SOURCE = "https://github.com/houbensebastian/GermanTrafficSignBenchmarks"


def numeric_summary(values):
    array = np.asarray(values)
    return {"min": float(array.min()), "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)), "max": float(array.max())}


def duplicate_summary(groups):
    repeated = [rows for rows in groups.values() if len(rows) > 1]
    return {"duplicate_groups": len(repeated),
            "extra_copies": sum(len(rows) - 1 for rows in repeated),
            "cross_split_groups": sum(len({row[0] for row in rows}) > 1 for rows in repeated),
            "conflicting_label_groups": sum(len({row[1] for row in rows}) > 1 for rows in repeated)}


def read_annotations(archive, split):
    rows = []
    for member in sorted(archive.namelist()):
        if not member.lower().endswith(".csv"):
            continue
        for row in csv.DictReader(io.StringIO(archive.read(member).decode("utf-8-sig")), delimiter=";"):
            row = {key.lower(): value for key, value in row.items()}
            category = int(row["classid"])
            filename = row["filename"]
            image_member = ((Path(member).parent / filename).as_posix() if split == "train"
                            else f"GTSRB/Final_Test/Images/{filename}")
            rows.append({"split": split, "class_id": category, "filename": filename,
                         "member": image_member, "width": int(row["width"]), "height": int(row["height"]),
                         "x1": int(row["roi.x1"]), "y1": int(row["roi.y1"]),
                         "x2": int(row["roi.x2"]), "y2": int(row["roi.y2"]),
                         "track": f"{category}:{Path(filename).stem.split('_')[0]}" if split == "train" else ""})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Inspect official GTSRB images and produce a reproducible EDA report.")
    parser.add_argument("--archives-dir", type=Path, default=Path("data/raw/gtsrb/archives"))
    parser.add_argument("--output-dir", type=Path, default=Path("report/gtsrb_eda"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archives_dir / "GTSRB_Final_Training_Images.zip") as training, \
            zipfile.ZipFile(args.archives_dir / "GTSRB_Final_Test_Images.zip") as testing, \
            zipfile.ZipFile(args.archives_dir / "GTSRB_Final_Test_GT.zip") as labels:
        rows = read_annotations(training, "train") + read_annotations(labels, "test")
        counts = {split: Counter(r["class_id"] for r in rows if r["split"] == split) for split in ["train", "test"]}
        failures, size_mismatches, invalid_rois = [], [], []
        hashes = defaultdict(list)
        brightness = []
        for index, row in enumerate(rows):
            archive = training if row["split"] == "train" else testing
            try:
                with Image.open(io.BytesIO(archive.read(row["member"]))) as image:
                    image.load()
                    rgb = image.convert("RGB")
                    if image.size != (row["width"], row["height"]):
                        size_mismatches.append(row["member"])
                    width, height = image.size
                    if not (0 <= row["x1"] <= row["x2"] < width and 0 <= row["y1"] <= row["y2"] < height):
                        invalid_rois.append(row["member"])
                    digest = hashlib.sha256(f"{width}x{height}:".encode() + rgb.tobytes()).hexdigest()
                    hashes[digest].append((row["split"], row["class_id"], row["member"]))
                    if row["split"] == "train":
                        brightness.append(float(np.asarray(rgb.convert("L")).mean()))
            except (OSError, ValueError, KeyError) as exc:
                failures.append({"member": row["member"], "error": str(exc)})
            if (index + 1) % 10000 == 0:
                print(f"Checked {index + 1:,} images", flush=True)

        train_rows = [r for r in rows if r["split"] == "train"]
        tracks = Counter(r["track"] for r in train_rows)
        duplicate_files = [[{"split": split, "class_id": category, "member": member}
                            for split, category, member in entries]
                           for entries in hashes.values() if len(entries) > 1]
        (args.output_dir / "exact_duplicate_files.json").write_text(json.dumps(duplicate_files, indent=2))
        summary = {"source": SOURCE, "task": "Single-image traffic-sign classification",
                   "counts": {split: dict(sorted(c.items())) for split, c in counts.items()},
                   "total_images": len(rows), "training_images": len(train_rows),
                   "test_images": len(rows) - len(train_rows), "classes": len(counts["train"]),
                   "training_class_imbalance_ratio": max(counts["train"].values()) / min(counts["train"].values()),
                   "training_image_width": numeric_summary([r["width"] for r in train_rows]),
                   "training_image_height": numeric_summary([r["height"] for r in train_rows]),
                   "training_mean_grayscale_brightness_0_to_255": numeric_summary(brightness),
                   "training_tracks": len(tracks), "images_per_training_track": numeric_summary(list(tracks.values())),
                   "unreadable_images": failures, "annotation_size_mismatches": size_mismatches,
                   "invalid_inclusive_rois": invalid_rois, "exact_decoded_rgb_duplicates": duplicate_summary(hashes),
                   "proposed_classes_not_final": {str(k): v for k, v in PROPOSED.items()},
                   "test_policy": "Test labels used for counts and images checked for integrity/duplicates only. No model trained or selected. Galleries and descriptive distributions use training images only.",
                   "archive_manifest_sha256": hashlib.sha256((args.archives_dir / "download_manifest.json").read_bytes()).hexdigest()}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        with (args.output_dir / "class_counts.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["class_id", "proposed_name", "official_train", "official_test", "training_tracks"])
            for category in sorted(counts["train"]):
                writer.writerow([category, PROPOSED.get(category, ""), counts["train"][category], counts["test"][category],
                                 len({r["track"] for r in train_rows if r["class_id"] == category})])

        plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
        fig, ax = plt.subplots(figsize=(13, 4.5))
        ids = sorted(counts["train"])
        ax.bar(ids, [counts["train"][c] for c in ids], color=["#159b85" if c in PROPOSED else "#87939d" for c in ids])
        ax.set(xticks=ids, xlabel="Official class ID", ylabel="Training images", title="GTSRB: class sizes are unequal (green = proposed subset)")
        fig.tight_layout()
        fig.savefig(args.output_dir / "class_distribution.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        axes[0].hist([r["width"] for r in train_rows], bins=40, color="#168d7a")
        axes[0].set(title="Training image width", xlabel="Pixels", ylabel="Images")
        axes[1].hist([r["height"] for r in train_rows], bins=40, color="#5978a6")
        axes[1].set(title="Training image height", xlabel="Pixels")
        axes[2].hist(brightness, bins=40, color="#b66b46")
        axes[2].set(title="Training brightness", xlabel="Mean grayscale value (0-255)")
        fig.tight_layout()
        fig.savefig(args.output_dir / "image_properties.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        for line, (category, name) in enumerate(PROPOSED.items()):
            candidates = [r for r in train_rows if r["class_id"] == category]
            by_track = {}
            for row in candidates:
                by_track.setdefault(row["track"], []).append(row)
            track_names = sorted(by_track)
            for column, pos in enumerate(np.linspace(0, len(track_names) - 1, 5, dtype=int)):
                entries = by_track[track_names[pos]]
                row = entries[len(entries) // 2]
                with Image.open(io.BytesIO(training.read(row["member"]))) as image:
                    axes[line, column].imshow(image, interpolation="nearest")
                axes[line, column].set_title(f"{name}\n{row['width']} x {row['height']} px", fontsize=9)
                axes[line, column].axis("off")
        fig.suptitle("Real training photographs: five different sign tracks per proposed category", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(args.output_dir / "real_image_examples.png", dpi=130)
        plt.close(fig)

    table = "\n".join(f"| {name} | {counts['train'][c]:,} | {counts['test'][c]:,} |" for c, name in PROPOSED.items())
    duplicate = summary["exact_decoded_rgb_duplicates"]
    report = f"""# GTSRB Exploratory Data Analysis

## Purpose
Inspect the real dataset before choosing final categories or training models. This analysis does not report model accuracy.

## Dataset Overview
- Source: [GTSRB dataset authors]({SOURCE}).
- Locally verified: **{len(rows):,} images**, **{len(train_rows):,} official training images**, **{len(rows) - len(train_rows):,} official test images**, **{len(counts['train'])} classes**.
- Each image shows an individual traffic sign with some surrounding background. This supports classification, not a complete road-video detector.
- The official test set is preserved. Galleries and descriptive distributions below use training images only; test images are checked for integrity and exact duplicates.

## Class Distribution
![Class distribution](class_distribution.png)

Training class counts range from {min(counts['train'].values()):,} to {max(counts['train'].values()):,}, a {summary['training_class_imbalance_ratio']:.2f}:1 ratio. Accuracy alone can hide poor performance on smaller classes. Report macro-F1, per-class precision and recall, and a confusion matrix alongside accuracy.

### Proposed Five-Class Subset (Not Yet Final)
| Category | Official Training Images | Official Test Images |
| --- | ---: | ---: |
{table}

Each proposed category has enough images for 500 training and 150 validation examples. These are image counts, not counts of independent physical signs. Traffic-light objects are not a GTSRB class, so this subset changes the original category list.

## Image Size and Appearance
![Image properties](image_properties.png)

Training widths: {summary['training_image_width']}. Training heights: {summary['training_image_height']}. Images need a consistent model input size. Enlarging tiny images does not recover missing detail. Resizing and normalization must be identical across classifier comparisons. Brightness is only a descriptive pixel statistic, not a label for lighting quality or a blur measurement.

![Real image examples](real_image_examples.png)

The gallery uses real training photographs from different sign tracks, without synthetic sign generation or brightness enhancement. Enlargement is for viewing only. Small or dark signs can be harder to recognise. Similar speed-limit shapes make the printed number important. Avoid horizontal flipping because it can reverse numbers or sign meanings.

## Data Quality Checks
- All {len(rows):,} annotated images were opened and decoded.
- Unreadable/missing images: {len(failures)}.
- Image size versus annotation mismatches: {len(size_mismatches)}.
- Out-of-bounds/invalid ROIs under an inclusive-maximum interpretation: {len(invalid_rois)}. Detailed file lists are in summary.json; any flagged rows require review before cropping.
- Exact decoded-RGB duplicate groups: {duplicate['duplicate_groups']}; extra copies: {duplicate['extra_copies']}.
- Exact duplicate groups shared across official train/test: {duplicate['cross_split_groups']}.
- Exact duplicate groups with conflicting category labels: {duplicate['conflicting_label_groups']}.
- The affected files are listed in exact_duplicate_files.json. Before training, exclude affected training tracks from the development pool while keeping the official test set unchanged, then record the post-filtering counts. The proposed subset must be checked against this list too.
- This exact hash check does not detect near-duplicates or prove the labels are all correct.

## Leakage and Splitting
There are {len(tracks):,} training sign tracks. Images per track: {summary['images_per_training_track']}. A track contains related views of the same physical sign. Use disjoint tracks for training and validation, not an unrestricted random image split. Keep the official test set out of tuning and checkpoint selection. Do not claim that 500 images are 500 distinct signs.

## Decisions Before Training
1. Confirm a classification-only scope and final categories with the mentor.
2. Remove training tracks affected by exact train/test duplication before sampling. If the proposed subset is approved, verify the remaining counts and use the same 500 training and 150 validation images per class for each model, grouped by sign track; retain all official test examples of those classes.
3. Compare ResNet-18, MobileNetV3-Small and EfficientNet-B0 with consistent preprocessing, data split and training settings. Use validation macro-F1 for model selection, then report test metrics once the selection is fixed.
4. Report model size and measured inference latency alongside recognition quality. Model comparisons, exports and revised presentation results remain future work.
5. Existing synthetic checkpoints, scores and video are not results on this dataset.

## Reproduce
From the repository root: `python scripts/eda_gtsrb.py`. The script reads the downloaded archives without modifying the source images. Raw counts and checks are saved in class_counts.csv and summary.json. Archive provenance and SHA-256 hashes are recorded by the download script in download_manifest.json.

## Reference
J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification competition. IEEE IJCNN, pp. 1453-1460, 2011.
"""
    (args.output_dir / "README.md").write_text(report)
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GTSRB EDA</title><style>body{font:16px/1.6 system-ui,sans-serif;color:#202b32;background:#fff;margin:32px auto;padding:0 20px;max-width:1100px}h1{font-size:30px}h2{margin-top:32px;font-size:23px}img{display:block;width:100%;height:auto}a{color:#087a68}table{border-collapse:collapse;max-width:100%}td,th{border-bottom:1px solid #d9dfe1;padding:8px 16px;text-align:left}code{overflow-wrap:anywhere}li{margin:6px 0}</style><main>' + markdown.markdown(report, extensions=["tables"]) + "</main></html>"
    (args.output_dir / "gtsrb_eda.html").write_text(page)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
