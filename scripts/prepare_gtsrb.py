from __future__ import annotations

import argparse
import csv
import io
import hashlib
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image


CLASS_NAMES = {1: "speed_limit_30", 2: "speed_limit_50", 13: "yield", 14: "stop", 17: "no_entry"}
FIELDS = ["image_path", "xmin", "ymin", "xmax", "ymax", "sign_label", "split", "group_id", "original_class_id"]


def split_tracks(rows, train_count, val_count, seed):
    groups = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
    group_ids = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    selections = {}
    offset = 0
    for split, count in [("val", val_count), ("train", train_count)]:
        selected = []
        while len(selected) < count and offset < len(group_ids):
            selected.extend(groups[group_ids[offset]])
            offset += 1
        if len(selected) < count:
            raise ValueError(f"Not enough separate tracks for {count} {split} images.")
        rng.shuffle(selected)
        selections[split] = [{**row, "split": split} for row in selected[:count]]
    return selections


def archive_rows(archive, member, split):
    data = archive.read(member).decode("utf-8-sig")
    parent = Path(member).parent
    result = []
    for row in csv.DictReader(io.StringIO(data), delimiter=";"):
        row = {key.lower(): value for key, value in row.items()}
        category = int(row["classid"])
        if category not in CLASS_NAMES:
            continue
        filename = row["filename"]
        if split == "official_train":
            image_member = (parent / filename).as_posix()
            group = f"train:{category}:{Path(filename).stem.split('_')[0]}"
        else:
            image_member = f"GTSRB/Final_Test/Images/{filename}"
            group = f"test:{filename}"
        result.append({"image_path": f"images/{split}/{category:05d}/{filename}",
                       "xmin": int(row["roi.x1"]), "ymin": int(row["roi.y1"]),
                       "xmax": int(row["roi.x2"]) + 1, "ymax": int(row["roi.y2"]) + 1,
                       "sign_label": CLASS_NAMES[category], "split": split,
                       "group_id": group, "original_class_id": category,
                       "archive_member": image_member})
    return result


def main():
    parser = argparse.ArgumentParser(description="Prepare five GTSRB classes with track-separated validation and official testing.")
    parser.add_argument("--archives-dir", type=Path, default=Path("data/raw/gtsrb/archives"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/gtsrb"))
    parser.add_argument("--class-ids", nargs="+", type=int, required=True)
    parser.add_argument("--train-per-class", type=int, default=500)
    parser.add_argument("--val-per-class", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duplicate-report", type=Path, default=Path("report/gtsrb_eda/exact_duplicate_files.json"))
    args = parser.parse_args()
    if any(category not in CLASS_NAMES for category in args.class_ids):
        parser.error("Supported subset IDs: 1, 2, 13, 14, 17.")
    if len(set(args.class_ids)) != len(args.class_ids):
        parser.error("Class IDs must be unique.")
    if min(args.train_per_class, args.val_per_class) < 1:
        parser.error("Image counts must be positive.")
    if (args.output_dir / "processed/annotations_split.csv").exists():
        parser.error("Prepared data already exists. Use a new output directory.")
    selected_classes = set(args.class_ids)
    duplicates = json.loads(args.duplicate_report.read_text())
    excluded_tracks = set()
    for group in duplicates:
        if {item["split"] for item in group} == {"train", "test"}:
            for item in group:
                if item["split"] == "train":
                    filename = Path(item["member"]).stem
                    excluded_tracks.add(f"train:{item['class_id']}:{filename.split('_')[0]}")
    train_rows, test_rows = [], []
    train_archive = args.archives_dir / "GTSRB_Final_Training_Images.zip"
    test_archive = args.archives_dir / "GTSRB_Final_Test_Images.zip"
    ground_truth = args.archives_dir / "GTSRB_Final_Test_GT.zip"
    with zipfile.ZipFile(train_archive) as archive:
        for name in archive.namelist():
            if name.lower().endswith(".csv"):
                train_rows.extend(archive_rows(archive, name, "official_train"))
    with zipfile.ZipFile(ground_truth) as archive:
        for name in archive.namelist():
            if name.lower().endswith(".csv"):
                test_rows.extend(archive_rows(archive, name, "test"))
    original_counts = Counter(row["sign_label"] for row in train_rows if row["original_class_id"] in selected_classes)
    excluded_rows = [row for row in train_rows if row["group_id"] in excluded_tracks]
    train_rows = [row for row in train_rows if row["group_id"] not in excluded_tracks]
    prepared = []
    for category in args.class_ids:
        rows = [row for row in train_rows if row["original_class_id"] == category]
        split = split_tracks(rows, args.train_per_class, args.val_per_class, args.seed + category)
        prepared.extend(split["train"])
        prepared.extend(split["val"])
        prepared.extend(row for row in test_rows if row["original_class_id"] == category)
    with zipfile.ZipFile(train_archive) as training, zipfile.ZipFile(test_archive) as testing:
        for row in prepared:
            source = testing if row["split"] == "test" else training
            destination = args.output_dir / "raw" / row["image_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read(row["archive_member"]))
            with Image.open(destination) as image:
                width, height = image.size
                if not (0 <= row["xmin"] < row["xmax"] <= width and 0 <= row["ymin"] < row["ymax"] <= height):
                    raise ValueError(f"Invalid ROI for {destination}")
                image.verify()
    destination = args.output_dir / "processed/annotations_split.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(prepared)
    counts = {split: dict(Counter(row["sign_label"] for row in prepared if row["split"] == split)) for split in ["train", "val", "test"]}
    summary = {"source": "GTSRB official final training and test archives", "class_ids": {str(c): CLASS_NAMES[c] for c in args.class_ids}, "counts": counts,
               "official_available_train_counts": dict(original_counts),
               "eligible_train_counts": dict(Counter(row["sign_label"] for row in train_rows if row["original_class_id"] in selected_classes)),
               "excluded_training_tracks": sorted(excluded_tracks),
               "excluded_training_images": len(excluded_rows),
               "duplicate_report_sha256": hashlib.sha256(args.duplicate_report.read_bytes()).hexdigest(),
               "split_policy": "Training and validation use disjoint physical-sign tracks. All official test images for selected classes are held out.",
               "seed": args.seed, "roi_conversion": "Inclusive source ROI maxima converted to exclusive crop maxima.",
               "sampling": "Subset of real images from selected tracks; unused views and tracks are omitted. No generated images.",
               "group_counts": {split: len({row['group_id'] for row in prepared if row['split'] == split}) for split in ["train", "val"]}}
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
