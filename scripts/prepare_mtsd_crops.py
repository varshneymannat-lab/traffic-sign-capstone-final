"""Create classifier crops without changing the protected Mapillary scene split."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
from PIL import Image


def crop_bounds(row, width, height):
    left = max(0, math.floor((row.x_center - row.box_width / 2) * width))
    top = max(0, math.floor((row.y_center - row.box_height / 2) * height))
    right = min(width, math.ceil((row.x_center + row.box_width / 2) * width))
    bottom = min(height, math.ceil((row.y_center + row.box_height / 2) * height))
    if right <= left or bottom <= top:
        raise ValueError("Empty annotation crop")
    return left, top, right, bottom


def prepare(source, destination):
    annotations = source / "annotations_split.csv"
    scenes = pd.read_csv(annotations)
    if scenes.groupby("image_key").split.nunique().max() != 1:
        raise ValueError("A source photograph crosses splits")
    if scenes.groupby("group_id").split.nunique().max() != 1:
        raise ValueError("A protected scene group crosses splits")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Use an empty destination to preserve existing datasets")
    rows = []
    for index, row in enumerate(scenes.itertuples(index=False)):
        with Image.open(source / row.image_path) as scene:
            scene = scene.convert("RGB")
            bounds = crop_bounds(row, *scene.size)
            crop = scene.crop(bounds)
        relative = Path("images") / row.split / f"{row.image_key}_{index:05d}.png"
        target = destination / "raw" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        crop.save(target)
        rows.append({"image_path": relative.as_posix(), "sign_label": row.sign_label,
                     "split": row.split, "group_id": row.group_id,
                     "source_image_key": row.image_key, "source_image_path": row.image_path,
                     "official_split": row.official_split,
                     "xmin": 0, "ymin": 0, "xmax": crop.width, "ymax": crop.height,
                     "source_crop_xyxy": json.dumps(bounds),
                     "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    frame = pd.DataFrame(rows)
    processed = destination / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    frame.to_csv(processed / "annotations_split.csv", index=False)
    summary = {"dataset": "Mapillary Traffic Sign Dataset v2 fully annotated",
               "source_annotations_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
               "source_scenes": int(scenes.image_key.nunique()), "crops": len(frame),
               "split_counts": frame.groupby("split").size().to_dict(),
               "class_counts": frame.groupby(["split", "sign_label"]).size().unstack(fill_value=0).to_dict(),
               "split_policy": "Inherit existing protected scene/group split; no independent crop split",
               "crop_policy": "Ground-truth rectangles from prepared, aspect-preserved 1280px-longest-side scenes; floor left/top, ceil right/bottom, clamp to image; lossless PNG crops",
               "limitations": ["Small source signs remain low-resolution after resizing for the classifier.",
                               "Conditional crop accuracy is not end-to-end detection accuracy.",
                               "This fixed holdout has been inspected during earlier project development; it is not an external blind evaluation."]}
    (destination / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/mtsd"))
    parser.add_argument("--output", type=Path, default=Path("data/mtsd_crops"))
    args = parser.parse_args()
    prepare(args.source, args.output)
