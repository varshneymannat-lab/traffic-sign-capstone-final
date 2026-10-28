"""Publish measured Mapillary crop results, provenance and TensorBoard curves."""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    source = Path("outputs/mtsd_classifier_comparison")
    data = Path("data/mtsd_crops")
    output = Path("report/mtsd_classifier_results")
    figures = output / "figures"
    demo = Path("demo/mtsd_crops")
    figures.mkdir(parents=True, exist_ok=True)
    demo.mkdir(parents=True, exist_ok=True)
    comparison = json.loads((source / "comparison.json").read_text())
    classes = comparison["settings"]["classes"]
    chosen = next(e for e in comparison["experiments"] if e["architecture"] == comparison["selected_architecture"])
    shutil.copyfile(source / "comparison.json", output / "comparison.json")
    shutil.copyfile(source / "comparison.csv", output / "comparison.csv")
    shutil.copyfile(data / "dataset_summary.json", output / "dataset_summary.json")
    frame = pd.read_csv(data / "processed/annotations_split.csv")
    crossed = int((frame.groupby("sha256").split.nunique() > 1).sum())
    if crossed:
        raise ValueError("Exact crop hashes cross evaluation splits")
    quality = {"exact_crop_hashes_crossing_splits": crossed,
               "crops_shorter_side_below_16_pixels": int(((frame.xmax < 16) | (frame.ymax < 16)).sum()),
               "median_dimensions_by_split": frame.groupby("split")[["xmax", "ymax"]].median().to_dict(),
               "scope": "Exact file hashes and dimensions; does not establish physical-sign independence"}
    (output / "crop_quality.json").write_text(json.dumps(quality, indent=2))
    train = frame[frame.split == "train"]
    test = frame[frame.split == "test"].copy()
    dummy = DummyClassifier(strategy="most_frequent").fit([[0]] * len(train), train.sign_label)
    predicted = dummy.predict([[0]] * len(test))
    baseline = {"strategy": "Most frequent training label", "label": str(predicted[0]),
                "accuracy": accuracy_score(test.sign_label, predicted),
                "macro_f1": f1_score(test.sign_label, predicted, labels=classes, average="macro", zero_division=0)}
    (output / "baseline.json").write_text(json.dumps(baseline, indent=2))
    test["predicted_label"] = [p["predicted"] for p in chosen["test"]["predictions"]]
    test.to_csv(output / "selected_test_predictions.csv", index=False)
    records = []
    for category in classes:
        row = test[test.sign_label == category].iloc[0]
        shutil.copyfile(data / "raw" / row.image_path, demo / f"{category}.png")
        records.append({"file": f"{category}.png", "source_scene": row.source_image_key,
                        "source_crop": row.image_path, "split": "test", "actual": category,
                        "predicted": row.predicted_label,
                        "selection": "First saved test crop per class, not selected for correctness"})
    (demo / "provenance.json").write_text(json.dumps(records, indent=2))
    learning = {}
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for experiment in comparison["experiments"]:
        architecture = experiment["architecture"]
        shutil.copyfile(source / architecture / "test_confusion_matrix.png", figures / f"{architecture}_confusion.png")
        event = next((source / architecture / "logs").rglob("events.out*"))
        accumulator = EventAccumulator(str(event), size_guidance={"scalars": 0})
        accumulator.Reload()
        learning[architecture] = {}
        for axis, tag in zip(axes, ["val/f1", "val/loss"]):
            points = accumulator.Scalars(tag)
            learning[architecture][tag] = [p.value for p in points]
            axis.plot(range(1, len(points) + 1), [p.value for p in points], marker="o", label=architecture)
            axis.set(xlabel="Epoch", title=tag)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "learning_curves.png", dpi=150)
    plt.close(figure)
    (output / "learning_curves.json").write_text(json.dumps(learning, indent=2))
    correct = test[test.sign_label == test.predicted_label]
    wrong = test[test.sign_label != test.predicted_label]
    examples = pd.concat([correct.head(4), wrong.head(4)])
    figure, axes = plt.subplots(2, 4, figsize=(11, 5.5))
    for axis in axes.flat:
        axis.axis("off")
    for axis, row in zip(axes.flat, examples.itertuples()):
        with Image.open(data / "raw" / row.image_path) as image:
            axis.imshow(image, interpolation="nearest")
        axis.set_title(f"Actual: {row.sign_label}\nPredicted: {row.predicted_label}", fontsize=9)
    figure.suptitle("First four correct and first four incorrect Mapillary test crops, where available")
    figure.tight_layout()
    figure.savefig(figures / "test_examples.png", dpi=150)
    plt.close(figure)
    print(f"Published {len(test)} Mapillary test crop predictions; selected {chosen['architecture']}")


if __name__ == "__main__":
    main()
