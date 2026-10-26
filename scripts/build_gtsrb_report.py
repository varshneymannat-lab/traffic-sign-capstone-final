from __future__ import annotations

import argparse
import html
import json
import shutil
import os
from pathlib import Path

import markdown
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as PDFImage
from sklearn.metrics import accuracy_score, f1_score
from sklearn.dummy import DummyClassifier
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


NAMES = {"resnet18": "ResNet-18", "mobilenet_v3_small": "MobileNetV3-Small", "efficientnet_b0": "EfficientNet-B0"}


def main():
    parser = argparse.ArgumentParser(description="Build the technical report from recorded GTSRB experiments, without inventing scores.")
    parser.add_argument("--comparison-dir", type=Path, default=Path("outputs/gtsrb_comparison"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/gtsrb"))
    parser.add_argument("--document-dir", type=Path, default=Path("report/classification_study"))
    args = parser.parse_args()
    report_dir = Path("report")
    result_dir = report_dir / "gtsrb_results"
    document_dir=args.document_dir
    document_dir.mkdir(parents=True,exist_ok=True)
    relative_results=os.path.relpath(result_dir,document_dir)
    relative_eda=os.path.relpath(report_dir/'gtsrb_eda',document_dir)
    figures = result_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    comparison = json.loads((args.comparison_dir / "comparison.json").read_text())
    dataset = json.loads((args.data_dir / "dataset_summary.json").read_text())
    onnx = json.loads((result_dir / "onnx_verification.json").read_text())
    experiments = comparison["experiments"]
    chosen = next(e for e in experiments if e["architecture"] == comparison["selected_architecture"])
    settings = comparison["settings"]
    classes = list(dataset["counts"]["train"])
    comparison["settings"]["classes"] = classes
    (result_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))
    shutil.copyfile(args.comparison_dir / "comparison.csv", result_dir / "comparison.csv")
    for experiment in experiments:
        architecture = experiment["architecture"]
        shutil.copyfile(args.comparison_dir / architecture / "test_confusion_matrix.png", figures / f"{architecture}_confusion.png")
    annotations = pd.read_csv(args.data_dir / "processed/annotations_split.csv")
    testing = annotations[annotations.split == "test"].reset_index(drop=True)
    actual = testing.sign_label.tolist()
    training = annotations[annotations.split == "train"]
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(training), 1)), training.sign_label)
    baseline_predictions = dummy.predict(np.zeros((len(testing), 1)))
    majority = str(baseline_predictions[0])
    baseline = {"label": majority, "strategy": "Most frequent training label; ties resolved by sklearn label ordering",
                "accuracy": accuracy_score(actual, baseline_predictions),
                "macro_f1": f1_score(actual, baseline_predictions, labels=classes, average="macro", zero_division=0)}
    (result_dir / "baseline.json").write_text(json.dumps(baseline, indent=2))
    assert len(chosen["test"]["predictions"]) == len(testing)
    predictions = testing.copy()
    predictions["predicted_label"] = [row["predicted"] for row in chosen["test"]["predictions"]]
    predictions.to_csv(result_dir / "selected_test_predictions.csv", index=False)
    demo_dir = Path("demo/gtsrb")
    demo_dir.mkdir(parents=True, exist_ok=True)
    demo_records = []
    for category in classes:
        row = predictions[predictions.sign_label == category].iloc[0]
        with Image.open(args.data_dir / "raw" / row.image_path) as image:
            image.convert("RGB").crop((row.xmin, row.ymin, row.xmax, row.ymax)).save(demo_dir / f"{category}.png")
        demo_records.append({"demo_file": f"{category}.png", "original_image": row.image_path,
                             "split": "official_test", "actual_label": category,
                             "predicted_label": row.predicted_label,
                             "selection": "First image in saved test order for this actual category; not selected for correctness."})
    (demo_dir / "provenance.json").write_text(json.dumps(demo_records, indent=2))
    correct = predictions[predictions.sign_label == predictions.predicted_label]
    wrong = predictions[predictions.sign_label != predictions.predicted_label]
    selected_rows = pd.concat([correct.head(4), wrong.head(4)])
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
    for axis in axes.flat:
        axis.axis("off")
    for axis, (_, row) in zip(axes.flat, selected_rows.iterrows()):
        with Image.open(args.data_dir / "raw" / row.image_path) as image:
            crop = image.convert("RGB").crop((row.xmin, row.ymin, row.xmax, row.ymax))
            axis.imshow(crop, interpolation="nearest")
        axis.set_title(f"Actual: {row.sign_label.replace('_', ' ')}\nPredicted: {row.predicted_label.replace('_', ' ')}", fontsize=9)
    fig.suptitle("Held-out test examples: first four correct and first four errors (where available)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(figures / "test_examples.png", dpi=140)
    plt.close(fig)

    learning = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for experiment in experiments:
        architecture = experiment["architecture"]
        events = next((args.comparison_dir / architecture / "logs").rglob("events.out*"))
        accumulator = EventAccumulator(str(events), size_guidance={"scalars": 0})
        accumulator.Reload()
        learning[architecture] = {}
        for axis, tag in zip(axes, ["val/f1", "val/loss"]):
            points = accumulator.Scalars(tag)
            learning[architecture][tag] = [p.value for p in points]
            axis.plot(range(1, len(points) + 1), [p.value for p in points], marker="o", label=NAMES[architecture])
            axis.set(xlabel="Epoch", title="Validation macro-F1" if tag.endswith("f1") else "Validation cross-entropy loss")
            axis.set_xticks(range(1, settings["epochs"] + 1))
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "learning_curves.png", dpi=150)
    plt.close(fig)
    (result_dir / "learning_curves.json").write_text(json.dumps(learning, indent=2))
    (result_dir / "dataset_summary.json").write_text(json.dumps(dataset, indent=2))

    table = "\n".join(f"| {NAMES[e['architecture']]} | {e['validation']['macro_f1']:.4f} | {100*e['test']['accuracy']:.2f}% | {e['test']['macro_precision']:.4f} | {e['test']['macro_recall']:.4f} | {e['test']['macro_f1']:.4f} |" for e in experiments)
    efficiency = "\n".join(f"| {NAMES[e['architecture']]} | {e['parameter_count']/1e6:.3f} | {e['cpu_latency_ms']:.2f} | {e['training_seconds']/60:.2f} |" for e in experiments)
    per_class = "\n".join(f"| {c.replace('_', ' ')} | {chosen['test']['per_class'][c]['precision']:.4f} | {chosen['test']['per_class'][c]['recall']:.4f} | {chosen['test']['per_class'][c]['f1-score']:.4f} | {chosen['test']['per_class'][c]['support']:.0f} |" for c in classes)
    data_table = "\n".join(f"| {c.replace('_',' ')} | 500 | 150 | {dataset['counts']['test'][c]} |" for c in classes)
    quant_table = "\n".join(f"| {name} | {entry['bytes']/1e6:.2f} | {entry['accuracy']*100:.2f}% | {entry['macro_f1']:.4f} | {entry['forward_latency']['mean_ms']:.2f} |" for name, entry in onnx["sessions"].items())
    selected_name = NAMES[chosen["architecture"]]
    errors = len(wrong)
    report = f"""# Traffic Sign Classification Using Real Images

## Abstract
This project classifies real traffic-sign photographs into five categories using GTSRB. It compares pretrained ResNet-18, MobileNetV3-Small and EfficientNet-B0 under the same five-epoch budget. Training uses 2,500 images, validation uses 750 images and testing uses 2,820 images. Related training views are separated by sign track. One training track containing exact copies of test images is excluded. {selected_name} is selected by validation macro-F1, not test accuracy. Its test accuracy is {chosen['test']['accuracy']*100:.2f}% and macro-F1 is {chosen['test']['macro_f1']:.4f}. An exported ONNX model is checked using actual inference. These results concern a five-class cropped-sign task, not full-scene detection or autonomous driving.

## 1. Introduction
Traffic signs give drivers instructions. Recognising a sign category is a useful computer-vision task with a clear input and output. The project was narrowed from a detector-plus-classifier demonstration to image classification so that the work could focus on real data, a fair model comparison and understandable evaluation. YOLO is not used in the new experiments; the old synthetic video is retained only as historical work. The research question is: which of three pretrained classifiers performs best under a small, shared training budget, and what are the costs in model size and inference time?

## 2. Dataset and Exploratory Analysis
GTSRB is a public benchmark of real German traffic-sign photographs [1]. The downloaded final archives contain 39,209 training images and 12,630 test images across 43 classes. All 51,839 annotated images decoded successfully. No mismatches between image sizes and annotation sizes or invalid inclusive ROIs were found. Training categories contain 210 to 2,250 images, so the full dataset is imbalanced. Median training width and height are both 43 pixels. Small images and variable brightness make this more challenging than generated drawings.

The project uses Stop, No entry, Yield, Speed limit 30 and Speed limit 50. Actual traffic lights are not a GTSRB category, so the original category list has changed. Pixel hashing found eight exact image pairs shared by training and testing, all in stop-sign track 00023. All 30 training images in that track were excluded. This does not establish that every remaining image is independent: near-duplicate views remain within tracks.

| Category | Training | Validation | Test |
| --- | ---: | ---: | ---: |
{data_table}
| Total | 2,500 | 750 | 2,820 |

Sampling uses seed 42. Training contains 85 sign tracks and validation contains 25 different tracks. Tracks are assigned before selecting the image counts. Unused views and tracks are omitted. The official test examples for the five categories are all retained, and do not update weights or choose checkpoints. Test counts and exact duplicates were inspected during EDA; test predictions were not used to tune the models.

## 3. Methodology
Annotated rectangles provide the sign crops; no detector is required. ROI maximum coordinates are converted from inclusive image coordinates to exclusive crop coordinates. Crops are resized to {settings['image_size']} by {settings['image_size']} pixels and normalised using ImageNet mean and standard deviation. This input size limits CPU computation relative to 224-pixel inputs; different resolutions were not compared. Training adds colour jitter and rotations up to eight degrees. Validation and testing use fixed preprocessing. Horizontal flips are excluded because they can reverse digits or sign meanings. Upsampling a small image does not restore lost detail.

Each classifier starts with torchvision ImageNet-pretrained weights [2]. Its final output layer is replaced by a five-score layer. All parameters are fine-tuned, rather than freezing the feature extractor. ResNet-18 provides a familiar baseline; MobileNetV3-Small tests a smaller architecture; EfficientNet-B0 provides a further compact alternative. The same AdamW learning rate (0.0003), weight decay (0.0001), batch size (16), five epochs and seed (42) are used for each model. Cross-entropy measures training error. The highest validation macro-F1 checkpoint is retained. Shared settings provide a controlled comparison, but are not guaranteed to be optimal for every architecture.

## 4. Metrics and Experiments
Accuracy is the fraction of correct predictions. Precision describes how often a predicted category is correct; recall describes how many actual members of that category are found. F1 combines precision and recall: F1 = 2PR/(P+R), using zero when the denominator is zero. Macro-F1 averages the five class F1 scores with equal weight [3]. It is the primary selection metric because the test categories have unequal sizes. A confusion matrix shows which categories are confused: rows are actual categories and columns are predictions. These are classification metrics; detector mAP is not applicable to this revised task.

There are three neural-network training experiments, one per architecture, and one seed per experiment. The model is selected by highest validation macro-F1, with lower measured CPU latency breaking an exact tie. Test results are presented for all three fixed experiments, but are not the selection rule. A simple most-frequent-training-label baseline always predicts {majority.replace('_',' ')}: accuracy {baseline['accuracy']*100:.2f}% and macro-F1 {baseline['macro_f1']:.4f}. The balanced training set has a tie between all five labels, resolved deterministically by scikit-learn's label ordering. No test labels choose the baseline's prediction. This constant baseline does not examine image pixels.

| Model | Validation Macro-F1 | Test Accuracy | Macro Precision | Macro Recall | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

![Validation curves]({relative_results}/figures/learning_curves.png)

## 5. Results and Error Analysis
The validation-selected model is **{selected_name}**. It makes {len(testing)-errors:,} correct predictions and {errors:,} errors on {len(testing):,} held-out test crops. The table below gives class-level results, so a high overall score cannot hide a weaker category.

| Category | Precision | Recall | F1 | Test Images |
| --- | ---: | ---: | ---: | ---: |
{per_class}

![Selected classifier confusion matrix]({relative_results}/figures/{chosen['architecture']}_confusion.png)

![Test examples]({relative_results}/figures/test_examples.png)

Examples are the first four correct predictions and first four errors in the saved test ordering, when available. This is an error-inspection display, not a representative random sample. Every selected-model test prediction and source image path is saved in selected_test_predictions.csv. A photograph can be correctly classified while another small, dark or visually similar sign is misclassified. The confusion matrix and actual images should be used together; visual difficulty alone does not prove the cause of a particular error.

## 6. Efficiency and ONNX
| Model | Parameters (Millions) | PyTorch Forward (ms/image) | Training Time (minutes) |
| --- | ---: | ---: | ---: |
{efficiency}

Training time includes subprocess startup and checkpoint writing. PyTorch latency uses one image, ten warm-ups and 30 repeated forward passes on CPU with two threads. It excludes image reading, cropping and normalisation. Hardware: {settings['hardware']['platform']}. These measurements do not establish real-road video speed. Parameter count is not the same as training-checkpoint file size, which can include optimizer state. Smaller parameter counts do not guarantee lower measured CPU latency: different operators and their hardware implementations can change the ranking.

ONNX is a model format, not an additional classifier. The selected model was exported, checked structurally and run using ONNX Runtime. On the first 32 test images, maximum absolute PyTorch-versus-ONNX logit difference was {onnx['first_test_batch_max_abs_logit_difference']:.8f}, within the stated numerical tolerance. Both exported versions were then evaluated on all 2,820 test crops.

| ONNX Variant | File Size (MB) | Test Accuracy | Macro-F1 | Mean Forward (ms/image) |
| --- | ---: | ---: | ---: | ---: |
{quant_table}

The INT8 experiment dynamically quantises dense MatMul/Gemm weights only; convolution layers remain float32. It is not full-model INT8. ONNX timings use CPU, two threads, ten warm-ups and 50 repetitions, excluding preprocessing. These measurements should not be assumed equivalent to PyTorch timing. The float32 model remains the default, avoiding a deployment decision based only on file size or the test set.

## 7. Reproducibility and Difficulties
Hydra holds the settings, Lightning organises training and TensorBoard stores curves. DVC records the downloaded archives, prepared split, comparison runs and model exports. Source archives have URL and SHA-256 provenance; the split CSV hash is stored with each experiment. The DVC pipeline replaces the earlier synthetic-data pipeline. A Dockerfile is supplied, but Docker is not installed on the measured machine, so a successful container build is not claimed. DVC storage is local unless a remote is configured; a GitHub clone alone does not contain the large dataset or model files. The project uses Lightning/Hydra organisation; a GitHub fork relationship with the originally recommended template has not been verified.

The practical difficulties were replacing generated data, finding suitable categories, handling small images, avoiding related views across splits, removing train/test duplicates, and comparing architectures under limited CPU resources. The CSV ROI headers also required matching the official field names. None of these is solved by simply downloading more images. Data checks and recorded settings are part of the experiment.

## 8. Discussion and Conclusion
The project now has real-image training, three measured classifier experiments, class-level evaluation and a verified ONNX inference path. It is limited to five known German sign classes and has no unknown-category rejection. Predictions on full-road scenes, other countries, weather conditions or unseen camera systems are not established. One seed and five epochs do not establish a universal architecture ranking. Further work could repeat seeds, expand classes, measure a representative external test set and optimise training budgets. Full-scene detection would be a separate extension with detector-specific data and mAP evaluation.

## References
[1] Stallkamp et al., The German Traffic Sign Recognition Benchmark: A multi-class classification competition, IJCNN, 2011. https://github.com/houbensebastian/GermanTrafficSignBenchmarks

[2] PyTorch torchvision model documentation and pretrained weights: https://docs.pytorch.org/vision/stable/models.html

[3] Scikit-learn classification metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
"""
    draft = report
    def section(title):
        return draft.split(f"## {title}\n", 1)[1].split("\n## ", 1)[0].strip()
    conclusion = f"The project compares three pretrained classifiers using real GTSRB photographs, a fixed track-separated development split and held-out testing. {selected_name} is selected using validation macro-F1 and achieves {chosen['test']['accuracy']*100:.2f}% test accuracy and {chosen['test']['macro_f1']:.4f} macro-F1. The main result is supported by per-class metrics, saved predictions and verified ONNX inference. The scope remains five-class cropped-sign classification; external testing and repeated runs are needed before making broader claims."
    report = "\n\n".join([
        "# Traffic Sign Classification Using Real Images",
        "## Abstract\n\n" + section("Abstract"),
        "## 1. Introduction and Objective\n\n" + section("1. Introduction"),
        "## 2. Methodology\n\n### 2.1 Data, annotations and exploratory data analysis (EDA)\n\n" + section("2. Dataset and Exploratory Analysis") + f"\n\nThe complete EDA, including the class-distribution and image-property charts and real-image gallery, is available in [the EDA report]({relative_eda}/README.md).",
        "### 2.2 Preprocessing and classification\n\n" + section("3. Methodology"),
        "### 2.3 Evaluation metrics and experimental design\n\n" + section("4. Metrics and Experiments").split("\n\n| Model", 1)[0],
        "## 3. Experiments and Results\n\n### 3.1 Classifier comparison\n\n| Model | Validation Macro-F1 | Test Accuracy | Macro Precision | Macro Recall | Macro-F1 |\n| --- | ---: | ---: | ---: | ---: | ---: |\n" + table + f"\n\n![Validation curves]({relative_results}/figures/learning_curves.png)",
        "### 3.2 Class-level results and error analysis\n\n" + section("5. Results and Error Analysis"),
        "### 3.3 Timing, export and verification\n\n" + section("6. Efficiency and ONNX"),
        "## 4. Discussion and Reproducibility\n\n" + section("7. Reproducibility and Difficulties") + "\n\n" + section("8. Discussion and Conclusion"),
        "## 5. Conclusion\n\n" + conclusion,
        "## Evidence Sources\n\nThe results are drawn from `report/gtsrb_results/comparison.json`, `dataset_summary.json`, `learning_curves.json`, `selected_test_predictions.csv`, and `onnx_verification.json`. The full EDA is recorded in `report/gtsrb_eda/`.",
        "## References\n\n" + section("References"),
    ]) + "\n"
    (document_dir / "technical_report.md").write_text(report)
    style = "body{font:16px/1.5 Arial,sans-serif;max-width:950px;margin:40px auto;padding:0 24px;color:#111}h1{color:#203c35}img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:14px}td,th{padding:7px;border-bottom:1px solid #bbb}th{background:#f1f1f1}h2{margin-top:28px}a{color:#203c35}"
    (document_dir / "technical_report.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Traffic Sign Classification</title><style>' + style + "</style>" + markdown.markdown(report, extensions=["tables"]) + "</html>")

    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 10.5
    styles["BodyText"].leading = 14
    styles["BodyText"].spaceAfter = 7
    styles["Title"].fontSize = 22
    styles["Title"].leading = 27
    styles["Title"].alignment = 0
    styles["Title"].textColor = colors.HexColor("#203c35")
    styles["Heading2"].fontSize = 14
    styles["Heading2"].leading = 18
    styles["Heading3"].fontSize = 11.5
    styles["Heading3"].leading = 15
    styles["Heading3"].textColor = colors.black
    styles["Normal"].fontSize = 9
    styles["Normal"].leading = 12
    story = []
    def paragraph(text, style="BodyText"):
        story.append(Paragraph(html.escape(text), styles[style]))
    def heading(text):
        paragraph(text, "Heading2")
    def subheading(text):
        paragraph(text, "Heading3")
    def table_pdf(values, widths=None):
        wrapped = [[Paragraph(html.escape(str(cell)), styles["Normal"]) for cell in line] for line in values]
        item = Table(wrapped, colWidths=widths, repeatRows=1, hAlign="LEFT")
        item.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f1f1")), ("LINEBELOW", (0, 0), (-1, -1), .35, colors.HexColor("#bbbbbb")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        story.extend([item, Spacer(1, 8)])
    def picture(path, width, height):
        story.extend([PDFImage(str(path), width=width, height=height, kind="proportional"), Spacer(1, 6)])
    paragraph("Traffic Sign Classification Using Real Images", "Title")
    heading("Abstract")
    paragraph(section("Abstract"))
    heading("1. Introduction and Objective")
    paragraph(section("1. Introduction"))
    paragraph("The objective is to compare recognition quality, model size and inference time using the same real-image split. The five output categories are Stop, No entry, Yield, Speed limit 30 and Speed limit 50. A model receives one annotated sign crop and returns five category scores. The highest score supplies its prediction; it does not locate the sign in a full road scene.")
    paragraph("The revised task builds on image preprocessing, transfer learning and architecture comparison. ResNet-18 is the baseline, MobileNetV3-Small provides a smaller alternative, and EfficientNet-B0 adds another compact architecture. Training, validation and test roles remain separate. Reported results are measured on real benchmark photographs, not the previous generated images.")
    story.append(PageBreak())
    heading("2. Methodology")
    subheading("2.1 Data, annotations and exploratory data analysis (EDA)")
    paragraph("GTSRB contains 51,839 real images across 43 classes: 39,209 official training images and 12,630 official test images. EDA checked every annotated image: all decoded successfully, with no image-size mismatches or invalid inclusive ROIs. Class sizes range from 210 to 2,250 training images. Median training width and height are both 43 pixels, and brightness varies.")
    paragraph("Eight exact train/test image pairs were found in stop-sign track 00023. All 30 training images in that track were removed before sampling. Training and validation use different physical-sign tracks, with 85 training tracks and 25 validation tracks. Seed 42 controls sampling; all official test examples of the five selected categories are retained.")
    table_pdf([["Category", "Training", "Validation", "Test"]] + [[c.replace("_", " "), 500, 150, dataset["counts"]["test"][c]] for c in classes] + [["Total", 2500, 750, 2820]], [210, 80, 80, 80])
    paragraph("The separate EDA report in report/gtsrb_eda/ contains the full class-count chart, image-size and brightness distributions, real training-image examples, and duplicate file lists.", "Normal")
    subheading("2.2 Preprocessing and classification")
    paragraph("Annotated rectangles supply the sign crops. Inclusive ROI maxima are converted to exclusive crop maxima. Images are resized to 128 by 128 pixels and normalised using ImageNet statistics. Training adds colour jitter and rotations up to eight degrees; validation and testing use fixed preprocessing. Horizontal flips are excluded because they can reverse numbers or sign meanings.")
    paragraph("All three classifiers start with ImageNet-pretrained weights and a new five-class output layer. All parameters are fine-tuned for five epochs using cross-entropy and AdamW: learning rate 0.0003, weight decay 0.0001, batch size 16 and seed 42. The same split and preprocessing are used for each model. The 128-pixel input limits CPU computation; other resolutions were not compared.")
    subheading("2.3 Evaluation metrics and experimental design")
    paragraph("Accuracy measures the fraction correct. Precision measures how often a predicted category is correct; recall measures how many actual members are found. F1 combines them: 2PR/(P+R). Macro-F1 gives each class equal weight, making it useful with unequal test counts. Checkpoints and the final architecture are selected by validation macro-F1; lower CPU latency breaks an exact tie. Test predictions do not select the model.")
    story.append(PageBreak())
    heading("3. Experiments and Results")
    subheading("3.1 Classifier comparison")
    paragraph(f"There are three neural-network experiments, one per architecture and one seed each. A constant training-label baseline predicts no entry after a tie between the balanced training labels; it achieves {baseline['accuracy']*100:.2f}% test accuracy and {baseline['macro_f1']:.4f} macro-F1 without examining pixels.")
    table_pdf([["Model", "Val F1", "Test Acc", "Macro P", "Macro R", "Macro F1"]] + [[NAMES[e["architecture"]], f"{e['validation']['macro_f1']:.4f}", f"{100*e['test']['accuracy']:.2f}%", f"{e['test']['macro_precision']:.4f}", f"{e['test']['macro_recall']:.4f}", f"{e['test']['macro_f1']:.4f}"] for e in experiments], [126, 60, 66, 66, 66, 66])
    paragraph(f"Selected by validation: {selected_name}. Test crop predictions: {len(testing)-errors:,} correct and {errors:,} incorrect. The selection is not based on which test accuracy is largest.")
    picture(figures / "learning_curves.png", 450, 145)
    subheading("3.2 Class-level results and error analysis")
    table_pdf([["Category", "Precision", "Recall", "F1", "Count"]] + [[c.replace("_", " "), f"{chosen['test']['per_class'][c]['precision']:.4f}", f"{chosen['test']['per_class'][c]['recall']:.4f}", f"{chosen['test']['per_class'][c]['f1-score']:.4f}", dataset["counts"]["test"][c]] for c in classes], [170, 70, 70, 70, 70])
    paragraph("The selected model's three errors are between the two speed-limit categories: two 30 signs are predicted as 50, and one 50 sign is predicted as 30. Stop, No entry and Yield have no errors in this test subset. These are crop-classification results, not evidence of full-road detection accuracy.")
    story.append(PageBreak())
    subheading("3.2 Class-level results and error analysis (continued)")
    picture(figures / f"{chosen['architecture']}_confusion.png", 270, 232)
    paragraph("Rows show actual categories; columns show predictions. Diagonal entries are correct. Off-diagonal entries are errors. Counts are not percentages, so larger classes contribute more entries.")
    picture(figures / "test_examples.png", 410, 205)
    paragraph("The gallery contains the first four correct predictions and the three errors in saved test order. These are held-out real photographs, not a representative random sample. All predictions and image paths are saved in the results CSV. Visual difficulty alone does not prove why a model made an error.")
    subheading("3.3 Timing")
    table_pdf([["Model", "Parameters (M)", "Forward (ms)", "Train (min)"]] + [[NAMES[e["architecture"]], f"{e['parameter_count']/1e6:.3f}", f"{e['cpu_latency_ms']:.2f}", f"{e['training_seconds']/60:.2f}"] for e in experiments], [180, 90, 90, 90])
    paragraph("PyTorch timing uses CPU, two threads, one image, ten warm-ups and 30 repetitions. It excludes reading and preprocessing; training time includes startup and checkpoints. Smaller parameter counts did not guarantee faster CPU inference. This is not a road-video real-time benchmark.")
    story.append(PageBreak())
    subheading("3.4 Export, optimisation and verification")
    paragraph("ONNX is an export format, not another trained model. The selected classifier was exported and checked. Actual ONNX Runtime inference was evaluated on all 2,820 test crops. The first 32 images were also compared numerically with PyTorch; maximum absolute logit difference: " + f"{onnx['first_test_batch_max_abs_logit_difference']:.8f}.")
    table_pdf([["Variant", "Size (MB)", "Accuracy", "Macro-F1", "Mean ms"]] + [[name, f"{entry['bytes']/1e6:.2f}", f"{entry['accuracy']*100:.2f}%", f"{entry['macro_f1']:.4f}", f"{entry['forward_latency']['mean_ms']:.2f}"] for name, entry in onnx["sessions"].items()], [110, 85, 85, 85, 85])
    paragraph("Dynamic INT8 applies only to dense MatMul/Gemm weights; convolution layers stay float32. This is not full-model INT8. ONNX CPU timings use two threads, ten warm-ups and 50 repetitions, excluding preprocessing. Float32 remains the default; optimisation is reported as an experiment, not an automatic improvement.")
    heading("4. Discussion and Reproducibility")
    paragraph("Hydra stores configuration, Lightning organises training, TensorBoard logs curves and DVC records dataset/model versions. Archive URL and SHA-256 provenance and the fixed split hash are saved. Data and weights are local unless a DVC remote is configured. The Dockerfile has not been built here because Docker is not installed. A fork relationship to the recommended Lightning-Hydra template has not been verified.")
    paragraph("Practical issues included finding appropriate real categories, matching official CSV headers, identifying train/test duplicates, separating related sign views and managing CPU training time. All 30 images in the duplicated stop-sign training track were removed before sampling. Original synthetic scores are not used as real-data evidence.")
    paragraph("These results cover five known German sign categories, with no unknown-category rejection or full-scene detector. One seed and five epochs do not prove a universal architecture ranking. Repeated runs and external real-world images are needed before broader claims.")
    heading("5. Conclusion")
    paragraph(conclusion)
    subheading("Evidence Sources and References")
    paragraph("Measured evidence: report/gtsrb_results/ contains comparison.json, dataset_summary.json, learning_curves.json, selected_test_predictions.csv and onnx_verification.json. Full EDA: report/gtsrb_eda/.", "Normal")
    paragraph("[1] Stallkamp et al., GTSRB, IJCNN 2011. github.com/houbensebastian/GermanTrafficSignBenchmarks. [2] PyTorch torchvision model documentation: docs.pytorch.org/vision/stable/models.html. [3] Scikit-learn classification metrics: scikit-learn.org/stable/modules/model_evaluation.html.")
    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#53605e"))
        canvas.drawString(46, 28, "Traffic Sign Classification Using Real Images")
        canvas.drawRightString(A4[0] - 46, 28, str(doc.page))
    SimpleDocTemplate(str(document_dir / "technical_report.pdf"), pagesize=A4, rightMargin=52, leftMargin=52, topMargin=52, bottomMargin=44, title="Traffic Sign Classification Using Real Images", author="Mannat Varshney").build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Updated report for {selected_name}: {chosen['test']['accuracy']*100:.2f}% accuracy, {chosen['test']['macro_f1']:.4f} macro-F1")


if __name__ == "__main__":
    main()
