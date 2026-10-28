# Real-Time Traffic Sign Detection and Classification

This project explores traffic sign detection and classification for driver-assistance applications. It detects selected traffic signs in road frames and classifies their crops using real Mapillary photographs.

The repository includes Hydra configuration, PyTorch Lightning training, TensorBoard logging, a DVC pipeline, a Docker environment, and trained detector/classifier ONNX exports. It is a fork of the required [lightning-hydra-template](https://github.com/Vladimir-125/lightning-hydra-template), retaining the template history and MIT notice.

The original project title is retained. **Real-time operation is a goal, not a verified result:** warm tiled detector-plus-classifier processing averages 435.35 ms per scene on the development CPU. This is not a driving-safety system.

## Why I Selected This Project

I selected traffic sign detection because it is practical, focused, and connected to road safety. Finding small signs within a complex road photograph is a real computer-vision problem.

Compared with a more complex vehicle type/color project, this scope is easier to explain:

- Each supported sign has one clear category label.
- One real dataset supplies both full scenes and annotated sign crops.
- The project demonstrates detection, classification, evaluation, and deployment.
- Lightweight models and overlapping tiles allow a measurable accuracy-speed comparison.

## Project Goal

Build an end-to-end system that can:

1. Detect supported traffic signs in road frames.
2. Crop accepted sign regions when crop refinement is enabled.
3. Classify the crop as one of:
   - `stop`
   - `no_entry`
   - `yield`
   - `speed_limit_30`
   - `speed_limit_50`
4. Write annotated output video with bounding boxes and labels, plus CSV predictions and measured processing time.

These are the five trained Mapillary categories. The exact g1 source labels are recorded in [pipelines/detection/classes.yaml](pipelines/detection/classes.yaml); other categories and unmerged international variants remain outside the task. Speed names describe the displayed number, not an inferred national unit.

## Method

The project keeps the original detection-and-classification design:

1. **Detection stage:** a lightweight YOLO11n detector is fine-tuned on five-category Mapillary boxes. Whole-image and overlapping-tile inference are compared on the same validation scenes. This replaces the earlier generic COCO class filter.
2. **Classification stage:** ResNet-18, MobileNetV3-Small and EfficientNet-B0 are fine-tuned on real crops from the same Mapillary scene split. Validation macro-F1 selects ResNet-18. Optional crop refinement changes accepted labels, not boxes, and cannot recover misses.

The detector already predicts categories. The predeclared default is detector-only; `refine=true` in the video configuration enables the second stage. Both variants are evaluated rather than assuming an extra model always helps.

| Classifier | Validation macro-F1 | Accuracy on 425 test crops | Test macro-F1 |
| --- | ---: | ---: | ---: |
| ResNet-18 | 0.9619 | 96.94% (412/425) | 0.9687 |
| MobileNetV3-Small | 0.9463 | 95.76% | 0.9557 |
| EfficientNet-B0 | 0.9477 | 96.47% | 0.9633 |

A majority-training-label baseline scores 23.53% accuracy. Crop accuracy does not measure whether a sign is found. The detector scores mAP50 0.4839 and mAP50:95 0.3344 on 332 local test scenes containing 425 targets; actual ONNX results round to the same figures.

At validation-selected confidence 0.343343 and matching IoU 0.50, detector-only precision/recall/F1 are 0.5811/0.4635/0.5157 (TP 197, FP 142, FN 228). Optional Mapillary crop refinement gives 0.6726/0.5365/0.5969 (TP 228, FP 111, FN 197). Its 99.56% accuracy on 229 matched accepted boxes excludes missed signs; it is not the 425-crop or end-to-end score.

The supplied repositories are [credited references](report/reference_projects.md), not sources of imported code, weights or project scores: [road-sign-recognition](https://github.com/ayushgayakwad/road-sign-recognition) and [traffic-sign-detection-recognition](https://github.com/pedramebd/traffic-sign-detection-recognition). Their datasets and categories differ, so headline scores are not directly comparable.

## Repository Structure

```text
configs/                 Hydra experiment configs
data/                    DVC-tracked Mapillary scenes and crops
models/                  DVC-tracked model artifacts
src/
  data/                  Dataset and Lightning DataModule
  models/                Traffic sign classifier
  utils/                 Detection and video annotation utilities
  train.py               Train classifier
  eval.py                Evaluate classifier
  infer_video.py         Run video inference
  export_onnx.py         Export classifier to ONNX
scripts/                 Real-data preparation, EDA, experiments and export
tests/                   Data, model and inference tests
report/                  Technical report, EDA and experimental results
demo/                    Real-data input/output and provenance
pipelines/detection/     Detector settings and exact category map
artifacts/               Verified inference bundle for a fresh clone
dvc.yaml                 Reproducible pipeline stages
Dockerfile               Reproducible environment
```

## Dataset Format

The project uses **only Mapillary Traffic Sign Dataset v2 fully annotated** for traffic-sign training and evaluation. Obtain authorised access from [Mapillary](https://www.mapillary.com/dataset/trafficsign). Source layout: `data/raw/mtsd/mtsd_v2_fully_annotated/` containing `annotations/`, `splits/` and `images/`. Dataset terms apply separately from source-code licensing; temporary account-specific download links are not committed.

Scene annotations are in `data/mtsd/annotations_split.csv`. Crop annotations are in `data/mtsd_crops/processed/annotations_split.csv`, including these actual rows (additional provenance columns omitted):

```csv
image_path,xmin,ymin,xmax,ymax,sign_label
images/test/-2k5fP6nR3RmS7FfzwPNxg_00000.png,0,0,10,30,stop
images/test/-AcmoBAQMsDGwzmG9tzAJw_00001.png,0,0,26,22,speed_limit_30
```

Crop paths are relative to `data/mtsd_crops/raw/`. Classifier boxes use pixel coordinates with exclusive maxima; detector boxes use normalized scene coordinates. Supported labels are available through [configs/data/traffic_signs.yaml](configs/data/traffic_signs.yaml), inheriting the Mapillary crop configuration.

### EDA and Data Splits

The [Mapillary EDA](report/mtsd_eda/Mapillary_EDA_Report.pdf) covers real scenes, class counts, image quality, rectangles, similarity checks and the annotation/small-sign concerns.

| Category | Train scenes | Val scenes | Test scenes | Train crops | Val crops | Test crops |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stop | 433 | 77 | 71 | 498 | 87 | 83 |
| No entry | 458 | 89 | 71 | 546 | 102 | 87 |
| Yield | 478 | 84 | 81 | 576 | 101 | 100 |
| Speed 30 | 436 | 65 | 70 | 478 | 73 | 79 |
| Speed 50 | 427 | 73 | 70 | 465 | 82 | 76 |
| Unique scenes / total crops | 2,018 | 356 | 332 | 2,563 | 445 | 425 |

There are 2,706 scenes and 3,433 targets overall. Scene counts overlap because one photograph can contain several categories. Counts exceed 200 training scenes/category, not 500 for every category or necessarily independent physical signs. Crops are not additional collected photographs. No generated signs or copies inflate counts.

Official training scenes supply internal training/validation; selected official validation scenes supply the local test, not the hidden official test. Crops inherit their source scene/group split. Three training candidates overlapping held-out similarity groups were removed; no exact crop hashes cross splits. Hashes do not prove physical-sign independence. This fixed holdout was inspected in earlier development, so it is not a new blind external evaluation.

Only selected positive nonpanoramic scenes are used; selected ambiguous, occluded or out-of-frame targets were excluded. False alarms on sign-free journeys remain unmeasured. Source annotation ZIP MD5 and selected image-member CRC32/SHA256 were checked, not full image-archive MD5.

### Annotation and Small-Sign Concerns

The [source-label audit](report/mtsd_annotation_review/README.md) checks all scenes and 3,433 retained targets against source categories/coordinates, finding no mapped-target mismatches. Yellow shows selected targets; cyan shows other source categories. The nearby road-bump and priority signs identified during review are source-annotated but outside this task. This does not prove source annotation completeness. Nineteen related regulatory variants and one complementary speed sign remain outside the exact g1 map; broader coverage requires reviewed remapping and retraining.

The reviewed speed-50 boxes shrink from about 57-59 original pixels to 14 after preparation and 7 at whole-scene 640 input. Across all crop splits, 1,583 crops have a shortest side below 16 pixels. Enlarging them cannot reconstruct lost detail.

| Same detector and validation scenes | mAP50 | mAP50:95 | Mean read/detector ms |
| --- | ---: | ---: | ---: |
| Whole image | 0.3209 | 0.2198 | 51.5 |
| Overlapping tiles | 0.4859 | 0.3379 | 410.5 |

SAHI uses 640-pixel tiles, 20% overlap, a whole-image pass and class-aware NMS. Tiling improves aggregate validation quality at about eight times the processing cost; it does not solve every tiny sign. Larger labels improve report readability only.

## Quick Demo

The repository includes trained exports and a real-image clip, so a short demonstration needs no training-data download:

```bash
python3 scripts/restore_inference_models.py
python3 -m src.infer_video input_video=demo/mtsd_gallery/input.mp4 output_video=outputs/mapillary_demo.mp4 max_frames=2
```

Use a new output name each run. The bundle restores the selected detector and classifier after SHA256 checks and refuses to overwrite different models. The clip repeats real photographs; it is not synthetic imagery or continuous moving-road video.

## Technical Report

- [Read the technical report](report/technical_report.md).
- [Open or download the five-page PDF](report/technical_report.pdf).
- [View the browser version](report/technical_report.html).
- [Classifier comparisons](report/mtsd_classifier_results/comparison.json), [detector results](report/mtsd_results/evaluation.json) and [actual ONNX results](report/mtsd_onnx_results/evaluation.json).

The report includes introduction, methodology, EDA, experiments, metric selection, results, discussion and limitations. PPT and private speaking notes are delivered separately, not stored in GitHub.

## Setup

Use Python 3.11 from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-recorded.txt
```

`requirements.txt` remains available; `requirements-recorded.txt` pins the recorded direct dependencies.

## Train

```bash
python3 -m src.train
```

For the recorded comparison, use prepared Mapillary scenes/crops and an empty experiment destination:

```bash
python3 scripts/prepare_mtsd_crops.py
python3 scripts/compare_classifiers.py --data-dir data/mtsd_crops --output-dir outputs/mtsd_classifier_comparison --epochs 5 --image-size 128 --device auto
```

All three classifiers use ImageNet-pretrained weights, 128-pixel crops, five epochs, batch 16, AdamW, learning rate 0.0003, weight decay 0.0001 and seed 42. Training uses MPS on the development Mac; scoring and latency use CPU. The comparison selects checkpoints/architecture by validation macro-F1. Standalone training retains validation-loss checkpointing unless overridden by the comparison command.

Standalone TensorBoard logs are in `logs/tensorboard/`; recorded experiment logs are under `outputs/`:

```bash
tensorboard --logdir outputs
```

Detector settings are in `pipelines/detection/`: initial five-epoch 416/640 runs and a further 15-epoch 640 fine-tune. Initial runs use CPU and the extra run MPS; durations are not a controlled hardware-speed comparison. Pretrained weights are initialization, not additional project traffic-sign datasets.

## Evaluate

```bash
python3 -m src.eval ckpt_path=models/mtsd_classifier/classifier.ckpt
```

Accuracy, macro precision/recall/F1 and confusion matrices evaluate supplied crops. Detection uses mAP50, mAP50:95 and fixed-threshold precision/recall/F1. Validation selects checkpoints/settings, not the final test. CPU latency and model size are separate measurements.

```bash
python3 scripts/evaluate_detector.py
python3 scripts/evaluate_detector.py --model models/mtsd/detector.onnx --output-dir report/mtsd_onnx_results --demo-dir demo/mtsd_onnx
```

Evaluation needs authorised prepared data and recorded models. Use new result destinations rather than overwrite existing evaluations.

## Export ONNX

```bash
python3 -m src.export_onnx ckpt_path=models/mtsd_classifier/classifier.ckpt output_path=outputs/new_export/traffic_sign_classifier.onnx
```

The recorded selected-model export and verification use:

```bash
python3 scripts/export_selected.py
python3 scripts/export_detector.py
```

ONNX is a format, not another architecture. Actual exported inference was evaluated on all 425 crops and 332 scenes. Classifier float32 and dense-layer-only INT8 retain 96.94% crop accuracy; means of 10.15/10.11 ms and sizes 44.708/44.705 MB do not establish meaningful acceleration/compression. Convolutions remain float32; this is not full-detector quantisation or pruning.

## Video Inference

```bash
python3 -m src.infer_video \
  input_video=demo/mtsd_gallery/input.mp4 \
  output_video=outputs/annotated_sample.mp4
```

The original Hydra video entry point delegates to the tested Mapillary pipeline. `configs/infer.yaml` controls the trained detector, confidence, mode, frame limit and optional refinement. Use `refine=true` for the Mapillary ONNX classifier. Annotated video is saved at `output_video`; CSV and timing are in its sibling folder with the same stem. Existing outputs are protected.

The lower-level command also accepts still images:

```bash
python3 -m src.detect_signs path/to/input.jpg --output-dir outputs/my_image
python3 -m src.predict_image demo/mtsd_crops/stop.png
```

## DVC Pipeline

```bash
dvc repro
```

The pipeline keeps the original preparation, training and export responsibilities, expanded for real data and the required comparisons:

- Source audit, scene preparation and EDA check the licensed real annotations.
- `prepare_crops` derives classifier crops without changing scene splits.
- `compare` trains the classifiers; `train_detectors` runs detector comparisons.
- `export` and `export_detector` save trained models.
- Evaluation, inference-bundle, real-image-gallery and report stages retain reproducible outputs.

DVC tracks source/prepared data, experiments and models locally. **No shared DVC remote is configured.** A fresh clone supports packaged inference; full retraining needs authorised Mapillary data plus cache access or regeneration. `dvc pull` cannot fetch unconfigured storage. Training can differ across hardware.

```bash
dvc status
dvc dag
pytest -q
```

## Mapillary Video Demonstration

[Download the 20-second real-image demonstration video](demo/mtsd_gallery/real_image_gallery.mp4).

The [provenance record](demo/mtsd_gallery/provenance.json) identifies ten real local-test photographs repeated across 200 frames. Left is input; right is actual ONNX output. Examples are selected by source category/key, not correctness, and include misses. The [CSV](demo/mtsd_gallery/processed/detections.csv) has 139 frame-detection rows, not unique physical signs. This is a still-image gallery, not a labelled continuous-driving benchmark. Playback rate is not processing speed; real-time moving-road performance remains unverified.

## Docker Environment

```bash
docker build -t traffic-sign-capstone .
docker run --rm traffic-sign-capstone
docker run --rm traffic-sign-capstone python -m pytest -q
```

The default command restores models and processes two actual frames. [Commit 449ef7a passed Docker build, container tests and ONNX smoke inference](https://github.com/varshneymannat-lab/traffic-sign-capstone-final/actions/runs/40690970167). Later revisions need their own CI result; full retraining inside Docker is not established. The final repository remains the required template fork. Mapillary images, pretrained weights and dependencies have separate licence terms; distributed notices and attribution are retained.
