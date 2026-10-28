# Real-Time Traffic Sign Detection and Classification

This project explores traffic sign detection and classification for driver-assistance applications. It detects selected traffic signs in road frames and classifies their crops using real Mapillary photographs.

The repository includes Hydra configuration, PyTorch Lightning training, TensorBoard logging, a DVC pipeline, a Docker environment, and trained detector/classifier ONNX exports. It is a fork of the required [lightning-hydra-template](https://github.com/Vladimir-125/lightning-hydra-template), retaining the template history and MIT notice.

The original project title is retained. **Real-time operation is a goal, not a verified result.** Performance measurements and limitations are documented in the technical report. This is not a driving-safety system.

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

1. **Detection stage:** a lightweight YOLO11n detector finds supported signs in full Mapillary scenes, using overlapping tiles to help locate small objects.
2. **Classification stage:** an optional ResNet-18 classifier refines labels using cropped sign regions. It changes accepted labels, not boxes, and cannot recover missed detections.

The detector already predicts categories. The default is detector-only; `refine=true` in the video configuration enables the second stage. Experiment design, model comparisons, metric selection and results are presented in the technical report.

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

Source scenes are split before creating classifier crops, so crops inherit their scene/group split. The technical report documents the selected subset, split counts, quality checks and evaluation limitations. No generated signs are used.

### Annotation and Small-Sign Concerns

The project includes a [source-label audit](report/mtsd_annotation_review/README.md) and tiled inference for small signs. The technical report explains the annotation concerns, checks performed, small-sign comparison and remaining limitations. Neither the audit nor tiling guarantees that every sign is annotated or detected.

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

Training requires authorised, prepared Mapillary data. Settings are stored in `configs/` and `pipelines/detection/`. The technical report describes the recorded training experiments and comparison procedure.

Standalone TensorBoard logs are in `logs/tensorboard/`; recorded experiment logs are under `outputs/`:

```bash
tensorboard --logdir outputs
```

## Evaluate

```bash
python3 -m src.eval ckpt_path=models/mtsd_classifier/classifier.ckpt
```

Evaluation commands are shown below. Metric definitions, selection rationale and results are in the technical report.

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

ONNX is an exported model format, not another architecture. Export validation and optimisation findings are documented in the technical report.

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

Left is input; right is actual ONNX output. The clip repeats real Mapillary photographs and includes missed detections. The [provenance record](demo/mtsd_gallery/provenance.json) identifies its source images, and the [CSV](demo/mtsd_gallery/processed/detections.csv) contains predictions. This is a still-image gallery, not a continuous-driving benchmark. Playback rate is not processing speed; measured performance is discussed in the technical report.

## Docker Environment

```bash
docker build -t traffic-sign-capstone .
docker run --rm traffic-sign-capstone
docker run --rm traffic-sign-capstone python -m pytest -q
```

The default command restores models and processes two actual frames. [Commit 449ef7a passed Docker build, container tests and ONNX smoke inference](https://github.com/varshneymannat-lab/traffic-sign-capstone-final/actions/runs/40690970167). Later revisions need their own CI result; full retraining inside Docker is not established. The final repository remains the required template fork. Mapillary images, pretrained weights and dependencies have separate licence terms; distributed notices and attribution are retained.
