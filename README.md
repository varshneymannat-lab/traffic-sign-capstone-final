# Traffic Sign Detection and Classification

A traffic-sign review project using **only the Mapillary Traffic Sign Dataset (MTSD)**: real road photographs for a lightweight YOLO11n detector and their annotated sign crops for ResNet-18, MobileNetV3-Small and EfficientNet-B0 comparisons.

## Goal

Locate and identify stop, no entry, yield, speed limit 30 and speed limit 50. Detection uses five exact MTSD g1 source labels listed in pipelines/detection/classes.yaml. Other source categories and unmerged international variants are outside this recorded task. Speed labels describe the displayed number and sign family, not an inferred national speed unit.

The application saves annotated images/video, coordinates, scores and elapsed processing time. It is an inspection experiment, not a driving controller or proven real-time road-safety system.

The practical problem is inspecting road-survey images for supported traffic signs. A local human-review workspace now lets a reviewer accept, correct or reject model predictions, add missed supported signs and export an audit record. Reviewers still inspect the full original image. A user-efficiency study has not established saved time or improved human accuracy.

## Results

| Model or experiment | Evaluation | Recorded result |
| --- | --- | --- |
| ResNet-18 | 425 annotated Mapillary test crops | Accuracy 96.94% (412/425), macro-F1 0.9687 |
| MobileNetV3-Small | Same Mapillary crop split | Accuracy 95.76%, macro-F1 0.9557 |
| EfficientNet-B0 | Same Mapillary crop split | Accuracy 96.47%, macro-F1 0.9633 |
| YOLO11n with tiled inference | 332 local held-out MTSD scenes, 425 boxes | mAP50 0.4839, mAP50:95 0.3344 |
| Exported ONNX detector | Same 332 scenes | mAP50 0.4839, mAP50:95 0.3344 |

**96.94% annotated-crop accuracy is not road-scene system accuracy.** Architecture, checkpoint, inference mode and threshold selections use internal validation. The local MTSD test uses selected official validation scenes, not the hidden official test. This fixed holdout was inspected in earlier project development, so it is not a new blind external evaluation.

At validation-selected confidence 0.343 and matching IoU 0.50, detector-only precision/recall/F1 are 0.5811 / 0.4635 / 0.5157. Optional **Mapillary-trained** crop refinement gives 0.6726 / 0.5365 / 0.5969 (TP 228, FP 111, FN 197). It cannot recover missing boxes. The predeclared default remains detector-only, rather than switching after viewing test results. Refined accuracy of 99.56% (228/229) on matched accepted boxes excludes missed objects and uses a selected subset; it is not the 425-crop or end-to-end score.

## Report and Demonstration

- [Five-page technical report](report/technical_report.pdf), [browser version](report/technical_report.html)
- [Revised Mapillary EDA](report/mtsd_eda/Mapillary_EDA_Report.pdf)
- [Annotation review](report/mtsd_annotation_review/README.md)
- [Actual ONNX-processed gallery](demo/mtsd_gallery/real_image_gallery.mp4)
- [Detector results](report/mtsd_results/evaluation.json), [ONNX results](report/mtsd_onnx_results/evaluation.json)
- [Mapillary classifier comparisons](report/mtsd_classifier_results/comparison.json)
- [Crop quality and dimensions](report/mtsd_classifier_results/crop_quality.json)
- [Requirement status](report/final_evidence.md)
- [Human-review workflow](report/review_workflow.md)
- [Credited reference-project comparison](report/reference_projects.md)

The 20-second clip repeats ten real held-out photographs. Left is input and right is actual detector output. **It is a still-image gallery, not continuous driving footage or a temporal benchmark.** Examples are selected by source category and key, not correctness. Playback FPS is not processing FPS. Slides and private speaking notes are outside this repository.

## Quick Start

Use Python 3.11 and run from the repository root:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-recorded.txt
python scripts/restore_inference_models.py
python -m src.detect_signs demo/mtsd_gallery/input.mp4 --output-dir outputs/my_demo --max-frames 2
~~~

The included artifacts/inference_models.zip restores selected ONNX models, the detector checkpoint and metadata after SHA256 checks. No training-data download is needed for this demo. Existing different models are not overwritten.

~~~bash
python -m src.detect_signs path/to/input.jpg --output-dir outputs/my_image
python -m src.detect_signs path/to/input.mp4 --output-dir outputs/my_video
python -m src.predict_image demo/mtsd_crops/stop.png
~~~

Use a new empty output directory for each run. Default detection uses overlapping tiles. The option --inference-mode whole is faster but less accurate on recorded validation data. Optional --refine adds the crop classifier. Detector confidence and classifier scores are separate CSV fields.

### Local Review Interface

```bash
python scripts/restore_inference_models.py
python -m src.review_app --port 8765
```

Open [Traffic Sign Review](http://127.0.0.1:8765). Select **Run detector** for the included real-image sample or open your own JPEG/PNG. Review predictions, correct categories or boxes, reject false positives, and draw missed supported signs. Save JSON to reopen a session and CSV for a detection-level audit. The JSON embeds the image and retains image-level review status; review changes do not modify training data or evaluation results. See the [workflow and limits](report/review_workflow.md). Images stay on your computer. This is a single-user localhost tool, not a public hosted service.

## Real Data

**One source dataset, two input forms.** MTSD supplies 2,706 selected real road scenes and 3,433 target boxes. Classifier crops are cut from those scenes, not drawn or pasted. Source scene/group splits are assigned before cropping and inherited unchanged. No exact crop hashes cross splits.

| Category | Training scenes | Internal validation | Local test |
| --- | ---: | ---: | ---: |
| Stop | 433 | 77 | 71 |
| No entry | 458 | 89 | 71 |
| Yield | 478 | 84 | 81 |
| Speed limit 30 | 436 | 65 | 70 |
| Speed limit 50 | 427 | 73 | 70 |
| Unique scenes | 2,018 | 356 | 332 |

Category counts overlap. There are over 400 retained training scenes/category, but not 500 each or necessarily independent physical signs. No copies or generated scenes inflate counts.

| Category | Training crops | Validation crops | Test crops |
| --- | ---: | ---: | ---: |
| Stop | 498 | 87 | 83 |
| No entry | 546 | 102 | 87 |
| Yield | 576 | 101 | 100 |
| Speed limit 30 | 478 | 73 | 79 |
| Speed limit 50 | 465 | 82 | 76 |
| Total annotated crops | 2,563 | 445 | 425 |

Crops retain parent-scene identity, category, coordinates and SHA256. They use prepared scenes (longest side at most 1,280 pixels) so low source resolution remains a limitation: 1,583 crops have a shortest side below 16 pixels. Resizing to 128 pixels adds no original detail. The earlier GTSRB study is retired from the active project; final classifier weights and scores come from new Mapillary training. ImageNet/general detector pretrained weights are initialization, not additional project traffic-sign datasets.

Official training supplies internal training/validation. Similarity groups stay together. Three training candidates overlapping held-out groups were removed. Hashes cannot prove physical-sign independence. Panoramas and scenes with targets flagged ambiguous, occluded or out-of-frame were excluded. Only target-positive scenes are sampled, so false-alarm rates on sign-free journeys are unmeasured.

### Annotation and Small-Sign Review

The revised EDA shows all original source boxes: yellow for selected targets, cyan for other categories. The nearby road-bump and oncoming-traffic-priority signs in the reviewed scene are source-annotated but outside the selected task. All 2,706 prepared scenes and 3,433 mapped target boxes match their source categories and coordinates.

That does not establish complete semantic coverage. Nineteen related regulatory stop/speed variants and one complementary speed sign remain outside the exact g1 mapping. Broader coverage needs reviewed remapping and retraining, not silent changes to test labels. The audit cannot discover every object omitted by source annotators.

The reviewed speed-50 boxes are approximately 57-59 original pixels across, 14 after preparation and 7 at a whole-scene 640 input. Larger page text fixes readability only. The measured validation comparison is:

| Inference using the same checkpoint | Val mAP50 | Val mAP50:95 | Mean read/detector ms |
| --- | ---: | ---: | ---: |
| Whole image | 0.3209 | 0.2198 | 51.5 |
| Overlapping tiles | 0.4859 | 0.3379 | 410.5 |

SAHI uses 640-pixel tiles, 20% overlap, a whole-scene pass and class-aware NMS. Tiling improves aggregate detection quality at a substantial speed cost. It cannot reconstruct lost pixels or guarantee detection of every tiny sign.

## Experiments and Reproduction

The submission checkout is a fork of [Vladimir-125/lightning-hydra-template](https://github.com/Vladimir-125/lightning-hydra-template), retaining its history and MIT notice. Classifiers use Lightning/Hydra. Detection uses the Ultralytics trainer with YAML settings.

Classifiers: ImageNet-pretrained, 128-pixel Mapillary crops, five epochs, batch 16, AdamW 0.0003, weight decay 0.0001, seed 42. All three train on Apple MPS; scoring and forward latency use CPU. Selection uses validation macro-F1: ResNet 0.9619, MobileNet 0.9463, EfficientNet 0.9477. Accuracy, macro precision/recall/F1, confusion matrices, parameter counts and CPU forward latency are compared. A majority-training-label baseline gets 23.53% accuracy and macro-F1 0.0762.

Detector: two initial five-epoch runs at 416/640, batch 8, ten frozen blocks, AdamW 0.001. A further 15-epoch run starts from the selected 640 checkpoint with five frozen blocks and learning rate 0.0005. Initial runs use CPU and additional training uses Apple M1 MPS. Their durations are not a controlled hardware-speed comparison. Whole/tiled mode and confidence are chosen using validation only.

~~~bash
python scripts/prepare_mtsd_crops.py
python scripts/compare_classifiers.py --data-dir data/mtsd_crops --output-dir outputs/mtsd_classifier_comparison --epochs 5 --image-size 128 --device auto
python scripts/export_selected.py
python scripts/build_classifier_results.py
~~~

MTSD requires authorised [Mapillary access](https://www.mapillary.com/dataset/trafficsign). Obtain the fully annotated v2 release. Expected layout: data/raw/mtsd/mtsd_v2_fully_annotated/ containing annotations/, splits/ and images/. The source folder has a DVC pointer. Its annotation ZIP MD5 and selected image-member CRC32/SHA256 were checked. Full image-archive MD5 verification is not claimed. Temporary account-specific download links must stay private.

~~~bash
python scripts/fetch_detector_weights.py
python scripts/compare_detectors.py
python scripts/finetune_detector.py --device auto
python scripts/select_operating_point.py
python scripts/export_detector.py
python scripts/evaluate_detector.py
python scripts/evaluate_detector.py --model models/mtsd/detector.onnx --output-dir report/mtsd_onnx_results --demo-dir demo/mtsd_onnx
~~~

These commands need prepared data and empty output destinations. Restore recorded DVC outputs rather than overwrite old experiments. Settings are in configs/, params.yaml and pipelines/detection/. To recreate preparation use the audited candidate CSV with scripts/prepare_mtsd.py; CLI help documents the required source and candidate paths.

~~~bash
dvc status
dvc dag
tensorboard --logdir outputs
pytest -q
node --test tests/review_logic.test.mjs
~~~

DVC records source data, splits, experiment outputs and models. **No shared DVC remote is configured.** A fresh clone can run the packaged inference demo, but training reproduction needs authorised data and either the original cache or regeneration. The command dvc pull cannot fetch unconfigured storage. A new training run may differ across hardware.

ONNX is an export format, not another architecture. Actual exported inference is evaluated on all local test scenes and 425 test crops. Classifier first-batch maximum absolute logit difference is 0.0000186. Dense-layer-only INT8 retains 96.94% crop accuracy, but mean forward times of 10.15/10.11 ms and sizes 44.708/44.705 MB show no convincing speed or compression gain. Convolutions stay float32; this is not full-detector quantisation or pruning.

## Docker

~~~bash
docker build -t traffic-sign-capstone .
docker run --rm traffic-sign-capstone
docker run --rm traffic-sign-capstone python -m pytest -q
~~~

The default command restores the included models and processes two actual demo frames. The [published GitHub Actions run](https://github.com/varshneymannat-lab/traffic-sign-capstone-final/actions/runs/40288335988) passed Docker build, container tests and ONNX smoke inference for commit 7e4a976. Docker is unavailable locally and full training was not repeated inside Docker. The review interface has separate local tests. For subsequent revisions, inspect that commit's own Actions result rather than attributing an older run to newer code. See report/container_verification.json for the historical verified commit and scope.

## Files

~~~text
configs/                 Hydra classifier settings
pipelines/detection/     Detector settings and exact label map
scripts/                 EDA, audits, preparation, training and export
src/detect_signs.py       Trained five-class image/video application
src/predict_image.py      Exported crop classifier
src/review_app.py         Local ONNX inference service for human review
web/                     Offline review interface and original icon licence
data/ models/ outputs/   DVC-managed data, models and training logs
artifacts/               Verified inference bundle for a fresh clone
report/                  EDA, reviews, results and technical report
demo/                    Real-data input/output and provenance
tests/                   Focused data, model, inference and smoke tests
~~~

## Sources and Terms

- [Mapillary Traffic Sign Dataset](https://www.mapillary.com/dataset/trafficsign), Ertler et al., ECCV 2020
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [SAHI](https://github.com/obss/sahi), Akyon et al., ICIP 2022
- [DFG traffic-sign inventory research](https://www.vicos.si/resources/dfg/)
- [Road Sign Recognition reference](https://github.com/ayushgayakwad/road-sign-recognition)
- [Traffic Sign Detection and Recognition reference](https://github.com/pedramebd/traffic-sign-detection-recognition)

The two reference repositories were reviewed at finalisation, not reproduced as experiments or used as sources of code, weights or project scores. See report/reference_projects.md for the task differences and comparison boundaries.

Dataset images, pretrained models and dependencies have separate licence terms from project code. Retain the distributed dataset notices and attribution. This project does not establish road-use reliability.
