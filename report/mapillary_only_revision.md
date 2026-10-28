# Mapillary-Only Experiment Revision

28 October 2026. The final project uses MTSD v2 fully annotated for both tasks.

## Retrained and Remeasured

- Real classifier crops were extracted from the existing protected scene split: 2,563 training, 445 validation and 425 test crops. Parent-scene/group identity is retained; no independent crop random split was made. No exact crop hashes cross splits.
- ResNet-18, MobileNetV3-Small and EfficientNet-B0 were each fine-tuned for five epochs on these crops using the same settings and Apple MPS. Scoring and forward latency use CPU. Validation macro-F1 selects ResNet-18, not its test result.
- The newly selected classifier was exported to ONNX and checked on all 425 test crops. Float32 and dense-only INT8 predictions were evaluated separately.
- Detector-plus-classifier evaluation was rerun on all 332 local test scenes for the PyTorch and ONNX detector backends, using the new Mapillary classifier. Refined labels, counts, previews and pipeline timings were regenerated.
- The inference bundle now contains the new Mapillary classifier. The report, configuration, DVC stages and private presentation materials use the new results. Old GTSRB experiment files were moved outside the active submission; they remain recoverable from earlier Git history.

## Retained Artifacts

The five-class YOLO11n detector, prepared scenes, protected splits, source-label audit and detector-only gallery already used Mapillary. Their trained weights and recorded validation comparisons are retained, not presented as newly trained runs. Detector ONNX SHA256 remains `333ed784e2d480dabe55e95defb5ecbf9f15e6019fa09a29b1338263a3161df4`.

The change to `scripts/evaluate_detector.py` adds the Mapillary classifier default, split option and evaluation provenance. The matching functions imported by detector validation are unchanged; this does not require retraining the detector. The default gallery path remains detector-only; only the optional classifier path in the application changed. DVC records those dependency updates against the retained outputs rather than claiming a new detector-training or gallery-generation run.

## Evaluation Boundaries

The fixed local holdout was inspected in earlier development. It is protected from gradient updates, and architecture/checkpoint selection uses internal validation, but it is not an external blind benchmark. Training and inference crops can differ in box placement. Small crops remain low-resolution; human review does not change the reported model scores. The predeclared application default remains detector-only, so the new test refinement comparison does not silently select a deployment policy.

ImageNet and general detector pretraining provide initial weights. No additional traffic-sign dataset is used for project training/evaluation. Descriptions of GTSRB/GTSDB in related work explain the external reference projects; they are not active project data.
