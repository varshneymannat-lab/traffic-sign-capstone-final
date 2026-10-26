# Final Evidence and Outstanding Checks

Checked 26 October 2026. This distinguishes local evidence from publication and external verification. The older scope-proposal checklist is superseded by this record.

| Requirement | Evidence and limits |
| --- | --- |
| Real data and EDA | Real GTSRB crops and 2,706 prepared MTSD scenes. Revised EDA includes full scenes, counts, source checks and mentor review cases. |
| Five categories | Five exact MTSD g1 labels. Related international variants are not all covered; the annotation audit lists them. |
| Annotation checking | 2,706 scenes and 3,433 mapped boxes match source labels and coordinates. Source annotation completeness remains unproven. |
| Examples per category | 427-478 training scenes/category. Exceeds 200, not 500 for every category or proven independent physical signs. |
| Detection experiments | Fine-tuned YOLO11n: 416/640 initial runs, additional 640 fine-tune, whole/tiled inference comparison. |
| Small signs | Whole/tiled validation mAP50:95 is 0.2198 / 0.3379, with mean read/detector time 51.5 / 410.5 ms. Numbered callouts fix report readability only. |
| Classification | Three pretrained models on the same GTSRB split/budget. Selection by validation macro-F1. |
| Metrics | Accuracy, macro precision/recall/F1, confusion matrices, AP50, AP50:95, fixed-threshold counts and timing. Crop and detection scores are separate. |
| Held-out evaluation | 332 MTSD local test scenes, 425 boxes. PyTorch and actual ONNX detector mAP50 both round to 0.4839. Hidden official test unused. |
| Deployment | Structure/raw-output checks and actual ONNX inference over all test scenes. Selected model bundle includes SHA256 checks. |
| Demo | 200 actually processed frames from ten repeated real stills. Not continuous driving video or proof of real-time processing. |
| DVC | Source subset pointer, pipeline/lock and local cache. No shared DVC remote; training reproduction needs authorised artifact access. |
| TensorBoard | Classifier and detector event logs retained in experiment outputs. |
| Required template | Final local submission checkout is a fork of Vladimir-125/lightning-hydra-template. Classifier loop uses Lightning/Hydra; detection uses Ultralytics. |
| Docker | Recipe and automated build/test/two-frame inference job. Local container execution unverified because Docker is unavailable. |
| Report | Five-page technical report with EDA, comparisons, mentor concerns, results and limitations. |
| Presentation | PowerPoint and private speaking guide outside Git. |
| GitHub | Fork exists. A local revision is not assumed uploaded until a successful push and remote files are verified. |

Remaining external steps: publish the prepared revision and check the actual CI outcome; configure shared DVC storage if required by the reviewer. Clarify with the mentor if a labelled continuous-road video or a strict 500 training scenes/category is required. Current evidence establishes neither.
