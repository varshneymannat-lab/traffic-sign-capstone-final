# Final Evidence and Outstanding Checks

Checked 28 October 2026. This distinguishes local evidence from publication and external verification. The older scope-proposal checklist is superseded by this record. Model experiments retain their recorded results; the review interface is a separately tested application addition.

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
| Docker | Published commit 7e4a976 passed build, container tests and two-frame ONNX inference in run 40288335988. For subsequent revisions, check their own commit-specific CI evidence. Full retraining was not repeated in Docker. |
| Report | Five-page technical report with EDA, comparisons, mentor concerns, results and limitations. |
| Presentation | PowerPoint and private speaking guide outside Git. |
| GitHub | The original completed release, commit 7e4a976, is published in the required fork. Any later local changes are not assumed uploaded until a successful push and remote verification. |
| Practical workflow | Local ONNX review interface: accept/correct/reject, box edits, manual additions, JSON save/reopen and CSV export. Original predictions remain separate. Functional desktop/mobile checks include actual inference. |
| Reference projects | Two supplied repositories reviewed and credited in reference_projects.md. No imported code, weights or benchmark results; datasets/tasks differ. |
| Operational benefit | Review workflow implemented, but no user study establishes time savings or improved human accuracy. |

Submission checks: verify the final revision is published and inspect its own CI outcome; configure shared DVC storage if required by the reviewer. Clarify with the mentor if a labelled continuous-road video or a strict 500 training scenes/category is required. Current evidence establishes neither. Functional completion of the scoped application does not guarantee acceptance of every possible interpretation of the example brief.
