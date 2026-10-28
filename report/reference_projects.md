# Reference Projects

Reviewed 28 October 2026 during final project review, after the recorded model experiments. These are implementation references, not reproduced benchmark baselines. No code, weights or reported results from either repository were imported into this project.

| Reference | What was inspected | Relevant idea | Boundary |
| --- | --- | --- | --- |
| [Ayush Gayakwad: Road Sign Recognition](https://github.com/ayushgayakwad/road-sign-recognition) | README and src/main.py | Camera-frame inference and visible bounding-box labels | Its README describes a 43-category YOLOv5/GTSRB system. The inspected script does not establish an equivalent full-scene evaluation protocol. |
| [Pedram: Traffic Sign Detection and Recognition](https://github.com/pedramebd/traffic-sign-detection-recognition) | README, demo.py and src/pipeline/detect_and_classify.py | Separate scene detection and crop recognition, with distinct scores | It describes GTSDB detection and 43-category GTSRB classification. Its dataset, class counts and training budgets differ from this project's five-category MTSD study. |

The final implementation retains its own YOLO11n weights, splits, model comparisons and ONNX checks. The optional crop-refinement experiment tests a similar two-stage idea without assuming that a second classifier always helps. The added local review workspace makes human decisions explicit and retains original predictions separately.

The repositories' headline percentages must not be ranked against this project's results as though they used the same test. A direct comparison would require shared data, category mapping, splits, matching rules and evaluation code. Their performance claims have not been independently reproduced here.

The project still needs an independent reviewer-efficiency study before claiming saved time or improved human accuracy. Software workflow checks demonstrate that review actions work; they do not establish operational benefit.

## Inspected Source Links

- https://github.com/ayushgayakwad/road-sign-recognition/blob/main/src/main.py
- https://github.com/pedramebd/traffic-sign-detection-recognition/blob/main/demo.py
- https://github.com/pedramebd/traffic-sign-detection-recognition/blob/main/src/pipeline/detect_and_classify.py

Dataset, dependency and repository licence notices remain separate. The local interface includes Lucide icons with the original licence in web/LUCIDE_LICENSE.txt.
