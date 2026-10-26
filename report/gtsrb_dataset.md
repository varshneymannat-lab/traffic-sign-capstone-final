# GTSRB Dataset

## Source and Task

The German Traffic Sign Recognition Benchmark (GTSRB) contains real photographs of individual traffic signs. It is a classification benchmark with 43 classes and more than 50,000 images in total. Images have category labels and sign regions of interest.

- [Dataset authors and documentation](https://github.com/houbensebastian/GermanTrafficSignBenchmarks)
- [Official archive download page](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html)

GTSRB is not a full-road video detection benchmark. It does not contain a category for actual traffic-light objects. The revised project uses five-class image classification rather than the previous detector-plus-classifier task.

## Acquisition

From the repository root, run:

```bash
python scripts/download_gtsrb.py
```

This downloads official final training images, final test images, and test ground-truth labels into `data/raw/gtsrb/archives`. The script checks archive integrity and records download URLs, file sizes, SHA-256 hashes, and class counts in `download_manifest.json`.

## Five-Class Subset

| GTSRB Class ID | Category | Official Training Images |
| --- | --- | --- |
| 14 | Stop | 780 |
| 17 | No entry | 1,110 |
| 13 | Yield | 2,160 |
| 1 | Speed limit 30 km/h | 2,220 |
| 2 | Speed limit 50 km/h | 2,250 |

These counts were read from the CSV annotations in the downloaded official final training archive. They count images, not unique physical signs.

The following command prepares the subset used by the GTSRB configuration:

```bash
python scripts/prepare_gtsrb.py --class-ids 14 17 13 1 2
```

The prepared subset uses 500 training images and 150 validation images per category. It preserves all 2,820 official test images for the selected categories. All 30 images of stop-sign training track 00023 are excluded because eight of its images exactly match official test images. Actual counts and exclusions are recorded in `data/gtsrb/dataset_summary.json` and copied to the results report.

Training images include multiple views of the same physical sign. Training and validation are separated by sign track, not merely by image filename, to avoid closely related views entering both sets. A count of 500 images does not mean 500 different physical signs.

## Evaluation Status

Existing synthetic-data results, checkpoints, ONNX files, and the video demonstration must not be presented as GTSRB results. The revised comparison saves real-data experiments to `outputs/gtsrb_comparison`, exports the validation-selected classifier to `models/gtsrb`, and generates the updated technical report from those recorded results.

## Citation

J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. *The German Traffic Sign Recognition Benchmark: A multi-class classification competition.* IEEE International Joint Conference on Neural Networks, pp. 1453-1460, 2011.
