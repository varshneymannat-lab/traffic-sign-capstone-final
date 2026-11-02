"""Link completed, verified revision artifacts without changing baseline records."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    result_path = root / 'report/mtsd_revision_nov02'
    destination = result_path / 'release_results.json'
    if destination.exists():
        raise ValueError('Release manifest already exists; preserve the recorded release.')
    load = lambda p: json.loads((root / p).read_text())
    crop = load('report/mtsd_revision_nov02/crop_resolution_comparison.json')
    direct = load('report/mtsd_revision_nov02/detection/evaluation.json')
    onnx = load('report/mtsd_revision_nov02/onnx/evaluation.json')
    if crop['selected_crop_source'] != 'original_resolution':
        raise ValueError('Original-source classifier was not selected by validation.')
    if direct['test_images'] != 332 or onnx['test_images'] != 332:
        raise ValueError('Incomplete held-out scene evaluation.')
    if abs(direct['detection']['mAP50'] - onnx['detection']['mAP50']) > .001:
        raise ValueError('Unexpected PyTorch/ONNX metric difference.')
    comparison=load('outputs/mtsd_revision_release/comparison.json')
    public_comparison={k:v for k,v in comparison.items() if k!='selected_checkpoint'}
    public_comparison['experiments']=[{k:v for k,v in row.items() if k!='checkpoint'} for row in comparison['experiments']]
    (result_path/'detector_validation.json').write_text(json.dumps(public_comparison,indent=2))
    (result_path/'inference_comparison.json').write_text(json.dumps(load('outputs/mtsd_revision_release/inference_comparison.json'),indent=2))
    files = ['models/mtsd_revision/detector.onnx', 'models/mtsd_classifier_original/classifier.onnx']
    record = {'dataset': 'Mapillary Traffic Sign Dataset v2 fully annotated',
              'detector_comparison': 'report/mtsd_revision_nov02/detector_validation.json',
              'inference_comparison': 'report/mtsd_revision_nov02/inference_comparison.json',
              'evaluation': 'report/mtsd_revision_nov02/detection/evaluation.json',
              'onnx_evaluation': 'report/mtsd_revision_nov02/onnx/evaluation.json',
              'detector_dir': 'models/mtsd_revision',
              'classifier_dir': 'models/mtsd_classifier_original',
              'crop_resolution_comparison': 'report/mtsd_revision_nov02/crop_resolution_comparison.json',
              'default_pipeline': 'detector only; crop refinement is optional',
              'refinement_crop_policy': 'Corresponding original photograph; boxes mapped from prepared scene coordinates.',
              'video_scope': 'Gallery inference uses the actual incoming video pixels, not an unavailable raw-photo side channel.',
              'selected_model_sha256': {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},
              'limitations': 'Previously inspected local holdout; selected positive scenes; no independent continuous-road or safety validation.'}
    destination.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
