"""Freeze a detector choice using comparable validation records, before final testing."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def select(baseline, candidate):
    old = copy.deepcopy(next(row for row in baseline['experiments'] if row['name'] == baseline['selected_experiment']))
    new = copy.deepcopy(next(row for row in candidate['experiments'] if row['name'] == candidate['selected_experiment']))
    if baseline['dataset'] != candidate['dataset']:
        raise ValueError('Detector comparison requires the same prepared dataset and splits')
    old['name'] = 'baseline_640_finetuned'
    new['name'] = 'unfrozen_640_additional_10_epochs'
    chosen = new if new['validation']['mAP50_95'] > old['validation']['mAP50_95'] else old
    return {'selected_experiment': chosen['name'], 'selected_checkpoint': chosen['checkpoint'],
            'selected_image_size': chosen['image_size'],
            'selection_rule': 'Higher CPU validation mAP50:95; retain baseline on a tie. No new test scores select the model.',
            'experiments': [old, new], 'dataset': baseline['dataset'],
            'scope': 'Same YOLO11n architecture and protected data; additional training/unfreezing changes training budget, not a controlled architecture comparison.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('outputs/mtsd_detection/comparison.json'))
    parser.add_argument('--candidate', type=Path, default=Path('outputs/mtsd_detection_nov02/comparison.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/mtsd_revision_release'))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new output directory to preserve the recorded selection.')
    baseline=json.loads(args.baseline.read_text())
    candidate=json.loads(args.candidate.read_text())
    for record,path in [(baseline,args.baseline),(candidate,args.candidate)]:
        for row in record['experiments']:
            if not Path(row['checkpoint']).is_file():
                row['checkpoint']=str((path.parent/row['name']/'weights/best.pt').resolve())
    result = select(baseline,candidate)
    for row in result['experiments']:
        row['checkpoint_sha256'] = hashlib.sha256(Path(row['checkpoint']).read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'comparison.json').write_text(json.dumps(result, indent=2))
    print('Selected using validation only: '+result['selected_experiment'])


if __name__ == '__main__':
    main()
