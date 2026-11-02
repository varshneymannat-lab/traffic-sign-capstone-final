"""Record the fixed ResNet-18 crop-source ablation after validation selection."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.compare_classifiers import evaluate, save_confusion
from src.data.sign_datamodule import TrafficSignDataModule
from src.utils.video_pipeline import load_classifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if (args.output_dir / 'crop_resolution_comparison.json').exists():
        parser.error('This evaluation is already recorded; use another output directory.')
    baseline = json.loads((PROJECT / 'report/mtsd_classifier_results/comparison.json').read_text())
    revised = json.loads((args.comparison_dir / 'comparison.json').read_text())
    old = next(row for row in baseline['experiments'] if row['architecture'] == 'resnet18')
    new = next(row for row in revised['experiments'] if row['architecture'] == 'resnet18')
    identity = ['image_path', 'sign_label', 'split', 'group_id', 'source_image_key']
    before = pd.read_csv(PROJECT / 'data/mtsd_crops/processed/annotations_split.csv')
    after = pd.read_csv(args.data_dir / 'processed/annotations_split.csv')
    if not before[identity].equals(after[identity]):
        raise ValueError('The ablation must retain the same crop identities and source splits.')
    if (after.groupby('sha256').split.nunique() > 1).any():
        raise ValueError('Exact crop hashes cross splits.')
    settings = revised['settings']
    for key in ['epochs', 'batch_size', 'training_seed', 'image_size', 'learning_rate', 'weight_decay', 'pretrained']:
        if baseline['settings'][key] != settings[key]:
            raise ValueError('Different training setting: '+key)
    torch.set_num_threads(2)
    classes = settings['classes']
    checkpoint = args.comparison_dir / 'resnet18/models/checkpoints/best.ckpt'
    model = load_classifier(checkpoint, classes, 'cpu')
    data = TrafficSignDataModule(str(args.data_dir.resolve()), str((args.data_dir / 'processed/annotations_split.csv').resolve()), settings['image_size'], 16, 0, classes)
    data.setup()
    test = evaluate(model, data.test_dataloader(), classes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_confusion(test, classes, args.output_dir / 'original_crop_test_confusion.png', 'ResNet-18: original-resolution source crops')
    selected = 'original_resolution' if new['validation']['macro_f1'] > old['validation']['macro_f1'] else 'prepared_1280'
    result = {'experiment': 'ResNet-18 source-crop resolution ablation',
              'selection_rule': 'Higher validation macro-F1; retain baseline on a tie. Test results do not choose the crop source.',
              'selected_crop_source': selected,
              'prepared_1280': {'validation': old['validation'], 'test': old['test']},
              'original_resolution': {'validation': new['validation'], 'test': test},
              'training_settings': settings,
              'split_identity_verified': True,
              'median_crop_width': {'prepared': float(before.xmax.median()), 'original': float(after.xmax.median())},
              'limitations': ['Single seed and short fixed training budget.', 'Both inputs are derived from the same Mapillary dataset.',
                              'Changing crop source also avoids prepared-scene JPEG re-encoding and changes pixel-bound rounding; not an isolated test of pixel count alone.',
                              'Supplied annotated-crop scores do not measure end-to-end detection.',
                              'Held-out scenes were inspected in prior development; not a blind external test.']}
    (args.output_dir / 'crop_resolution_comparison.json').write_text(json.dumps(result, indent=2))
    records = after[after.split == 'test'].copy()
    records['prepared_crop_prediction'] = [row['predicted'] for row in old['test']['predictions']]
    records['original_crop_prediction'] = [row['predicted'] for row in test['predictions']]
    records.to_csv(args.output_dir / 'paired_crop_predictions.csv', index=False)
    print(json.dumps({'selected_crop_source': selected, 'validation_macro_f1': new['validation']['macro_f1'],
                      'test_samples': test['samples'], 'test_accuracy': test['accuracy'], 'test_macro_f1': test['macro_f1']}, indent=2))


if __name__ == '__main__':
    main()
