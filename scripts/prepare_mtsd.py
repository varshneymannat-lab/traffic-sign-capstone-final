from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

PROJECT = Path(__file__).resolve().parents[1]


def difference_hash(image):
    pixels = np.asarray(image.convert('L').resize((9, 8)))
    value = 0
    for bit in (pixels[:, 1:] > pixels[:, :-1]).ravel():
        value = (value << 1) | int(bit)
    return value


def visual_groups(records, distance=4):
    parents = list(range(len(records)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for i, record in enumerate(records):
        for j in range(i):
            if bin(record['dhash'] ^ records[j]['dhash']).count('1') <= distance:
                parents[find(i)] = find(j)
    return [find(i) for i in range(len(records))]


def main():
    parser = argparse.ArgumentParser(description='Prepare five-class real MTSD detection data with image-group split checks.')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'data/mtsd')
    parser.add_argument('--minimum-train', type=int, default=300)
    parser.add_argument('--validation-fraction', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        parser.error('Validation fraction must be between zero and one.')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new output directory to preserve existing prepared data.')
    config = OmegaConf.load(PROJECT / 'pipelines/detection/classes.yaml')
    classes = list(config.classes)
    with args.candidates.open() as stream:
        rows = list(csv.DictReader(stream))
    grouped = {}
    for row in rows:
        grouped.setdefault(row['image_key'], []).append(row)
    source_manifest = json.loads((args.source / 'images/download_manifest.json').read_text())
    if source_manifest['missing_keys']:
        raise ValueError('Source download is incomplete.')
    records, excluded, pixel_hashes = [], [], {}
    for source_record in sorted(source_manifest['records'], key=lambda row: (grouped[row['image_key']][0]['split'] != 'val', row['image_key'])):
        key = source_record['image_key']
        source = args.source / 'images' / (key + '.jpg')
        if hashlib.sha256(source.read_bytes()).hexdigest() != source_record['sha256']:
            raise ValueError('Downloaded image SHA256 mismatch: ' + key)
        with Image.open(source) as image:
            image.load()
            rgb = image.convert('RGB')
            first = grouped[key][0]
            if rgb.size != (int(first['width']), int(first['height'])):
                raise ValueError('Image dimensions disagree with supplied annotation: ' + key)
            digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            dhash = difference_hash(rgb)
        if digest in pixel_hashes:
            excluded.append({'image_key': key, 'reason': 'exact_pixel_duplicate', 'matches': pixel_hashes[digest]})
            continue
        pixel_hashes[digest] = key
        records.append({'image_key': key, 'official_split': first['split'], 'dhash': dhash, 'pixel_sha256': digest})
    groups = visual_groups(records)
    for record, group in zip(records, groups):
        record['group_id'] = str(group)
    test_groups = {r['group_id'] for r in records if r['official_split'] == 'val'}
    retained = []
    for record in records:
        if record['official_split'] == 'train' and record['group_id'] in test_groups:
            excluded.append({'image_key': record['image_key'], 'reason': 'dhash_group_overlaps_official_validation'})
        else:
            retained.append(record)
    train_source = [r for r in retained if r['official_split'] == 'train']
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.validation_fraction, random_state=args.seed)
    train_indices, val_indices = next(splitter.split(train_source, groups=[r['group_id'] for r in train_source]))
    for indices, split in [(train_indices, 'train'), (val_indices, 'val')]:
        for index in indices:
            train_source[index]['split'] = split
    for record in retained:
        if record['official_split'] == 'val':
            record['split'] = 'test'
    counts = {s: Counter() for s in ['train', 'val', 'test']}
    image_counts = {s: Counter() for s in counts}
    for record in retained:
        categories = [r['sign_label'] for r in grouped[record['image_key']]]
        counts[record['split']].update(categories)
        image_counts[record['split']].update(set(categories))
    for split, minimum in [('train', args.minimum_train), ('val', 20), ('test', 20)]:
        for category in classes:
            if image_counts[split][category] < minimum:
                raise ValueError(f'Insufficient independent retained images: {split}/{category}={image_counts[split][category]}, minimum {minimum}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for record in retained:
        key, split = record['image_key'], record['split']
        image_dir = args.output_dir / 'images' / split
        label_dir = args.output_dir / 'labels' / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(args.source / 'images' / (key + '.jpg')) as source:
            image = source.convert('RGB')
            image.thumbnail((1280, 1280))
            image.save(image_dir / (key + '.jpg'), quality=95)
        lines = []
        for row in grouped[key]:
            width, height = int(row['width']), int(row['height'])
            x1, y1, x2, y2 = [float(row[k]) for k in ['xmin', 'ymin', 'xmax', 'ymax']]
            xywh = [(x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height]
            lines.append(str(classes.index(row['sign_label'])) + ' ' + ' '.join(f'{x:.9f}' for x in xywh))
            manifest.append({'image_path': f'images/{split}/{key}.jpg', 'image_key': key, 'split': split,
                             'official_split': record['official_split'], 'group_id': record['group_id'],
                             'sign_label': row['sign_label'], 'pixel_sha256': record['pixel_sha256'],
                             'x_center': xywh[0], 'y_center': xywh[1], 'box_width': xywh[2], 'box_height': xywh[3]})
        (label_dir / (key + '.txt')).write_text('\n'.join(lines) + '\n')
    with (args.output_dir / 'annotations_split.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    OmegaConf.save(OmegaConf.create({'path': '.', 'train': 'images/train', 'val': 'images/val', 'test': 'images/test', 'names': classes}), args.output_dir / 'dataset.yaml')
    summary = {'source': 'MTSD v2 fully annotated, selected real scenes', 'classes': classes,
               'seed': args.seed, 'minimum_retained_training_images_per_class': args.minimum_train,
               'box_counts': {s: dict(counts[s]) for s in counts},
               'image_counts_per_class': {s: dict(image_counts[s]) for s in counts},
               'split_images': dict(Counter(r['split'] for r in retained)), 'exclusions': excluded,
               'split_policy': 'Official train -> internal group train/validation. Selected official validation -> final local test, never tuning. Hidden official test is unused.',
               'group_policy': '64-bit full-scene difference-hash connected components at Hamming distance <=4; remove training components overlapping held-out scenes.',
               'limitations': 'Difference hashes screen visual similarity, not proof of independent physical-sign identity. No sequence metadata supplied. Only selected positive scenes, not representative sign-free drives.',
               'image_preprocessing': 'RGB; longest side <=1280 pixels; JPEG quality 95. Boxes normalized from original annotation coordinates.',
               'selection': 'At most 500 official-training and 70 official-validation image keys per class, sampled before inspecting model predictions; all five target boxes retained in each selected scene.'}
    (args.output_dir / 'dataset_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
