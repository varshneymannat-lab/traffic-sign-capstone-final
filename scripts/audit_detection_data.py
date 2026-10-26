from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / 'pipelines/detection/classes.yaml'


def valid_box(box, width, height):
    values = [float(box[k]) for k in ['xmin', 'ymin', 'xmax', 'ymax']]
    x1, y1, x2, y2 = values
    return (all(math.isfinite(x) for x in values)
            and 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)


def audit_mtsd(root, minimum=500, heldout=20, annotations_only=False):
    """Audit extracted fully annotated MTSD without changing the supplied splits."""
    config = OmegaConf.load(CONFIG)
    classes = list(config.classes)
    mapping = dict(config.mtsd_labels)
    annotation_dir = root / 'annotations'
    if not annotation_dir.is_dir():
        raise ValueError('Expected annotations/ under the extracted MTSD folder.')
    memberships = {}
    for split in ['train', 'val']:
        path = root / 'splits' / (split + '.txt')
        if not path.is_file():
            raise ValueError('Missing official split: ' + str(path))
        for key in path.read_text().splitlines():
            key = key.strip()
            if not key:
                continue
            if Path(key).name != key or Path(key).suffix:
                raise ValueError('Expected image keys, not paths, in official splits.')
            if key in memberships:
                raise ValueError('Duplicate image key in supplied splits: ' + key)
            memberships[key] = split
    image_paths = {}
    for path in ([] if annotations_only else root.rglob('*')):
        if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            if path.stem in image_paths:
                raise ValueError('Duplicate image key in image folders: ' + path.stem)
            image_paths[path.stem] = path
    counts = {s: Counter() for s in ['train', 'val']}
    image_counts = {s: Counter() for s in ['train', 'val']}
    label_counts = Counter()
    exclusions = Counter()
    manifest = []
    seen_hashes = {}
    leakage = []
    errors = []
    selected_images = 0
    for key, split in sorted(memberships.items()):
        annotation = annotation_dir / (key + '.json')
        if not annotation.is_file():
            errors.append('Missing annotation: ' + key)
            continue
        data = json.loads(annotation.read_text())
        if not isinstance(data.get('objects'), list):
            raise ValueError('Unrecognized MTSD schema in ' + str(annotation))
        if data.get('ispano', False):
            exclusions['panorama_image'] += 1
            continue
        targets = []
        exclude_image = False
        for obj in data['objects']:
            label = obj['label']
            label_counts[label] += 1
            if label not in mapping:
                continue
            flags = obj.get('properties', {})
            if any(flags.get(k, False) for k in ['ambiguous', 'out-of-frame', 'occluded']):
                exclude_image = True
                exclusions['difficult_target_image'] += 1
                break
            if not valid_box(obj['bbox'], data['width'], data['height']):
                errors.append('Invalid target box: ' + key)
                exclude_image = True
                break
            targets.append(obj)
        if exclude_image:
            continue
        if not targets:
            exclusions['no_selected_class'] += 1
            continue
        image_path = image_paths.get(key)
        if image_path is None and not annotations_only:
            errors.append('Missing selected image: ' + key)
            continue
        digest = None
        if not annotations_only:
            with Image.open(image_path) as image:
                image.load()
                if image.size != (data['width'], data['height']):
                    errors.append('Image dimensions disagree with annotation: ' + key)
                    continue
                digest = hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
        if digest is not None and digest in seen_hashes:
            earlier = seen_hashes[digest]
            if earlier['split'] != split:
                leakage.append([earlier['key'], key])
            else:
                exclusions['exact_duplicate_image'] += 1
            continue
        if digest is not None:
            seen_hashes[digest] = {'key': key, 'split': split}
        selected_images += 1
        present = set()
        for obj in targets:
            category = mapping[obj['label']]
            counts[split][category] += 1
            present.add(category)
            manifest.append({'image_path': str(image_path.resolve()) if image_path else '', 'image_key': key,
                             'split': split, 'sign_label': category,
                             'width': data['width'], 'height': data['height'],
                             **obj['bbox'], 'pixel_sha256': digest})
        image_counts[split].update(present)
    deficits = {}
    for split, target in [('train', minimum), ('val', heldout)]:
        for category in classes:
            missing = max(0, target - image_counts[split][category])
            if missing:
                deficits[split + '/' + category] = missing
    return {'source': 'MTSD fully annotated; original train/val lists',
            'classes': classes, 'mapping': mapping,
            'selected_images': selected_images,
            'box_counts': {s: {c: counts[s][c] for c in classes} for s in counts},
            'unique_image_counts': {s: {c: image_counts[s][c] for c in classes} for s in counts},
            'all_encountered_labels': dict(label_counts), 'exclusions': dict(exclusions),
            'annotations_only': annotations_only,
            'image_counts_verified': not annotations_only,
            'errors': errors, 'cross_split_exact_duplicates': leakage,
            'deficits': deficits,
            'annotation_counts_sufficient': not errors and not deficits,
            'ready_for_split_design': not annotations_only and not errors and not leakage and not deficits,
            'limitations': ['Readiness is not semantic label approval or final training approval.',
                           'Official validation is not a second independent local test set.',
                           'A group-aware internal validation partition must be made from official training before training.',
                           'Near-duplicate scenes require visual or sequence-based review.',
                           'Only exact g1 labels are mapped; unreviewed variants are not merged.',
                           'Discarding difficult-target scenes changes evaluation scope and must be reported.',
                           'Fully annotated status and image redistribution terms require source review.']}, manifest


def audit_gtsdb(path):
    config = OmegaConf.load(CONFIG)
    mapping = dict(zip(config.gtsdb_ids, config.classes))
    counts = {s: Counter() for s in ['official_train', 'official_test']}
    images = {s: defaultdict(set) for s in counts}
    with path.open() as stream:
        for row in csv.reader(stream, delimiter=';'):
            category = mapping.get(int(row[5]))
            if category:
                split = 'official_train' if int(Path(row[0]).stem) < 600 else 'official_test'
                counts[split][category] += 1
                images[split][category].add(row[0])
    return {'source': 'GTSDB supplied gt.txt; official 600/300 image split',
            'box_counts': {s: {c: counts[s][c] for c in config.classes} for s in counts},
            'unique_image_counts': {s: {c: len(images[s][c]) for c in config.classes} for s in counts},
            'meets_500_training_images_per_class': all(len(images['official_train'][c]) >= 500 for c in config.classes),
            'note': 'Bounding boxes in one photograph are not distinct training images. No images are generated or repeated to inflate the count.'}


def main():
    parser = argparse.ArgumentParser(description='Check real five-class detection data before investing in training.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--mtsd-root', type=Path)
    group.add_argument('--gtsdb-gt', type=Path)
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'report/detection_readiness')
    parser.add_argument('--minimum', type=int, default=500)
    parser.add_argument('--heldout-minimum', type=int, default=20)
    parser.add_argument('--annotations-only', action='store_true', help='Count candidate image keys before downloading images; never establishes readiness.')
    args = parser.parse_args()
    if min(args.minimum, args.heldout_minimum) < 1:
        parser.error('Minimum counts must be positive.')
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        parser.error('Use a new output folder to preserve previous audits.')
    if args.mtsd_root:
        audit, manifest = audit_mtsd(args.mtsd_root, args.minimum, args.heldout_minimum, args.annotations_only)
    else:
        audit, manifest = audit_gtsdb(args.gtsdb_gt), []
    output.mkdir(parents=True, exist_ok=True)
    (output / 'audit.json').write_text(json.dumps(audit, indent=2))
    if manifest:
        with (output / 'candidate_annotations.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
    print(json.dumps(audit, indent=2))
    successful=audit.get('annotation_counts_sufficient',False) if args.annotations_only else audit.get('ready_for_split_design',audit.get('meets_500_training_images_per_class',False))
    if not successful:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
