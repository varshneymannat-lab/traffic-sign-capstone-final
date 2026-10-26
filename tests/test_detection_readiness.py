import json

import pytest
from PIL import Image

from scripts.audit_detection_data import audit_gtsdb, audit_mtsd, valid_box


def make_mtsd(tmp_path):
    (tmp_path / 'annotations').mkdir()
    (tmp_path / 'splits').mkdir()
    (tmp_path / 'images').mkdir()
    for split, key, colour in [('train', 'a', 'red'), ('val', 'b', 'blue')]:
        (tmp_path / 'splits' / (split + '.txt')).write_text(key + '\n')
        Image.new('RGB', (40, 30), colour).save(tmp_path / 'images' / (key + '.png'))
        data = {'width': 40, 'height': 30, 'objects': [
            {'label': 'regulatory--stop--g1', 'bbox': {'xmin': 2, 'ymin': 3, 'xmax': 20, 'ymax': 25}, 'properties': {}}]}
        (tmp_path / 'annotations' / (key + '.json')).write_text(json.dumps(data))
    return tmp_path


@pytest.mark.parametrize('box', [
    {'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 1},
    {'xmin': -1, 'ymin': 0, 'xmax': 1, 'ymax': 1},
    {'xmin': 0, 'ymin': 0, 'xmax': 41, 'ymax': 1},
    {'xmin': 0, 'ymin': 0, 'xmax': float('nan'), 'ymax': 1},
])
def test_invalid_boxes(box):
    assert not valid_box(box, 40, 30)


def test_missing_classes_fail_readiness(tmp_path):
    audit, manifest = audit_mtsd(make_mtsd(tmp_path), 1, 1)
    assert audit['unique_image_counts']['train']['stop'] == 1
    assert not audit['ready_for_split_design']
    assert audit['deficits']['train/no_entry'] == 1
    assert len(manifest) == 2


def test_repeated_box_does_not_inflate_image_count(tmp_path):
    root = make_mtsd(tmp_path)
    path = root / 'annotations/a.json'
    data = json.loads(path.read_text())
    data['objects'] *= 2
    path.write_text(json.dumps(data))
    audit, _ = audit_mtsd(root, 2, 1)
    assert audit['box_counts']['train']['stop'] == 2
    assert audit['unique_image_counts']['train']['stop'] == 1
    assert audit['deficits']['train/stop'] == 1


def test_cross_split_duplicate_pixels_detected(tmp_path):
    root = make_mtsd(tmp_path)
    Image.new('RGB', (40, 30), 'red').save(root / 'images/b.png')
    audit, _ = audit_mtsd(root, 1, 1)
    assert audit['cross_split_exact_duplicates'] == [['a', 'b']]
    assert not audit['ready_for_split_design']


def test_missing_selected_image_is_error(tmp_path):
    root = make_mtsd(tmp_path)
    (root / 'images/a.png').unlink()
    audit, _ = audit_mtsd(root, 1, 1)
    assert 'Missing selected image: a' in audit['errors']


def test_annotation_only_audit_never_claims_image_readiness(tmp_path):
    root = make_mtsd(tmp_path)
    (root / 'images/a.png').unlink()
    audit, manifest = audit_mtsd(root, 1, 1, annotations_only=True)
    assert audit['unique_image_counts']['train']['stop'] == 1
    assert not audit['image_counts_verified']
    assert not audit['ready_for_split_design']
    assert manifest[0]['image_path'] == ''
    assert not audit['errors']


def test_unsupported_speed_variants_not_silently_mapped(tmp_path):
    root = make_mtsd(tmp_path)
    path = root / 'annotations/a.json'
    data = json.loads(path.read_text())
    data['objects'][0]['label'] = 'regulatory--maximum-speed-limit-30--g2'
    path.write_text(json.dumps(data))
    audit, manifest = audit_mtsd(root, 1, 1)
    assert not [r for r in manifest if r['split'] == 'train']
    assert audit['exclusions']['no_selected_class'] == 1


def test_difficult_target_discards_whole_image(tmp_path):
    root = make_mtsd(tmp_path)
    path = root / 'annotations/a.json'
    data = json.loads(path.read_text())
    data['objects'][0]['properties']['occluded'] = True
    path.write_text(json.dumps(data))
    audit, manifest = audit_mtsd(root, 1, 1)
    assert audit['exclusions']['difficult_target_image'] == 1
    assert not [r for r in manifest if r['split'] == 'train']


def test_overlapping_split_membership_fails(tmp_path):
    root = make_mtsd(tmp_path)
    (root / 'splits/val.txt').write_text('a\n')
    with pytest.raises(ValueError, match='Duplicate image key'):
        audit_mtsd(root)


def test_gtsdb_uses_official_split_and_distinct_image_counts(tmp_path):
    path = tmp_path / 'gt.txt'
    path.write_text('00001.ppm;0;0;10;10;14\n00001.ppm;20;20;30;30;14\n00600.ppm;0;0;10;10;14\n')
    audit = audit_gtsdb(path)
    assert audit['box_counts']['official_train']['stop'] == 2
    assert audit['unique_image_counts']['official_train']['stop'] == 1
    assert audit['box_counts']['official_test']['stop'] == 1
    assert not audit['meets_500_training_images_per_class']
