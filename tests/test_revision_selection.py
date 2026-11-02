from scripts.select_revision_detector import select
import pytest


def comparison(score, name, test_score):
    return {'selected_experiment': name, 'dataset': {'split': 'fixed'},
            'experiments': [{'name': name, 'checkpoint': name+'.pt', 'image_size': 640,
                             'validation': {'mAP50_95': score}, 'test': {'mAP50_95': test_score}}]}


def test_detector_choice_uses_validation_not_test():
    result = select(comparison(.3, 'old', .9), comparison(.4, 'new', .1))
    assert result['selected_checkpoint'] == 'new.pt'


def test_tie_preserves_baseline_and_inputs_are_not_mutated():
    old = comparison(.3, 'old', .1)
    result = select(old, comparison(.3, 'new', .9))
    assert result['selected_checkpoint'] == 'old.pt'
    assert old['experiments'][0]['name'] == 'old'


def test_different_datasets_are_rejected():
    new = comparison(.4, 'new', .1)
    new['dataset']['split'] = 'changed'
    with pytest.raises(ValueError, match='same prepared'):
        select(comparison(.3, 'old', .9), new)
