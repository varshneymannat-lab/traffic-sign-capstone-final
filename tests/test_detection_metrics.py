import numpy as np
import torch

from scripts.evaluate_detector import aggregate_counts, fixed_metrics
from scripts.prepare_mtsd import visual_groups


def test_detection_requires_both_box_and_correct_class():
    result = fixed_metrics(torch.tensor([1.]), torch.tensor([0]), torch.tensor([[0.9]]))
    assert (result['tp'],result['fp'],result['fn']) == (0,1,1)


def test_one_ground_truth_cannot_match_two_predictions():
    result = fixed_metrics(torch.tensor([0.,0.]),torch.tensor([0]),torch.tensor([[0.9,0.8]]))
    assert (result['tp'],result['fp'],result['fn']) == (1,1,0)


def test_empty_predictions_count_missed_signs():
    result = fixed_metrics(torch.empty(0),torch.tensor([0,1]),torch.empty(2,0))
    assert (result['tp'],result['fp'],result['fn']) == (0,0,2)


def test_aggregate_uses_counts_not_mean_of_image_accuracies():
    result = aggregate_counts([{'tp':1,'fp':0,'fn':0},{'tp':0,'fp':2,'fn':3}])
    assert result['precision'] == 1/3
    assert result['recall'] == 1/4


def test_visual_groups_are_transitive():
    records = [{'dhash':0},{'dhash':1},{'dhash':3},{'dhash':255}]
    groups = visual_groups(records,distance=1)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]
