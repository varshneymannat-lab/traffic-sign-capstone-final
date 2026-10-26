import pytest

from scripts.prepare_gtsrb import split_tracks


def test_related_views_remain_in_one_partition():
    rows = [{"group_id": str(group), "image_path": f"{group}_{frame}"} for group in range(5) for frame in range(3)]
    result = split_tracks(rows, 6, 4, 42)
    assert len(result["train"]) == 6
    assert len(result["val"]) == 4
    assert {row["group_id"] for row in result["train"]}.isdisjoint({row["group_id"] for row in result["val"]})
    assert result == split_tracks(rows, 6, 4, 42)


def test_insufficient_independent_tracks_fails():
    rows = [{"group_id": "same", "image_path": str(i)} for i in range(10)]
    with pytest.raises(ValueError, match="Not enough separate tracks"):
        split_tracks(rows, 3, 3, 42)
