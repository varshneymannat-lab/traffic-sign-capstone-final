import pytest

from src.utils.crop_geometry import map_box_to_source


def test_maps_independent_axes_from_scene_coordinates():
    assert map_box_to_source([10, 20, 30, 40], (100, 80), (400, 200)) == (40, 50, 120, 100)


def test_rounds_outwards_and_clamps():
    assert map_box_to_source([-1, 0.1, 101, 80.1], (100, 80), (400, 200)) == (0, 0, 400, 200)


@pytest.mark.parametrize("box", [[1, 1, 1, 3], [0, 0, float('nan'), 3], [200, 200, 300, 300]])
def test_rejects_invalid_boxes(box):
    with pytest.raises(ValueError):
        map_box_to_source(box, (100, 80), (400, 200))


def test_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        map_box_to_source([0, 0, 2, 2], (0, 80), (400, 200))
