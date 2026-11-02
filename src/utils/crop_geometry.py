"""Map boxes already expressed in scene pixels to another copy of that scene."""
import math


def map_box_to_source(box, scene_size, source_size):
    width, height = scene_size
    source_width, source_height = source_size
    if min(width, height, source_width, source_height) <= 0:
        raise ValueError("Image dimensions must be positive")
    if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
        raise ValueError("Expected four finite box coordinates")
    x1, y1, x2, y2 = [float(value) for value in box]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Empty box")
    left = max(0, math.floor(x1 * source_width / width))
    top = max(0, math.floor(y1 * source_height / height))
    right = min(source_width, math.ceil(x2 * source_width / width))
    bottom = min(source_height, math.ceil(y2 * source_height / height))
    if right <= left or bottom <= top:
        raise ValueError("Box lies outside the source image")
    return left, top, right, bottom
