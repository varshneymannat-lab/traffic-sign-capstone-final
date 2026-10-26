import io
import zipfile

from scripts.eda_gtsrb import duplicate_summary, numeric_summary, read_annotations
from scripts.prepare_gtsrb import archive_rows


def test_duplicate_summary_distinguishes_leakage_and_label_conflicts():
    groups = {
        "a": [("train", 1, "a"), ("test", 1, "b")],
        "b": [("train", 1, "c"), ("train", 2, "d")],
        "c": [("test", 3, "e")],
    }
    assert duplicate_summary(groups) == {
        "duplicate_groups": 2, "extra_copies": 2,
        "cross_split_groups": 1, "conflicting_label_groups": 1,
    }


def test_numeric_summary():
    result = numeric_summary([10, 20, 30])
    assert result["min"] == 10
    assert result["median"] == 20
    assert result["max"] == 30


def test_official_roi_header_capitalization():
    buffer = io.BytesIO()
    member = "GTSRB/Final_Training/Images/00014/GT-00014.csv"
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(member, "Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId\n00000_00000.ppm;48;45;6;5;43;40;14\n")
    with zipfile.ZipFile(buffer) as archive:
        row = read_annotations(archive, "train")[0]
        prepared = archive_rows(archive, member, "official_train")[0]
    assert row["x1"] == 6
    assert row["x2"] == 43
    assert row["track"] == "14:00000"
    assert prepared["xmax"] == 44
