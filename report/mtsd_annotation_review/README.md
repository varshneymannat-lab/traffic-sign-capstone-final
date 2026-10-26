# Annotation Review

Compared all 2,706 prepared scenes and 3,433 retained target boxes with original MTSD annotations. Mismatched scenes: 0. Target category counts and normalized box coordinates are checked, not merely the existence of label files.

The first review example contains one selected speed-30 sign and three other source-annotated signs. The large road-bump and priority-for-oncoming-traffic signs are outside the five categories; they are not missing from the source. The numbered image shows all source boxes, with yellow for selected targets and cyan for other categories.

The second example contains two selected speed-50 signs. Their original boxes are approximately 57-59 pixels across, but are about 14 pixels across after the existing 1280-pixel preparation. Enlarged crops illustrate available original detail, not reconstructed detail. Increasing label font size only fixes presentation readability.

Other source classes are deliberately background in this closed-set five-category detector. A prediction on one of them is not automatically correct. This policy is different from losing an annotation for an in-scope sign. No cross-class relabelling is used to hide missing labels.

The category names abbreviate five exact g1 source labels. Related but unmapped source variants also occur: {"complementary--maximum-speed-limit-30--g1": 1, "regulatory--maximum-speed-limit-30--g3": 2, "regulatory--maximum-speed-limit-50--g6": 4, "regulatory--stop--g10": 6, "regulatory--stop--g2": 7}. They were not merged during the recorded experiments. This is a real coverage limitation, including related stop/speed variants, not evidence that all visual variants of those five categories are labelled as targets. A broader semantic task needs reviewed remapping and retraining; test labels must not be silently revised after seeing results.

The audit cannot prove that the source contains every real sign, and does not substitute for a complete manual annotation review. Difficult and unsupported categories remain limitations. Whole-image versus overlapping-tile inference was selected on internal validation only; measured accuracy and timing are recorded in the final technical report.

Images and annotations: Mapillary Traffic Sign Dataset v2. Source access: https://www.mapillary.com/dataset/trafficsign . Preserve the distributed dataset licence.
