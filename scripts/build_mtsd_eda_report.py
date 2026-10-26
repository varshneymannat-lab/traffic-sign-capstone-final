"""Build the dataset-review PDF from the prepared MTSD subset and EDA files."""

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
EDA = ROOT / "report/mtsd_eda"
DATA = ROOT / "data/mtsd"
OUTPUT = EDA / "Mapillary_EDA_Report.pdf"
GREEN = colors.HexColor("#17604d")
INK = colors.HexColor("#202a2e")
WIDTH = A4[0] - 92
styles = getSampleStyleSheet()
styles.add(ParagraphStyle("ReportTitle", fontName="Helvetica-Bold", fontSize=22,
                          leading=26, textColor=GREEN, spaceAfter=12))
styles.add(ParagraphStyle("SectionTitle", fontName="Helvetica-Bold", fontSize=14,
                          leading=18, textColor=GREEN, spaceBefore=9, spaceAfter=8))
styles.add(ParagraphStyle("ReportBody", fontName="Helvetica", fontSize=10.5,
                          leading=14, textColor=INK, spaceAfter=8))
styles.add(ParagraphStyle("CaptionText", fontName="Helvetica", fontSize=9,
                          leading=12, textColor=colors.HexColor("#45545b"), spaceAfter=10))
styles.add(ParagraphStyle("CellText", fontName="Helvetica", fontSize=9,
                          leading=12, alignment=TA_LEFT))


def paragraph(text, style="ReportBody"):
    return Paragraph(text, styles[style])


def image(name, max_height, width=WIDTH):
    path = EDA / name
    with PILImage.open(path) as source:
        w, h = source.size
    scale = min(width / w, max_height / h)
    return Image(str(path), width=w * scale, height=h * scale)


def table(rows, widths):
    wrapped = [[paragraph(str(cell), "CellText") for cell in row] for row in rows]
    result = Table(wrapped, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6efeb")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), .7, GREEN),
        ("LINEBELOW", (0, 1), (-1, -1), .35, colors.HexColor("#d8dfdc")),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return result


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#c8d5ce"))
    canvas.line(46, 38, A4[0] - 46, 38)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#52615b"))
    canvas.drawString(46, 25, "Mannat Varshney | Mapillary dataset review | Revised 26 October 2026")
    canvas.drawRightString(A4[0] - 46, 25, str(doc.page))
    canvas.restoreState()


def verify_data(summary):
    with (DATA / "annotations_split.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    scene_splits = defaultdict(set)
    group_splits = defaultdict(set)
    hash_splits = defaultdict(set)
    boxes = {s: Counter() for s in ("train", "val", "test")}
    scenes = {s: defaultdict(set) for s in boxes}
    dimensions = []
    checked = set()
    for row in rows:
        split, category = row["split"], row["sign_label"]
        scene_splits[row["image_key"]].add(split)
        group_splits[row["group_id"]].add(split)
        hash_splits[row["pixel_sha256"]].add(split)
        boxes[split][category] += 1
        scenes[split][category].add(row["image_key"])
        x, y, w, h = [float(row[k]) for k in ("x_center", "y_center", "box_width", "box_height")]
        assert all(math.isfinite(n) for n in (x, y, w, h))
        assert w > 0 and h > 0
        assert x - w / 2 >= -1e-6 and y - h / 2 >= -1e-6
        assert x + w / 2 <= 1 + 1e-6 and y + h / 2 <= 1 + 1e-6
        if row["image_path"] not in checked:
            with PILImage.open(DATA / row["image_path"]) as source:
                source.load()
                dimensions.append(source.size)
            checked.add(row["image_path"])
    assert all(len(s) == 1 for groups in (scene_splits, group_splits, hash_splits) for s in groups.values())
    for split in boxes:
        assert dict(boxes[split]) == summary["box_counts"][split]
        assert {c: len(v) for c, v in scenes[split].items()} == summary["image_counts_per_class"][split]
        assert sum(s == {split} for s in scene_splits.values()) == summary["split_images"][split]
    audit = {
        "images_decoded": len(checked), "boxes_checked": len(rows),
        "normalized_boxes_valid": True, "image_keys_disjoint": True,
        "recorded_source_pixel_hashes_disjoint": True, "similarity_groups_disjoint": True,
        "prepared_width_range": [min(w for w, h in dimensions), max(w for w, h in dimensions)],
        "prepared_height_range": [min(h for w, h in dimensions), max(h for w, h in dimensions)],
    }
    (EDA / "report_validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main():
    summary = json.loads((EDA / "dataset_summary.json").read_text())
    stats = json.loads((EDA / "eda_statistics.json").read_text())
    audit = verify_data(summary)
    counts = summary["image_counts_per_class"]
    classes = summary["classes"]
    content = [paragraph("Mapillary Traffic Sign Dataset<br/>Exploratory Data Analysis", "ReportTitle"),
               paragraph("Mannat Varshney | Capstone dataset review | Revised 26 October 2026", "CaptionText"),
               paragraph("1. Purpose and Selected Data", "SectionTitle"),
               paragraph("This analysis checks whether real road photographs can support a five-category traffic-sign detection project. Unlike cropped-sign classification, detection must locate each target sign in a complete scene and identify its category. The source is the fully annotated Mapillary Traffic Sign Dataset (MTSD), version 2."),
               paragraph("The prepared subset contains <b>2,706 real photographs and 3,433 target bounding boxes</b>. The images are original road photographs, not generated drawings or signs pasted onto backgrounds. These counts describe the selected project subset, not the entire Mapillary dataset."),
               paragraph("Distinct Photographs Containing Each Category", "SectionTitle")]
    content.append(table([["Category", "Training", "Validation", "Local test"]] + [
        [c.replace("_", " "), counts["train"][c], counts["val"][c], counts["test"][c]] for c in classes
    ], [WIDTH * .37, WIDTH * .21, WIDTH * .21, WIDTH * .21]))
    content += [Spacer(1, 8), paragraph("One photograph may contain several categories, so the category counts above do not sum to the total number of photographs.", "CaptionText")]
    split_rows = [["Split", "Photographs", "Target boxes", "Purpose"]]
    for split, label, purpose in [("train", "Training", "Learn model weights"), ("val", "Validation", "Choose settings and checkpoint"), ("test", "Local test", "Final held-out evaluation")]:
        split_rows.append([label, summary["split_images"][split], sum(summary["box_counts"][split].values()), purpose])
    content += [table(split_rows, [80, 90, 85, WIDTH - 255]), Spacer(1, 10),
                paragraph("There are <b>427-478 distinct training photographs per category</b>. This exceeds 200 per category but does not provide 500 training photographs for every category after splitting and filtering. No duplicate copies were added to increase these counts."),
                paragraph("This is a dataset-suitability review. Annotated examples in this report are ground truth supplied by the dataset, <b>not predictions or proof of model accuracy</b>.")]

    content += [PageBreak(), paragraph("2. Class Distribution and Sign Appearance", "SectionTitle"),
                image("distribution_and_sign_size.png", 165),
                paragraph("Figure 1. Left: distinct scene counts by category and split. Right: relative sign-size proxy at two possible detector input sizes. The proxy is sqrt(normalized box area) multiplied by the square input size; it is not the actual letterboxed sign width or height.", "CaptionText"),
                paragraph(f"The median size proxy is {stats['approximate_object_side_pixels_median']['416']:.1f} pixels at 416 and {stats['approximate_object_side_pixels_median']['640']:.1f} pixels at 640. Many targets are small, making detection harder even though the categories are visually familiar. This motivates comparing input resolutions, but does not establish which setting will perform best."),
                image("real_category_examples.png", 335),
                paragraph("Figure 2. Five real training crops per category, in row order: stop, no entry, yield, speed limit 30, speed limit 50. Crops are enlarged for inspection; enlargement cannot recover missing detail. Some examples come from the same photograph. These are visual examples, not independent sample counts.", "CaptionText"),
                paragraph("Variation includes small apparent size, blur, perspective, lighting and international text. The 30/50 labels identify the displayed number and sign family; the application does not infer national speed units.")]

    content += [PageBreak(), paragraph("3. Full-Scene Annotation Examples", "SectionTitle"),
                paragraph("Yellow rectangles and labels below come from the source annotations. Only the five selected categories are marked. All examples on pages 3-5 are from the training split, not the final local test split."),
                image("ground_truth_stop.jpg", 267),
                paragraph("Figure 3. Stop-category example. The full scene also contains another selected category. Keeping the surrounding road context distinguishes detection data from a collection of isolated sign crops.", "CaptionText"),
                image("ground_truth_no_entry.jpg", 267),
                paragraph("Figure 4. No-entry example. Signs occupy only part of the photograph; a detector must locate them before any optional crop classifier can identify them.", "CaptionText")]

    content += [PageBreak(), paragraph("4. Further Real-Scene Examples", "SectionTitle"),
                image("ground_truth_yield.jpg", 287),
                paragraph("Figure 5. Yield-category example. Multiple target categories can occur together. Every retained annotation belonging to a selected category is included in the prepared labels.", "CaptionText"),
                image("ground_truth_speed_limit_30.jpg", 287),
                paragraph("Figure 6. Speed-limit-30 example. At page scale, the marked sign can be difficult to read. The enlarged crops on page 2 illustrate why tiny objects need careful evaluation rather than assumptions based on the scene alone.", "CaptionText"),
                paragraph("The prepared images preserve aspect ratio, with the longest side capped at 1,280 pixels. Bounding boxes are normalized from the original annotations. Resizing can reduce visible detail in already small signs.")]

    content += [PageBreak(), paragraph("5. Quality Checks, Splits and Suitability", "SectionTitle"),
                image("ground_truth_speed_limit_50.jpg", 190),
                paragraph("Figure 7. Speed-limit-50 example from a full road photograph. Dataset images: Mapillary Traffic Sign Dataset; boxes overlaid from the supplied annotations.", "CaptionText"),
                paragraph("Checks Completed", "SectionTitle"),
                paragraph(f"All {audit['images_decoded']:,} prepared images decoded successfully, and all {audit['boxes_checked']:,} boxes have finite, positive dimensions within normalized image bounds. Prepared widths range from {audit['prepared_width_range'][0]} to {audit['prepared_width_range'][1]} pixels and heights from {audit['prepared_height_range'][0]} to {audit['prepared_height_range'][1]} pixels. Thumbnail mean grayscale brightness ranges from {stats['brightness_mean_gray_range'][0]:.1f} to {stats['brightness_mean_gray_range'][1]:.1f} on a 0-255 scale; this is a basic brightness check, not a full image-quality score."),
                paragraph("Official training images supply the internal training and validation splits. Selected official validation images are reserved as the <b>final local test</b>; the hidden official test set is unused. Exact source-pixel hashes and full-scene difference-hash similarity groups do not overlap across the prepared splits. Three training candidates overlapping held-out similarity groups were removed. These checks do not prove that every physical sign or road sequence is independent."),
                paragraph("Limits and Conclusion", "SectionTitle"),
                paragraph("Panoramas and scenes with selected signs flagged ambiguous, occluded or out of frame were excluded. Only target-positive scenes were sampled, so the subset does not measure false alarms on sign-free drives. Other sign classes remain unlabelled background for this five-category task. Crop spot checks are not a manual audit of every annotation."),
                paragraph("<b>Conclusion:</b> this subset is suitable for a bounded five-category detection experiment: it contains real full scenes, bounding boxes, and more than 400 training scenes per category. Small objects remain a major challenge. EDA establishes data readiness, not model performance; held-out detection precision, recall and mAP are still needed to assess the system."),
                paragraph('Sources: <link href="https://www.mapillary.com/dataset/trafficsign" color="#17604d">Mapillary Traffic Sign Dataset access</link>; Ertler et al., <i>The Mapillary Traffic Sign Dataset for Detection and Classification on a Global Scale</i>, ECCV 2020 (<link href="https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680069.pdf" color="#17604d">paper</link>). Dataset images and annotations remain subject to the distributed licence; retain DATASET_LICENSE.txt with this report. Counts and checks: prepared annotations, dataset_summary.json and report_validation.json.', "CaptionText")]
    review_dir = ROOT / 'report/mtsd_annotation_review'
    review = json.loads((review_dir/'audit.json').read_text())
    modes = json.loads((ROOT/'outputs/mtsd_detection/inference_comparison.json').read_text())
    first, second = [e['image_key'] for e in review['examples']]
    content += [PageBreak(), paragraph('6. Review of Apparently Missing Labels', 'SectionTitle'),
                paragraph('The earlier five-category figures did not display every source annotation. The large nearby road-bump and oncoming-traffic-priority signs in the speed-30 scene are outside the five selected categories. They are labelled in the original dataset. The view below displays all source boxes, with numbered labels for readability.'),
                image('../mtsd_annotation_review/'+first+'_all_source.jpg', 320),
                paragraph('Figure 8. Yellow: selected category. Cyan: another category in the original annotations. These are source boxes, not model predictions.', 'CaptionText'),
                table([['ID','Original source category','Used as target?']]+[
                    [a['id'],a['source_label'].replace('--',' / '),'Yes' if a['selected'] else 'No']
                    for a in review['examples'][0]['all_annotations']], [35, WIDTH-110, 75]),
                Spacer(1,8),
                paragraph(f"The source-retention audit compared all {review['prepared_scenes_checked']:,} prepared scenes and {review['retained_target_boxes']:,} target boxes. It found {len(review['mismatched_scenes'])} scenes with missing, extra or geometrically changed mapped target boxes. This confirms retention of supplied target labels; it does not prove that the source annotations contain every real sign."),
                paragraph('Other categories are background in this closed-set task. The five names abbreviate exact g1 source labels. Nineteen related regulatory stop/speed variants and one complementary speed sign remain outside that mapping. This is a coverage limitation, not proof that every visual variant of each category has been retained. Broader coverage needs semantic review, remapping and retraining; recorded test labels must not be silently changed.')]
    content += [PageBreak(), paragraph('7. Review of Small Signs', 'SectionTitle'),
                image('../mtsd_annotation_review/'+second+'_all_source.jpg',230),
                paragraph('Figure 9. The two yellow boxes mark speed-50 targets. The cyan boxes identify other source-annotated categories. Numbered callouts make the annotation locations visible without suggesting that the signs themselves are larger.', 'CaptionText'),
                image('../mtsd_annotation_review/'+second+'_target_crops.png',125),
                paragraph('Figure 10. Enlarged crops from the original image. Source boxes are about 57-59 pixels across, approximately 14 pixels after the existing 1,280-pixel preparation and 7 pixels at a whole-scene 640-pixel input. Enlarging the display does not add image detail.', 'CaptionText'),
                paragraph('Validation Experiment', 'SectionTitle'),
                paragraph('The same trained detector was compared using whole-image inference and SAHI overlapping 640-pixel tiles, including a whole-image prediction. Tiles overlap by 20%, and class-aware NMS merges their detections. The comparison uses only 356 internal validation scenes; it is not additional training data.'),
                table([['Inference mode','Val mAP50','Val mAP50:95','Mean ms/scene']]+[
                    [r['inference_mode'],f"{r['mAP50']:.4f}",f"{r['mAP50_95']:.4f}",f"{r['mean_read_and_detector_ms']:.1f}"] for r in modes], [145,110,120,WIDTH-375]),
                Spacer(1,8), paragraph('Tiled inference improves aggregate validation detection quality but costs more processing time. It cannot reconstruct detail lost during preparation and does not guarantee that every tiny sign is detected. Larger page labels improve readability only. No real-time guarantee is made.'),
                paragraph('Annotation audit: report/mtsd_annotation_review/audit.json. Experiment evidence: outputs/mtsd_detection/inference_comparison.json. Tiling implementation: SAHI, github.com/obss/sahi. Dataset images and annotations: Mapillary Traffic Sign Dataset v2.', 'CaptionText')]
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=46, rightMargin=46,
                            topMargin=40, bottomMargin=50, title="Mapillary Traffic Sign Dataset - Exploratory Data Analysis",
                            author="Mannat Varshney", subject="Five-category real-scene dataset suitability review")
    doc.build(content, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
