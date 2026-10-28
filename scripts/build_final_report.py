from __future__ import annotations

import html
import json
import os
from pathlib import Path

import markdown
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as PDFImage

PROJECT=Path(__file__).resolve().parents[1]
NAMES={'resnet18':'ResNet-18','mobilenet_v3_small':'MobileNetV3-Small','efficientnet_b0':'EfficientNet-B0'}


def main():
    report=PROJECT/'report'
    comparison=json.loads((PROJECT/'outputs/mtsd_detection/comparison.json').read_text())
    detection=json.loads((report/'mtsd_results/evaluation.json').read_text())
    deployment=json.loads((report/'mtsd_onnx_results/evaluation.json').read_text())
    export=json.loads((PROJECT/'models/mtsd/metadata.json').read_text())
    dataset=json.loads((PROJECT/'data/mtsd/dataset_summary.json').read_text())
    eda=json.loads((report/'mtsd_eda/eda_statistics.json').read_text())
    annotation_review=json.loads((report/'mtsd_annotation_review/audit.json').read_text())
    modes=json.loads((PROJECT/'outputs/mtsd_detection/inference_comparison.json').read_text())
    classifiers=json.loads((report/'mtsd_classifier_results/comparison.json').read_text())
    classifier_onnx=json.loads((report/'mtsd_classifier_results/onnx_verification.json').read_text())
    chosen=next(r for r in classifiers['experiments'] if r['architecture']==classifiers['selected_architecture'])
    classes=dataset['classes']
    selected=next(r for r in comparison['experiments'] if r['name']==comparison['selected_experiment'])
    styles=getSampleStyleSheet()
    styles['Normal'].fontName='Helvetica'
    styles['Normal'].fontSize=9.3
    styles['Normal'].leading=11.6
    styles['Normal'].spaceAfter=6
    styles['Title'].fontName='Helvetica-Bold'
    styles['Title'].fontSize=21
    styles['Title'].leading=25
    styles['Title'].textColor=colors.HexColor('#17604d')
    styles['Title'].alignment=0
    styles['Heading1'].fontName='Helvetica-Bold'
    styles['Heading1'].fontSize=13
    styles['Heading1'].leading=17
    styles['Heading1'].spaceBefore=8
    styles['Heading1'].spaceAfter=6
    styles['Heading2'].fontSize=11
    styles['Heading2'].leading=14
    styles['Heading2'].spaceBefore=8
    styles['Heading2'].spaceAfter=5
    styles['BodyText'].fontName='Helvetica'
    styles['BodyText'].fontSize=8.2
    styles['BodyText'].leading=10
    story=[]
    sections=[]
    def text(value,style='Normal'):
        story.append(Paragraph(value,styles[style]))
        sections.append(value)
    def heading(value):
        text(value,'Heading1')
    def table(rows,widths):
        converted=[[Paragraph(str(value),styles['BodyText']) for value in row] for row in rows]
        widget=Table(converted,colWidths=widths,repeatRows=1,hAlign='LEFT')
        widget.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eeeb')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#8ba399')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#dce2de')),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),('LEFTPADDING',(0,0),(-1,-1),6)]))
        story.append(widget)
        story.append(Spacer(1,7))
        sections.append('<table>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>')
    def picture(path,width,height):
        image=Image.open(path)
        factor=min(width/image.width,height/image.height)
        story.append(PDFImage(str(path),width=image.width*factor,height=image.height*factor))
        sections.append(f'<img src="{os.path.relpath(path,report)}" style="max-width:100%"/>')
    def page():
        story.append(PageBreak())
        sections.append('<hr/>')
    text('Traffic Sign Detection and Classification<br/>Using Real Road Images','Title')
    text('Mannat Varshney | Revised 28 October 2026','BodyText')
    heading('Abstract')
    text(f'This project develops a five-category traffic-sign detection and classification system using only the Mapillary Traffic Sign Dataset (MTSD). Real annotated crops compare pretrained ResNet-18, MobileNetV3-Small and EfficientNet-B0. Full road scenes train a lightweight YOLO11n detector. Both tasks inherit the same protected scene split: {dataset["split_images"]["train"]:,} training, {dataset["split_images"]["val"]} validation and {dataset["split_images"]["test"]} local test scenes. The selected classifier achieves {chosen["test"]["accuracy"]*100:.2f}% accuracy on 425 annotated test crops. Detector mAP@0.50 is {detection["detection"]["mAP50"]:.4f} and mAP@0.50:0.95 is {detection["detection"]["mAP50_95"]:.4f}. Actual ONNX inference is checked. Crop accuracy, scene detection quality and processing speed are reported separately.')
    heading('1. Introduction and Motivation')
    text('Traffic-sign inspection is a practical computer-vision task: a road photograph may contain a small, blurred or partly obscured sign among buildings, vehicles and other objects. The objective is to locate supported signs and label them as stop, no entry, yield, speed limit 30 or speed limit 50. The tool saves annotated images or video and a detection table. It is not a driving controller and does not decide whether a road is safe.')
    text('The applied problem is finding small signs within complex road photographs. The project tests whether lightweight models and overlapping inference tiles provide useful recognition and localisation under limited computing resources. It saves annotated images/video and prediction tables. It does not claim a complete road-inventory service or measured operational time savings.')
    text('The project retains the Lightning/Hydra classifier structure and uses one dataset in two forms: complete photographs for detection and their annotated sign regions for classification. Classification answers which category a supplied rectangle contains; detection must also find that rectangle. MTSD supplies both real scenes and manually annotated boxes [2]. No synthetic training images or second traffic-sign dataset are used in the final experiments.')
    heading('1.1 Research Questions')
    text('The experiments ask: which small classifier performs best under the same training budget; how detector input resolution changes validation detection quality; whether a separately trained crop classifier helps or hurts on detected global road-scene crops; and whether exported ONNX inference preserves the measured detector results. Model selection uses validation data, not final test scores.')
    heading('1.2 Related Work and Project Contribution')
    text('GTSRB and GTSDB motivate the distinction between recognition and detection [1]. The MTSD authors study globally varied signs and explicitly provide full scenes with fine-grained labels [2]. DFG demonstrates traffic-sign inventory as an application [3]. This project uses those ideas for a bounded student implementation: a five-class detector, controlled crop baselines, an executable image/video interface, deployment verification and reproducible experiment records. Published paper scores are not reported as this project\'s results.')
    text('Two additional repositories were reviewed during finalisation [5,6]. The first illustrates webcam detections; the second separates detection from crop recognition. Their class counts, datasets and budgets differ. Their reported scores are not comparable baselines here, and no code or weights were imported. Detailed comparison: report/reference_projects.md.','BodyText')
    page()
    heading('2. Methodology')
    text('2.1 Real Data, Annotations and EDA','Heading2')
    text('The classifiers use 2,563 training crops, 445 validation crops and 425 test crops cut from the prepared MTSD scenes. Crop splits are inherited from source photographs and similarity groups, never assigned independently. Each crop retains source-image identity, rectangle, category and hash. Training crop counts are stop 498, no entry 546, yield 576, speed 30 478 and speed 50 465. Resizing a small real crop does not create new image detail.')
    text('For detection, the authorised MTSD v2 fully annotated release is used. Its annotation ZIP matches the supplied MD5. Original selected image members pass ZIP CRC32 and have recorded SHA256 digests. Range downloads retrieve only the selected images; full image-archive MD5 verification is not claimed. No signs are drawn or pasted into generated scenes.')
    table([['Category','Train scenes','Val scenes','Test scenes','Train boxes']]+[[c.replace('_',' '),dataset['image_counts_per_class']['train'][c],dataset['image_counts_per_class']['val'][c],dataset['image_counts_per_class']['test'][c],dataset['box_counts']['train'][c]] for c in classes],[165,77,77,77,77])
    text('One scene can contain several boxes or categories, so category scene counts do not sum to total scenes. More than 400 distinct training scenes per class remain after validation is reserved; this is not 500 independent physical signs per class. Selected official validation scenes form the final local test; hidden test labels are unused. Global speed classes identify the posted number and sign family, not an inferred or converted national speed unit.')
    picture(report/'mtsd_eda/ground_truth_speed_limit_30.jpg',450,150)
    text('Figure 1. Actual MTSD training photograph. Yellow boxes mark only the five selected categories. Large nearby road-bump and oncoming-traffic-priority signs are annotated in the source but outside this task. These are ground-truth boxes, not detector predictions.','BodyText')
    text('2.2 Split Protection and Exclusions','Heading2')
    text('Exact decoded-pixel checks and 64-bit full-scene difference-hash components screen visually similar images. Three training scenes in components overlapping held-out data are removed. Internal training and validation are split by those groups. This screening does not prove physical-sign independence without sequence metadata. Panoramas and scenes with selected target signs flagged ambiguous, occluded or out-of-frame are excluded. The scope is therefore easier than unrestricted driving footage.')
    text(f'Following annotation review, all {annotation_review["prepared_scenes_checked"]:,} prepared scenes and {annotation_review["retained_target_boxes"]:,} target boxes were compared with the five exact g1 source labels and coordinates: {len(annotation_review["mismatched_scenes"])} scene mismatches were found. Other source labels are background. This includes 19 related regulatory stop/speed variants and one complementary speed sign not merged in the recorded task. Thus the five names do not imply all international variants are covered. Broader coverage requires reviewed remapping and retraining. The audit verifies mapped-label retention, not completeness of the original annotations.')
    page()
    text('2.3 Models, Preprocessing and Training','Heading2')
    text('The classifier study fine-tunes all parameters of three ImageNet-pretrained models for five epochs on Apple MPS, using 128-pixel RGB crops, ImageNet normalisation, AdamW, learning rate 0.0003, weight decay 0.0001, batch size 16 and seed 42. Training uses small rotations and colour jitter without horizontal flipping. Checkpoints and architecture are selected by validation macro-F1. Test scoring and forward latency use CPU. ImageNet/COCO initial weights are transfer learning, not additional project traffic-sign datasets.')
    text('YOLO11n starts from general pretrained weights with a new five-class head. Initial CPU runs compare 416/640 inputs for five epochs, batch 8, AdamW, learning rate 0.001, weight decay 0.0005 and seed 42; ten backbone blocks are frozen. Real scenes receive small rotation, translation, scaling and colour changes, not flips, mosaic or copy-paste. Prepared scenes preserve aspect ratio at up to 1,280 pixels, then are letterboxed for training.')
    if len(comparison['experiments'])>2:
        text('A further 15-epoch run starts from the selected 640 checkpoint, with five frozen blocks, learning rate 0.0005 and Apple M1 MPS. This validation-led extension changes initialization, budget and hardware; durations are not a controlled speed comparison.')
    text('Small objects are addressed using SAHI overlapping 640-pixel inference tiles, including a whole-image pass, 20% overlap and class-aware NMS at IoU 0.7. Whole versus tiled mode is selected on 356 internal validation scenes using mAP50:95. A confidence of 0.343 is selected by validation F1. Tiling changes inference, not training-image counts, and cannot reconstruct detail lost in the 1,280-pixel preparation.')
    text('2.4 Metrics and Their Selection','Heading2')
    text('Crop accuracy is the fraction correct. Precision measures how often a predicted category is correct; recall measures the fraction of actual members found. F1 = 2PR/(P+R). Macro-F1 weights unequal-support categories equally and selects the classifier. Confusion matrices expose individual category errors.')
    text('Detector IoU is box-intersection area divided by union area. mAP50 averages class average precision at IoU >=0.50; mAP50:95 averages ten IoU thresholds, tests tighter localisation and selects detector settings. Unlike crop accuracy, these scores account for misses and false detections. Fixed-threshold P/R/F1 describe one operating point.')
    heading('3. Experiments and Results')
    text('3.1 Mapillary Classifier Comparison','Heading2')
    table([['Model','Val F1','Test acc.','Macro P','Macro R','Macro F1']]+[[NAMES[e['architecture']],f'{e["validation"]["macro_f1"]:.4f}',f'{e["test"]["accuracy"]*100:.2f}%',f'{e["test"]["macro_precision"]:.4f}',f'{e["test"]["macro_recall"]:.4f}',f'{e["test"]["macro_f1"]:.4f}'] for e in classifiers['experiments']],[130,65,68,70,70,70])
    correct=sum(p['actual']==p['predicted'] for p in chosen['test']['predictions'])
    text(f'{NAMES[chosen["architecture"]]} is selected by validation, not test accuracy. It correctly predicts {correct} of 425 annotated test crops. The saved confusion matrix and per-image predictions identify the errors. The majority-training-label baseline scores {json.loads((report/"mtsd_classifier_results/baseline.json").read_text())["accuracy"]*100:.2f}% accuracy. Parameter counts and measured CPU forward times are retained with results. These are not scene-detection accuracy scores.')
    text('3.2 Controlled Detector Comparison','Heading2')
    table([['Run','Val mAP50','Val mAP50:95','Train min']]+[[e['name'],f'{e["validation"]["mAP50"]:.4f}',f'{e["validation"]["mAP50_95"]:.4f}',f'{e["training_seconds"]/60:.1f}'] for e in comparison['experiments']],[230,88,88,77])
    text(f'Selected before test evaluation: {comparison["selected_experiment"]}. The two initial resolution runs share a budget and split. The extra fine-tune has a different initialization and budget. These are settings of one detector architecture, not three architectures. Each is a single-seed run.')
    table([['Inference mode','Val mAP50','Val mAP50:95','Mean ms/scene']]+[[r['inference_mode'],f'{r["mAP50"]:.4f}',f'{r["mAP50_95"]:.4f}',f'{r["mean_read_and_detector_ms"]:.1f}'] for r in modes],[160,105,110,108])
    text('Same checkpoint, validation scenes and fixed-square preprocessing: tiling improves mAP50:95 from 0.2198 to 0.3379 at about eight times the read/detector cost. Training-validator preprocessing differs, so its scores are separate from this controlled mode comparison.','BodyText')
    page()
    text('3.3 Held-out Road-Scene Results','Heading2')
    text(f'On {detection["test_images"]} selected held-out scenes, detector mAP@0.50 is {detection["detection"]["mAP50"]:.4f} and mAP@0.50:0.95 is {detection["detection"]["mAP50_95"]:.4f}. Precision and recall at the maximum-F1 curve point are {detection["detection"]["precision_at_max_f1"]:.4f} and {detection["detection"]["recall_at_max_f1"]:.4f}; they are not the same as the fixed-confidence numbers below.')
    per_class={int(i):(a,b) for i,a,b in zip(detection['detection']['class_indices'],detection['detection']['per_class_AP50'],detection['detection']['per_class_AP50_95'])}
    table([['Category','AP50','AP50:95']]+[[c.replace('_',' '),f'{per_class.get(i,(0,0))[0]:.4f}',f'{per_class.get(i,(0,0))[1]:.4f}'] for i,c in enumerate(classes)],[280,102,101])
    text('3.4 Detector Labels versus Crop Refinement','Heading2')
    fixed=detection['fixed_threshold']
    table([['Pipeline','Precision','Recall','F1','TP / FP / FN']]+[[label,f'{r["precision"]:.4f}',f'{r["recall"]:.4f}',f'{r["f1"]:.4f}',f'{r["tp"]} / {r["fp"]} / {r["fn"]}'] for label,r in [('Detector only',fixed['detector_only']),('Detector + crop model',fixed['detector_plus_classifier'])]],[169,67,67,67,113])
    text(f'These counts use validation-selected confidence {fixed["confidence"]:.3f}, IoU 0.50 and one-to-one same-class matches. The new Mapillary-trained ONNX classifier changes labels, not boxes, and cannot recover misses. The predeclared application default remains detector-only; no test-driven switch is made. On {detection["matched_box_classification"]["refined_labels"]["samples"]} class-agnostically matched accepted boxes, refined crop accuracy is {detection["matched_box_classification"]["refined_labels"]["accuracy"]*100:.2f}%. That conditional score excludes missed signs. Ground-truth crop accuracy and predicted-box accuracy have different inputs.')
    preview=sorted((PROJECT/'demo/mtsd').glob('*_detector.jpg'))[0]
    picture(preview,450,165)
    text('Figure 2. Actual processed held-out road scene. The demo selects early examples by source category, not correctness, and also contains missed or incorrect detections. A gallery video of these real still photographs is not continuous road footage.','BodyText')
    text('3.5 Processing Time','Heading2')
    text(f'The warm PyTorch-detector-plus-ONNX-classifier pipeline averages {detection["timing"]["mean_ms"]:.2f} ms per scene, with median {detection["timing"]["median_ms"]:.2f} ms and 95th percentile {detection["timing"]["p95_ms"]:.2f} ms. Timing includes image reading, detector preprocessing/inference/NMS and crop inference; loading models, scoring and output encoding are excluded. Measurements use the recorded macOS CPU system, two PyTorch threads and two classifier ONNX threads. This is not a continuous-video real-time guarantee.')
    page()
    heading('4. Deployment, Reproducibility and Discussion')
    text('4.1 ONNX and Optimisation','Heading2')
    text(f'ONNX is an exported model format, not a third neural architecture. The selected detector is exported with opset 17, fixed batch size 1 and {export["image_size"]}-pixel input. ONNX model structure checks pass. Raw outputs on three real test images are compared numerically with PyTorch; the largest absolute difference is {max(r["max_abs_difference"] for r in export["numerical_checks"]):.6f}. Preprocessing and non-maximum suppression are still provided by the application, not embedded in the graph.')
    fp32=classifier_onnx['sessions']['float32']
    int8=classifier_onnx['sessions']['dense_int8']
    text(f'Actual ONNX detection on all {deployment["test_images"]} local test scenes gives mAP50 {deployment["detection"]["mAP50"]:.4f}, versus PyTorch {detection["detection"]["mAP50"]:.4f}. The new classifier float32 and dense-only dynamic INT8 exports both score {fp32["accuracy"]*100:.2f}% on 425 crops. CPU forward means are {fp32["forward_latency"]["mean_ms"]:.2f}/{int8["forward_latency"]["mean_ms"]:.2f} ms (10 warm-ups, 50 repeats), with sizes {fp32["bytes"]/1e6:.3f}/{int8["bytes"]/1e6:.3f} MB. No convincing speed/compression gain is established. Convolutions stay float32; this is not detector quantisation or pruning.')
    text('4.2 Versioning and Execution','Heading2')
    text('The project uses Git for code, DVC for data/model versions, TensorBoard for training curves, Hydra for classifier configuration, and Lightning for the classifier loop. Detection settings are in pipelines/detection/. The image/video command writes annotated media, CSV detections and measured timing. Dataset terms apply separately from code; temporary authorised download links are not committed. Restoring licensed data requires authorised artifact access or regeneration from an authorised download.')
    container_record=report/'container_verification.json'
    if container_record.exists() and json.loads(container_record.read_text()).get('conclusion')=='success':
        text('Published Mapillary-only commit 449ef7a passes Docker build, container tests and two-frame ONNX inference in Actions run 40690970167. Full retraining inside Docker is unverified; later revisions require their own CI evidence. The submission is a fork of Vladimir-125/lightning-hydra-template with its MIT notice and history. No shared DVC remote is configured; full training needs authorised data/cache access.')
    else:
        text('A Docker recipe and automated build/test workflow define the execution environment. Local Docker execution is not verified because Docker is unavailable on this machine; CI build status must be checked after upload. The final submission repository is a verified fork of Vladimir-125/lightning-hydra-template, with the upstream MIT notice and initial template commit preserved. Remote artifact access must be documented separately from the source repository.')
    text('4.3 Application and Software Verification','Heading2')
    text('The command-line application accepts an image or video, runs whole/tiled detection, optionally classifies accepted crops, and saves annotated media, CSV detections and timing. Scores from the two models are kept separate. The packaged models are hash-checked after fresh extraction and tested on real inputs. Software tests check data handling, training, export and inference utilities; they do not prove predictive reliability. Visual inspection of original images remains necessary because signs are still missed.')
    text('4.4 Difficulties and Limits','Heading2')
    text('The main difficulties were acquiring real scenes, protecting splits, preserving labels and finding small signs. The mentor identified apparently missing labels and tiny targets. Source comparison showed nearby omitted signs were unsupported categories; the reviewed speed-50 targets shrink to about 14 pixels at preparation and 7 at whole-scene input. Numbered views fix report readability; tiling measurably improves aggregate detection but does not solve every tiny sign. The final revision retrains all crop models on Mapillary instead of reusing earlier cross-dataset scores.')
    text('The study uses selected positive scenes, excludes difficult targets and uses one seed per run. Hashes do not prove physical-sign independence. The protected holdout was inspected in earlier project development, so this is not an external blind evaluation. Unsupported categories and false alarms on sign-free journeys remain unmeasured. The gallery is qualitative evidence, not a moving-video benchmark. More training, difficult scenes, repeated seeds and independent real-road evaluation are needed.')
    heading('5. Conclusion')
    text(f'One Mapillary dataset supports the detection, crop-classification and exported-inference workflow. The study compares three classifiers and detector settings, saves actual output and verifies ONNX inference. The selected classifier scores {chosen["test"]["accuracy"]*100:.2f}% on 425 annotated crops; local detector mAP50 is {detection["detection"]["mAP50"]:.4f}. These answer different questions. Small-sign detection remains imperfect; the study does not establish autonomous-driving reliability.')
    text('References and Evidence','Heading2')
    text('[1] Stallkamp et al., GTSRB, IJCNN 2011; Houben et al., GTSDB, IJCNN 2013 (related work only, not project data). [2] Ertler et al., MTSD, ECCV 2020. mapillary.com/dataset/trafficsign. [3] Tabernik and Skocaj, IEEE T-ITS 2019. vicos.si/resources/dfg/. [4] Akyon et al., SAHI, ICIP 2022. github.com/obss/sahi. [5] github.com/ayushgayakwad/road-sign-recognition. [6] github.com/pedramebd/traffic-sign-detection-recognition. Repositories reviewed 28 October 2026. Evidence: report/mtsd_classifier_results/, mtsd_results/, mtsd_onnx_results/, mtsd_annotation_review/.','BodyText')
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8)
        canvas.setFillColor(colors.HexColor('#53605e'))
        canvas.drawString(46,27,'Traffic Sign Detection and Classification Using Real Road Images')
        canvas.drawRightString(A4[0]-46,27,str(doc.page))
    SimpleDocTemplate(str(report/'technical_report.pdf'),pagesize=A4,leftMargin=46,rightMargin=46,topMargin=42,bottomMargin=43,title='Traffic Sign Detection and Classification Using Real Road Images',author='Mannat Varshney').build(story,onFirstPage=footer,onLaterPages=footer)
    content='\n\n'.join(sections)
    (report/'technical_report.md').write_text(content)
    (report/'technical_report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Traffic Sign Capstone Technical Report</title><style>body{font:16px/1.6 Arial,sans-serif;max-width:920px;margin:40px auto;padding:0 20px;color:#202923}table{border-collapse:collapse;width:100%;margin:16px 0}td{padding:8px;border-bottom:1px solid #cfd8d1}tr:first-child{background:#e8eeeb;font-weight:bold}hr{margin:40px 0}img{display:block;margin:20px auto}</style><body>'+''.join('<div>'+part+'</div>' for part in sections)+'</body></html>')
    public_comparison={k:v for k,v in comparison.items() if k not in {'selected_checkpoint'}}
    public_comparison['experiments']=[{k:v for k,v in row.items() if k!='checkpoint'} for row in comparison['experiments']]
    (report/'detector_comparison.json').write_text(json.dumps(public_comparison,indent=2))
    print('Generated technical report from completed real-data experiments.')


if __name__ == '__main__':
    main()
