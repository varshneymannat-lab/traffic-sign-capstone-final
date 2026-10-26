from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import classification_report, confusion_matrix
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import DetMetrics, box_iou

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from src.utils.onnx_classifier import CropClassifier
from src.utils.scene_detector import SceneDetector


def count_matches(pred_classes, true_classes, overlap):
    validator = SimpleNamespace(iouv=torch.linspace(0.5, 0.95, 10))
    return DetectionValidator.match_predictions(validator, pred_classes, true_classes, overlap).numpy()


def fixed_metrics(pred_classes, true_classes, overlap):
    correct = count_matches(pred_classes, true_classes, overlap)
    tp = int(correct[:, 0].sum())
    fp, fn = len(pred_classes)-tp, len(true_classes)-tp
    precision, recall = tp/max(1, tp+fp), tp/max(1, tp+fn)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall,
            'f1': 2*precision*recall/max(1e-12, precision+recall)}


def aggregate_counts(records):
    tp, fp, fn = [sum(r[k] for r in records) for k in ['tp', 'fp', 'fn']]
    p, r = tp/max(1, tp+fp), tp/max(1, tp+fn)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': p, 'recall': r,
            'f1': 2*p*r/max(1e-12,p+r)}


def main():
    parser = argparse.ArgumentParser(description='Evaluate full-scene detector and optional ONNX crop refinement on held-out real images.')
    parser.add_argument('--data-dir', type=Path, default=PROJECT / 'data/mtsd')
    parser.add_argument('--comparison', type=Path, default=PROJECT / 'outputs/mtsd_detection/comparison.json')
    parser.add_argument('--model', type=Path, help='Override with exported detector ONNX to verify deployment.')
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'report/mtsd_results')
    parser.add_argument('--demo-dir', type=Path, default=PROJECT / 'demo/mtsd')
    parser.add_argument('--classifier-dir', type=Path, default=PROJECT / 'models/gtsrb')
    parser.add_argument('--confidence', type=float)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new results directory to preserve previous evaluations.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.demo_dir.mkdir(parents=True, exist_ok=True)
    comparison = json.loads(args.comparison.read_text())
    operating_point=json.loads((args.comparison.parent/'operating_point.json').read_text())
    if args.confidence is None:
        args.confidence=operating_point['confidence']
    summary = json.loads((args.data_dir / 'dataset_summary.json').read_text())
    classes = summary['classes']
    torch.set_num_threads(2)
    model_path = args.model or Path(comparison['selected_checkpoint'])
    if not model_path.is_file() and not args.model:
        model_path = args.comparison.parent / comparison['selected_experiment'] / 'weights/best.pt'
    detector = SceneDetector(YOLO(str(model_path), task='detect'),comparison['selected_image_size'],operating_point.get('inference_mode','whole'))
    classifier = CropClassifier(args.classifier_dir)
    if classifier.classes != classes:
        raise ValueError('Classifier and detector class order do not match.')
    size = comparison['selected_image_size']
    metrics = DetMetrics(names=dict(enumerate(classes)))
    with (args.data_dir / 'annotations_split.csv').open() as stream:
        annotations = list(csv.DictReader(stream))
    gt = {}
    for row in annotations:
        if row['split'] == 'test':
            gt.setdefault(row['image_path'], []).append(row)
    first_path = args.data_dir / next(iter(gt))
    detector.predict(str(first_path), imgsz=size, device='cpu', conf=0.001, iou=0.7, rect=False, verbose=False)
    classifier(np.zeros((64,64,3),dtype=np.uint8))
    direct, refined, predictions, timings = [], [], [], []
    actual, before, after = [], [], []
    preview_counts = {c:0 for c in classes}
    for number, (relative, targets) in enumerate(sorted(gt.items())):
        start = time.perf_counter()
        image = cv2.imread(str(args.data_dir / relative))
        if image is None:
            raise ValueError('Could not read image: '+relative)
        h,w = image.shape[:2]
        result = detector.predict(image, imgsz=size, device='cpu', conf=0.001, iou=0.7, max_det=100, rect=False, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu()
        pred_classes = result.boxes.cls.cpu()
        scores = result.boxes.conf.cpu()
        gt_classes = torch.tensor([classes.index(r['sign_label']) for r in targets])
        gt_boxes = []
        for row in targets:
            x,y,bw,bh = [float(row[k]) for k in ['x_center','y_center','box_width','box_height']]
            gt_boxes.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
        overlap = box_iou(torch.tensor(gt_boxes), boxes)
        true_positive = count_matches(pred_classes, gt_classes, overlap)
        keep = scores >= args.confidence
        accepted_boxes = boxes[keep]
        accepted_classes = pred_classes[keep]
        accepted_scores = scores[keep]
        updated_classes = accepted_classes.clone()
        crop_scores = []
        for i, box in enumerate(accepted_boxes):
            x1,y1,x2,y2 = box.round().int().tolist()
            x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w,x2),min(h,y2)
            if x2 <= x1 or y2 <= y1:
                crop_scores.append(None)
                continue
            label, probability = classifier(image[y1:y2,x1:x2])
            updated_classes[i] = label
            crop_scores.append(probability)
        elapsed = time.perf_counter()-start
        timings.append(elapsed)
        accepted_overlap = overlap[:,keep]
        direct.append(fixed_metrics(accepted_classes,gt_classes,accepted_overlap))
        refined.append(fixed_metrics(updated_classes,gt_classes,accepted_overlap))
        if len(accepted_boxes) and len(gt_boxes):
            ious = accepted_overlap.numpy()
            truth_ids,pred_ids = linear_sum_assignment(-np.where(ious >= 0.5,ious,0))
            for truth_id,pred_id in zip(truth_ids,pred_ids):
                if ious[truth_id,pred_id] >= 0.5:
                    actual.append(int(gt_classes[truth_id]))
                    before.append(int(accepted_classes[pred_id]))
                    after.append(int(updated_classes[pred_id]))
        metrics.update_stats({'tp': true_positive, 'conf':scores.numpy(), 'pred_cls':pred_classes.numpy(),
                              'target_cls':gt_classes.numpy(), 'target_img':np.unique(gt_classes.numpy()), 'im_name':relative})
        predictions.append({'image':relative, 'ground_truth':targets,
                            'predicted_boxes':boxes.tolist(),'detector_classes':pred_classes.int().tolist(),
                            'detector_scores':scores.tolist(),'accepted_refined_classes':updated_classes.int().tolist(),
                            'crop_scores':crop_scores,'processing_seconds':elapsed})
        present = {row['sign_label'] for row in targets}
        if any(preview_counts[c] < 2 for c in present):
            for mode, labels in [('detector',accepted_classes),('refined',updated_classes)]:
                annotated = image.copy()
                for box,label,score in zip(accepted_boxes,labels,accepted_scores):
                    x1,y1,x2,y2 = box.round().int().tolist()
                    cv2.rectangle(annotated,(x1,y1),(x2,y2),(30,210,80),2)
                    cv2.putText(annotated,f'{classes[int(label)]} {float(score):.2f}',(max(0,x1),max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.55,(30,210,80),2)
                cv2.imwrite(str(args.demo_dir / (Path(relative).stem+'_'+mode+'.jpg')),annotated)
            for category in present:
                preview_counts[category] += 1
        if (number+1) % 50 == 0:
            print(f'Evaluated {number+1}/{len(gt)} held-out images',flush=True)
    metrics.process(save_dir=args.output_dir,plot=True)
    def crop_record(predicted):
        return {'samples':len(actual), 'accuracy':float(np.mean(np.array(actual)==np.array(predicted))) if actual else None,
                'classification_report':classification_report(actual,predicted,labels=list(range(5)),target_names=classes,zero_division=0,output_dict=True),
                'confusion_matrix':confusion_matrix(actual,predicted,labels=list(range(5))).tolist()}
    results = {'model_format':model_path.suffix,'test_images':len(gt),'classes':classes,
               'inference_mode':detector.mode,
               'detection':{'mAP50':float(metrics.box.map50),'mAP50_95':float(metrics.box.map),
                            'precision_at_max_f1':float(metrics.box.mp),'recall_at_max_f1':float(metrics.box.mr),
                            'per_class_AP50':metrics.box.ap50.tolist(),'per_class_AP50_95':metrics.box.ap.tolist(),
                            'class_indices':metrics.box.ap_class_index.tolist()},
               'fixed_threshold':{'confidence':args.confidence,'iou':0.5,'detector_only':aggregate_counts(direct),'detector_plus_classifier':aggregate_counts(refined)},
               'matched_box_classification':{'matching':'Class-agnostic one-to-one maximum-IoU assignment, IoU >=0.5; missed signs excluded only from this crop metric.',
                                             'detector_labels':crop_record(before),'refined_labels':crop_record(after)},
               'timing':{'mean_ms':float(np.mean(timings)*1000),'median_ms':float(np.median(timings)*1000),'p95_ms':float(np.percentile(timings,95)*1000),
                         'throughput_images_per_second':len(timings)/sum(timings),'scope':'Warm image read, detector preprocess/inference/NMS, ONNX crop classification. Excludes model loading, metric computation and output encoding; not continuous-video FPS.'},
               'hardware':{'platform':platform.platform(),'device':'cpu','torch_threads':2,'classifier_onnx_threads':2,
                           'detector_onnx_threads':'ONNX Runtime default through Ultralytics' if model_path.suffix=='.onnx' else None},
               'test_scope':summary['split_policy'], 'limitations':'Selected nonpanoramic positive scenes; difficult target scenes excluded. Crop classifier was trained on GTSRB, not the global MTSD domain. No real-time driving claim.'}
    (args.output_dir / 'evaluation.json').write_text(json.dumps(results,indent=2))
    (args.output_dir / 'predictions.json').write_text(json.dumps(predictions,indent=2))
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
