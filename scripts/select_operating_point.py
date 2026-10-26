from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics,box_iou,smooth

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts.evaluate_detector import count_matches
from src.utils.scene_detector import SceneDetector


def main():
    parser=argparse.ArgumentParser(description='Compare whole/tiled inference and choose confidence using internal validation only.')
    parser.add_argument('--comparison-dir',type=Path,default=PROJECT/'outputs/mtsd_detection')
    parser.add_argument('--data-dir',type=Path,default=PROJECT/'data/mtsd')
    args=parser.parse_args()
    comparison=json.loads((args.comparison_dir/'comparison.json').read_text())
    checkpoint=Path(comparison['selected_checkpoint'])
    if not checkpoint.exists():
        checkpoint=args.comparison_dir/comparison['selected_experiment']/'weights/best.pt'
    classes=json.loads((args.data_dir/'dataset_summary.json').read_text())['classes']
    with (args.data_dir/'annotations_split.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    targets={}
    for row in rows:
        if row['split']=='val':
            targets.setdefault(row['image_path'],[]).append(row)
    torch.set_num_threads(2)
    records=[]
    for mode in ['whole','sliced']:
        detector=SceneDetector(YOLO(str(checkpoint)),comparison['selected_image_size'],mode)
        detector.predict(str(args.data_dir/next(iter(targets))),imgsz=comparison['selected_image_size'],device='cpu',conf=.001,iou=.7,max_det=100,verbose=False)
        metrics=DetMetrics(names=dict(enumerate(classes)))
        times=[]
        for i,(relative,annotations) in enumerate(sorted(targets.items())):
            start=time.perf_counter()
            image=cv2.imread(str(args.data_dir/relative))
            output=detector.predict(image,imgsz=comparison['selected_image_size'],device='cpu',conf=.001,iou=.7,max_det=100,verbose=False)[0]
            times.append(time.perf_counter()-start)
            h,w=image.shape[:2]
            boxes=[]
            for row in annotations:
                x,y,bw,bh=[float(row[k]) for k in ['x_center','y_center','box_width','box_height']]
                boxes.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
            true_classes=torch.tensor([classes.index(r['sign_label']) for r in annotations])
            pred_classes=output.boxes.cls.cpu()
            tp=count_matches(pred_classes,true_classes,box_iou(torch.tensor(boxes),output.boxes.xyxy.cpu()))
            metrics.update_stats({'tp':tp,'conf':output.boxes.conf.cpu().numpy(),'pred_cls':pred_classes.numpy(),
                                  'target_cls':true_classes.numpy(),'target_img':np.unique(true_classes.numpy()),'im_name':relative})
            if (i+1)%100==0:
                print(f'{mode}: evaluated {i+1}/{len(targets)} internal validation scenes',flush=True)
        directory=args.comparison_dir/('inference_validation_'+mode)
        directory.mkdir(exist_ok=True)
        metrics.process(save_dir=directory,plot=True)
        index=int(np.argmax(smooth(metrics.box.f1_curve.mean(0),.1)))
        records.append({'inference_mode':mode,'validation_images':len(targets),
                        'mAP50':float(metrics.box.map50),'mAP50_95':float(metrics.box.map),
                        'confidence':max(.01,float(metrics.box.px[index])),
                        'validation_macro_f1_at_grid_point':float(metrics.box.f1_curve[:,index].mean()),
                        'mean_read_and_detector_ms':float(np.mean(times)*1000)})
        (args.comparison_dir/'inference_comparison.json').write_text(json.dumps(records,indent=2))
    chosen=max(records,key=lambda row:(row['mAP50_95'],-row['mean_read_and_detector_ms']))
    record={**chosen,'nms_iou':.7,'matching_iou':.5,
            'mode_selection':'Highest internal-validation mAP50:95; lower processing time breaks exact ties.',
            'confidence_selection':'Maximum smoothed mean per-class validation F1 over the confidence grid, with a predeclared 0.01 minimum.',
            'split':'internal validation; final local test remains untouched',
            'selected_experiment':comparison['selected_experiment'],'image_size':comparison['selected_image_size'],
            'sliced_settings':{'library':'SAHI 0.11.36','slice_size':640,'overlap':.2,'include_whole_image':True,
                               'postprocess':'class-aware NMS, IoU .7; maximum 100 final predictions'},
            'note':'Image tiles are an inference strategy, not extra training scenes or a new trained architecture.'}
    (args.comparison_dir/'operating_point.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    main()
