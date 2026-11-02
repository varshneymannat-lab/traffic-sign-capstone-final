from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from src.utils.onnx_classifier import CropClassifier
from src.utils.scene_detector import SceneDetector
from src.utils.crop_geometry import map_box_to_source


CLASSES = ['stop', 'no_entry', 'yield', 'speed_limit_30', 'speed_limit_50']


def annotate_frame(detector, frame, image_size, confidence, classifier=None, classifier_image=None):
    result = detector.predict(frame, imgsz=image_size, conf=confidence, iou=0.7,
                              max_det=100, device='cpu', rect=False, verbose=False)[0]
    detections = []
    original=(classifier_image if classifier_image is not None else frame.copy()) if classifier else None
    annotator=Annotator(frame,line_width=4)
    for box in result.boxes:
        category = int(box.cls.item())
        score = float(box.conf.item())
        x1,y1,x2,y2 = box.xyxy[0].round().int().tolist()
        predicted_category=category
        classifier_confidence=None
        if classifier:
            height,width=frame.shape[:2]
            clipped=(max(0,x1),max(0,y1),min(width,x2),min(height,y2))
            if clipped[2]>clipped[0] and clipped[3]>clipped[1]:
                cx1,cy1,cx2,cy2=map_box_to_source(clipped,(width,height),(original.shape[1],original.shape[0]))
                crop=original[cy1:cy2,cx1:cx2]
                if crop.size:
                    predicted_category,classifier_confidence=classifier(crop)
        annotator.box_label((x1,y1,x2,y2),f'{CLASSES[predicted_category]} {score:.2f}',color=(25,120,60),txt_color=(255,255,255))
        detections.append({'class':CLASSES[predicted_category],'detector_class':CLASSES[category],
                           'confidence':score,'classifier_confidence':classifier_confidence,
                           'xmin':x1,'ymin':y1,'xmax':x2,'ymax':y2})
    return annotator.result(),detections


def main(argv=None):
    parser = argparse.ArgumentParser(description='Detect five supported signs in real images or videos using the trained detector.')
    parser.add_argument('input',type=Path)
    parser.add_argument('--model',type=Path,default=Path('models/mtsd_revision/detector.onnx'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--image-size',type=int,help='Defaults to exported detector metadata.')
    parser.add_argument('--confidence',type=float)
    parser.add_argument('--max-frames',type=int)
    parser.add_argument('--refine',action='store_true',help='Optionally classify accepted boxes using the Mapillary ONNX crop classifier; boxes and missed signs are unchanged.')
    parser.add_argument('--classifier-dir',type=Path,default=Path('models/mtsd_classifier_original'))
    parser.add_argument('--original-image',type=Path,
                        help='For one prepared still image, crop this corresponding full-resolution original during --refine. Do not supply a different scene.')
    parser.add_argument('--inference-mode',choices=['whole','sliced'],help='Defaults to the validation-selected mode in model metadata.')
    args = parser.parse_args(argv)
    if not args.input.is_file() or not args.model.is_file():
        parser.error('Input and trained model must exist. Restore DVC artifacts or run training/export first.')
    if args.confidence is not None and not 0 < args.confidence < 1:
        parser.error('Confidence must be between zero and one.')
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('Maximum frames must be positive.')
    if args.original_image and (not args.refine or args.input.suffix.lower() not in {'.jpg','.jpeg','.png','.ppm','.bmp'}):
        parser.error('--original-image requires --refine and one still image, not a video.')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new output folder to preserve previous predictions.')
    metadata_path = args.model.parent / 'metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if args.confidence is None:
        args.confidence=metadata.get('confidence',0.25)
    image_size = args.image_size or metadata.get('image_size')
    if not image_size:
        parser.error('Provide image size or use a model folder containing metadata.json.')
    torch.set_num_threads(2)
    model = YOLO(str(args.model),task='detect')
    if [model.names[i] for i in range(len(model.names))] != CLASSES:
        parser.error('Model labels do not match this five-class detector; a generic COCO checkpoint is not supported.')
    inference_mode=args.inference_mode or metadata.get('inference_mode','whole')
    model=SceneDetector(model,image_size,inference_mode)
    classifier=CropClassifier(args.classifier_dir) if args.refine else None
    if classifier and classifier.classes!=CLASSES:
        parser.error('Crop classifier and detector class order do not match.')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    records = []
    start = time.perf_counter()
    frames = 0
    if args.input.suffix.lower() in {'.jpg','.jpeg','.png','.ppm','.bmp'}:
        frame = cv2.imread(str(args.input))
        if frame is None:
            raise ValueError('Cannot decode input image.')
        original=cv2.imread(str(args.original_image),cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION) if args.original_image else None
        if args.original_image and original is None:
            raise ValueError('Cannot decode the original image.')
        annotated,rows = annotate_frame(model,frame,image_size,args.confidence,classifier,original)
        if not cv2.imwrite(str(args.output_dir / 'annotated.jpg'),annotated):
            raise ValueError('Cannot write output image.')
        records.extend({'frame':0,**row} for row in rows)
        frames = 1
    else:
        capture = cv2.VideoCapture(str(args.input))
        if not capture.isOpened():
            raise ValueError('Cannot decode input video.')
        fps = capture.get(cv2.CAP_PROP_FPS) or 25
        width,height = [int(capture.get(k)) for k in [cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT]]
        writer = cv2.VideoWriter(str(args.output_dir / 'annotated.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
        if not writer.isOpened():
            capture.release()
            raise ValueError('Cannot create MP4 output with the available video codec.')
        try:
            while args.max_frames is None or frames < args.max_frames:
                ok,frame = capture.read()
                if not ok:
                    break
                annotated,rows = annotate_frame(model,frame,image_size,args.confidence,classifier)
                writer.write(annotated)
                records.extend({'frame':frames,**row} for row in rows)
                frames += 1
        finally:
            writer.release()
            capture.release()
        if frames == 0:
            raise ValueError('No video frames were decoded.')
    elapsed = time.perf_counter()-start
    with (args.output_dir / 'detections.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['frame','class','detector_class','confidence','classifier_confidence','xmin','ymin','xmax','ymax'])
        writer.writeheader()
        writer.writerows(records)
    summary = {'input':str(args.input),'model':str(args.model),'classes':CLASSES,'frames':frames,
               'pipeline':'detector_plus_classifier' if classifier else 'detector_only',
               'inference_mode':inference_mode,
               'elapsed_seconds':elapsed,'processed_frames_per_second':frames/elapsed,
               'timing_scope':'Includes decoding, prediction, annotation and output encoding; excludes model/backend construction and CSV/JSON serialization.',
               'confidence':args.confidence,'image_size':image_size,'hardware':platform.platform(),
               'note':'Output playback rate is not processing speed. Performance depends on scene and hardware.'}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
