from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def letterbox(image,width,height):
    scale=min(width/image.shape[1],height/image.shape[0])
    resized=cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)))
    canvas=np.full((height,width,3),245,dtype=np.uint8)
    x,y=(width-resized.shape[1])//2,(height-resized.shape[0])//2
    canvas[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    return canvas


def main():
    parser=argparse.ArgumentParser(description='Make a transparently labelled real-photo gallery for the video inference demonstration.')
    parser.add_argument('--data-dir',type=Path,default=Path('data/mtsd'))
    parser.add_argument('--output-dir',type=Path,default=Path('demo/mtsd_gallery'))
    parser.add_argument('--combine-processed',type=Path,help='After running src.detect_signs on input.mp4, supply its annotated.mp4 for side-by-side output.')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.combine_processed:
        original=cv2.VideoCapture(str(args.output_dir/'input.mp4'))
        processed=cv2.VideoCapture(str(args.combine_processed))
        if not original.isOpened() or not processed.isOpened():
            raise ValueError('Both original and actually processed videos are required.')
        fps=original.get(cv2.CAP_PROP_FPS)
        writer=cv2.VideoWriter(str(args.output_dir/'real_image_gallery.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,440))
        if not writer.isOpened():
            raise ValueError('Cannot encode the gallery video.')
        frames=0
        try:
            while True:
                ok,left=original.read()
                other,right=processed.read()
                if ok!=other:
                    raise ValueError('Processed and input videos have different frame counts.')
                if not ok:
                    break
                canvas=np.full((440,1280,3),255,dtype=np.uint8)
                canvas[50:410,:640]=letterbox(left,640,360)
                canvas[50:410,640:]=letterbox(right,640,360)
                cv2.putText(canvas,'Input: held-out real photograph',(20,32),cv2.FONT_HERSHEY_SIMPLEX,.65,(35,50,40),2)
                cv2.putText(canvas,'Output: trained ONNX detector',(660,32),cv2.FONT_HERSHEY_SIMPLEX,.65,(35,50,40),2)
                cv2.putText(canvas,'Real-image gallery: repeated still photographs, not continuous driving footage',(20,432),cv2.FONT_HERSHEY_SIMPLEX,.55,(35,50,40),1)
                writer.write(canvas)
                frames+=1
        finally:
            writer.release()
            original.release()
            processed.release()
        print(f'Combined {frames} actual processed frames.')
        return
    if (args.output_dir/'input.mp4').exists():
        parser.error('Gallery input already exists; use a new output folder.')
    with (args.data_dir/'annotations_split.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    classes=json.loads((args.data_dir/'dataset_summary.json').read_text())['classes']
    choices=[]
    used=set()
    for category in classes:
        paths=sorted({r['image_path'] for r in rows if r['split']=='test' and r['sign_label']==category})
        new=[p for p in paths if p not in used][:2]
        for p in new:
            choices.append({'image_path':p,'selection_category':category,
                            'ground_truth_classes':sorted({r['sign_label'] for r in rows if r['image_path']==p})})
            used.add(p)
    writer=cv2.VideoWriter(str(args.output_dir/'input.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1280,720))
    if not writer.isOpened():
        raise ValueError('Cannot encode MP4 input gallery.')
    try:
        for choice in choices:
            image=cv2.imread(str(args.data_dir/choice['image_path']))
            if image is None:
                raise ValueError('Cannot read selected real image.')
            frame=letterbox(image,1280,720)
            for _ in range(20):
                writer.write(frame)
    finally:
        writer.release()
    provenance={'source':'MTSD v2 fully annotated, selected official-validation scenes reserved for local test',
                'selection':'First two unused sorted test image keys per category, before inspecting prediction correctness.',
                'fps':10,'frames_per_scene':20,'scenes':choices,
                'disclosure':'Repeated real still photographs. Not a continuous-road video or temporal benchmark. Must run the actual inference command to produce annotated output.'}
    (args.output_dir/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print(f'Created {len(choices)*2}-second input gallery from {len(choices)} real held-out photographs.')


if __name__=='__main__':
    main()
