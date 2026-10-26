"""Verify the local release, including inference from a freshly extracted model bundle."""

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.restore_inference_models import restore


def main():
    audit=json.loads((ROOT/'report/mtsd_annotation_review/audit.json').read_text())
    if audit['mismatched_scenes']:
        raise ValueError('Unresolved source annotation mismatch.')
    evaluations=[json.loads((ROOT/'report'/name/'evaluation.json').read_text())
                 for name in ['mtsd_results','mtsd_onnx_results']]
    if not all(e['test_images']==332 for e in evaluations):
        raise ValueError('Test results do not cover all held-out scenes.')
    if abs(evaluations[0]['detection']['mAP50']-evaluations[1]['detection']['mAP50'])>1e-3:
        raise ValueError('Exported detection result differs unexpectedly.')
    gallery=ROOT/'demo/mtsd_gallery/real_image_gallery.mp4'
    capture=cv2.VideoCapture(str(gallery))
    frames=0; lowest_std=float('inf')
    while True:
        ok,frame=capture.read()
        if not ok:
            break
        frames+=1
        lowest_std=min(lowest_std,float(np.std(frame)))
        if frame.shape[:2]!=(440,1280):
            raise ValueError('Unexpected gallery dimensions.')
    capture.release()
    if frames!=200 or lowest_std<5:
        raise ValueError('Gallery is truncated or blank.')
    with (ROOT/'demo/mtsd_gallery/processed/detections.csv').open() as stream:
        detections=sum(1 for row in csv.DictReader(stream))
    if not detections:
        raise ValueError('No actual detections in the saved gallery.')
    with tempfile.TemporaryDirectory(prefix='capstone-bundle-check-') as temporary:
        folder=Path(temporary)
        count=restore(ROOT/'artifacts/inference_models.zip',folder)
        subprocess.run([sys.executable,'-m','src.detect_signs',
                        'demo/mtsd_gallery/input.mp4','--model',str(folder/'models/mtsd/detector.onnx'),
                        '--output-dir',str(folder/'inference'),'--max-frames','2'],cwd=ROOT,check=True)
        smoke=json.loads((folder/'inference/summary.json').read_text())
        if smoke['frames']!=2:
            raise ValueError('Fresh model-bundle inference failed.')
    files=['report/technical_report.pdf','report/mtsd_eda/Mapillary_EDA_Report.pdf',
           'artifacts/inference_models.zip','demo/mtsd_gallery/real_image_gallery.mp4']
    record={'checked_at':datetime.now().astimezone().isoformat(),
            'annotation_scene_mismatches':0,'test_images_per_backend':332,
            'gallery_frames_decoded':frames,'gallery_minimum_frame_pixel_std':lowest_std,
            'gallery_detection_rows':detections,'fresh_bundle_files_verified':count,
            'fresh_bundle_inference_frames':smoke['frames'],
            'tensorboard_event_files':len(list((ROOT/'outputs').rglob('events.out.tfevents.*'))),
            'files':{p:{'bytes':(ROOT/p).stat().st_size,'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for p in files},
            'scope':'Local file and inference checks. Not a Docker run, fresh dependency installation, full retraining or GitHub publication verification.'}
    (ROOT/'report/local_verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
