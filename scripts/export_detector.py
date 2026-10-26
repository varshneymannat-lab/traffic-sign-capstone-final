from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description='Export the validation-selected five-class detector and verify actual ONNX output.')
    parser.add_argument('--comparison',type=Path,default=PROJECT / 'outputs/mtsd_detection/comparison.json')
    parser.add_argument('--data-dir',type=Path,default=PROJECT / 'data/mtsd')
    parser.add_argument('--output-dir',type=Path,default=PROJECT / 'models/mtsd')
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new export folder to preserve existing artifacts.')
    comparison=json.loads(args.comparison.read_text())
    classes=json.loads((args.data_dir / 'dataset_summary.json').read_text())['classes']
    size=comparison['selected_image_size']
    checkpoint=Path(comparison['selected_checkpoint'])
    if not checkpoint.is_file():
        checkpoint=args.comparison.parent / comparison['selected_experiment'] / 'weights/best.pt'
    model=YOLO(str(checkpoint))
    if [model.names[i] for i in range(len(model.names))] != classes:
        raise ValueError('Export model labels differ from the prepared dataset.')
    torch.set_num_threads(2)
    exported=Path(model.export(format='onnx',imgsz=size,batch=1,dynamic=False,simplify=False,opset=17,nms=False,device='cpu'))
    onnx.checker.check_model(onnx.load(exported))
    options=ort.SessionOptions()
    options.intra_op_num_threads=2
    options.inter_op_num_threads=1
    session=ort.InferenceSession(str(exported),sess_options=options,providers=['CPUExecutionProvider'])
    model.model.float().eval()
    examples=sorted((args.data_dir / 'images/test').glob('*.jpg'))[:3]
    checks=[]
    for path in examples:
        image=cv2.imread(str(path))
        letterbox=LetterBox(new_shape=(size,size),auto=False)
        resized=letterbox(image=image)
        tensor=np.ascontiguousarray(resized[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255.0
        with torch.inference_mode():
            reference=model.model(torch.from_numpy(tensor))[0].cpu().numpy()
        actual=session.run(None,{session.get_inputs()[0].name:tensor})[0]
        maximum=float(np.max(np.abs(actual-reference)))
        equivalent=bool(np.allclose(actual,reference,rtol=1e-4,atol=0.01))
        checks.append({'image_key':path.stem,'shape':list(actual.shape),'max_abs_difference':maximum,'allclose':equivalent})
        if not equivalent:
            raise ValueError('ONNX outputs failed numerical verification.')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(exported,args.output_dir / 'detector.onnx')
    shutil.copyfile(checkpoint,args.output_dir / 'detector.pt')
    operating_point=json.loads((args.comparison.parent/'operating_point.json').read_text())
    metadata={'classes':classes,'image_size':size,'confidence':operating_point['confidence'],
              'inference_mode':operating_point.get('inference_mode','whole'),
              'sliced_inference':{'library':'SAHI','tile_size':640,'overlap':.2,'include_whole_image':True,'postprocess':'class-aware NMS IoU .7'},
              'architecture':'YOLO11n','selection':comparison['selection_rule'],
              'selected_experiment':comparison['selected_experiment'],'onnx_opset':17,'onnx_batch_size':1,
              'onnx_dynamic':False,'onnx_nms_embedded':False,
              'postprocessing':'Use Ultralytics preprocessing and NMS, or implement equivalent documented postprocessing. Raw ONNX output alone does not draw boxes.',
              'sha256':hashlib.sha256((args.output_dir/'detector.onnx').read_bytes()).hexdigest(),
              'bytes':(args.output_dir/'detector.onnx').stat().st_size,'numerical_checks':checks}
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))


if __name__ == '__main__':
    main()
