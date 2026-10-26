from __future__ import annotations

import json

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


class CropClassifier:
    def __init__(self,directory):
        metadata=json.loads((directory/'metadata.json').read_text())
        self.classes=metadata['classes']
        self.size=metadata['image_size']
        self.mean=np.array(metadata['normalization_mean'],dtype=np.float32)
        self.std=np.array(metadata['normalization_std'],dtype=np.float32)
        options=ort.SessionOptions()
        options.intra_op_num_threads=2
        options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(directory/'classifier.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        self.input_name=self.session.get_inputs()[0].name

    def __call__(self,crop):
        rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
        image=Image.fromarray(rgb).resize((self.size,self.size),Image.Resampling.BILINEAR)
        array=(np.asarray(image,dtype=np.float32)/255.0-self.mean)/self.std
        tensor=np.ascontiguousarray(array.transpose(2,0,1)[None])
        logits=self.session.run(None,{self.input_name:tensor})[0][0]
        if not np.isfinite(logits).all():
            raise ValueError('Classifier produced nonfinite output.')
        scores=np.exp(logits-logits.max())
        scores/=scores.sum()
        return int(scores.argmax()),float(scores.max())
