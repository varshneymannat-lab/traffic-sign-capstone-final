from __future__ import annotations

import cv2
import torch
from ultralytics.engine.results import Results


class _SquareModel:
    def __init__(self,model):
        self.model=model

    def __getattr__(self,name):
        return getattr(self.model,name)

    def __call__(self,images,**kwargs):
        return self.model.predict(images,rect=False,iou=.7,max_det=100,**kwargs)


class SceneDetector:
    def __init__(self,model,image_size,mode='whole'):
        if mode not in {'whole','sliced'}:
            raise ValueError('Unknown scene inference mode.')
        self.model=model
        self.image_size=image_size
        self.mode=mode
        self.sahi_model=None

    @property
    def names(self):
        return self.model.names

    def predict(self,source,**kwargs):
        if self.mode=='whole':
            return self.model.predict(source,**{**kwargs,'rect':False})
        from sahi.models.ultralytics import UltralyticsDetectionModel
        from sahi.predict import get_sliced_prediction
        image=cv2.imread(str(source)) if not hasattr(source,'shape') else source
        if image is None:
            raise ValueError('Cannot read scene for sliced prediction.')
        confidence=kwargs.get('conf',.25)
        if self.sahi_model is None:
            self.sahi_model=UltralyticsDetectionModel(model=_SquareModel(self.model),
                            confidence_threshold=confidence,device='cpu',image_size=self.image_size)
        self.sahi_model.confidence_threshold=confidence
        prediction=get_sliced_prediction(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),
                    detection_model=self.sahi_model,slice_height=640,slice_width=640,
                    overlap_height_ratio=.2,overlap_width_ratio=.2,perform_standard_pred=True,
                    postprocess_type='NMS',postprocess_match_metric='IOU',
                    postprocess_match_threshold=.7,postprocess_class_agnostic=False,verbose=0)
        rows=[[*obj.bbox.to_xyxy(),float(obj.score.value),int(obj.category.id)] for obj in prediction.object_prediction_list]
        rows=sorted(rows,key=lambda row:row[4],reverse=True)[:100]
        boxes=torch.tensor(rows,dtype=torch.float32).reshape(-1,6)
        return [Results(orig_img=image,path='',names=self.names,boxes=boxes)]
