from types import SimpleNamespace

import numpy as np
import torch

from src.detect_signs import annotate_frame


def test_custom_five_class_ids_are_used_without_coco_filter():
    boxes=[SimpleNamespace(cls=torch.tensor([category]),conf=torch.tensor([.9]),xyxy=torch.tensor([[2.,25.,40.,60.]])) for category in [0,4]]
    class Detector:
        def predict(self,frame,**kwargs):
            assert kwargs['conf']==.25
            return [SimpleNamespace(boxes=boxes)]
    image=np.zeros((80,100,3),dtype=np.uint8)
    annotated,records=annotate_frame(Detector(),image,640,.25)
    assert [r['class'] for r in records]==['stop','speed_limit_50']
    assert annotated.shape==image.shape
    assert np.any(annotated)


def test_refinement_preserves_original_detector_label():
    box=SimpleNamespace(cls=torch.tensor([0]),conf=torch.tensor([.8]),xyxy=torch.tensor([[2.,25.,40.,60.]]))
    class Detector:
        def predict(self,frame,**kwargs):
            return [SimpleNamespace(boxes=[box])]
    def classifier(crop):
        assert not np.any(crop)
        return 1,.7
    _,records=annotate_frame(Detector(),np.zeros((80,100,3),dtype=np.uint8),640,.25,classifier)
    assert records[0]['detector_class']=='stop'
    assert records[0]['class']=='no_entry'
    assert records[0]['classifier_confidence']==.7
