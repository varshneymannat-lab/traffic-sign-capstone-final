from __future__ import annotations

import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts.compare_detectors import metrics_record


def main():
    parser=argparse.ArgumentParser(description='One additional validation-led training phase; final test remains untouched.')
    parser.add_argument('--comparison-dir',type=Path,default=PROJECT/'outputs/mtsd_detection')
    parser.add_argument('--data-dir',type=Path,default=PROJECT/'data/mtsd')
    parser.add_argument('--config',type=Path,default=PROJECT/'pipelines/detection/refinement.yaml')
    parser.add_argument('--device',default='auto',choices=['auto','cpu','mps'])
    args=parser.parse_args()
    if args.device=='auto':
        args.device='mps' if torch.backends.mps.is_available() else 'cpu'
    config=OmegaConf.load(args.config)
    comparison_path=args.comparison_dir/'comparison.json'
    comparison=json.loads(comparison_path.read_text())
    archive=args.comparison_dir/'initial_comparison.json'
    if archive.exists():
        parser.error('A refinement phase is already recorded; do not overwrite it.')
    if args.device=='mps' and not torch.backends.mps.is_available():
        parser.error('MPS GPU is not available in this execution environment.')
    torch.set_num_threads(2)
    checkpoint=Path(comparison['selected_checkpoint'])
    if not checkpoint.is_file():
        checkpoint=args.comparison_dir/comparison['selected_experiment']/'weights/best.pt'
    initialization=comparison['selected_experiment']
    image_size=comparison['selected_image_size']
    name=f'yolo11n_{image_size}_finetuned'
    if (args.comparison_dir/name).exists():
        parser.error('Refinement output already exists.')
    archive.write_text(json.dumps(comparison,indent=2))
    model=YOLO(str(checkpoint))
    writer=SummaryWriter(str(args.comparison_dir/'tensorboard'/name))
    def log_epoch(trainer):
        for key,value in trainer.metrics.items():
            writer.add_scalar(key,float(value),trainer.epoch)
        for key,value in trainer.label_loss_items(trainer.tloss,prefix='train').items():
            writer.add_scalar(key,float(value),trainer.epoch)
        writer.flush()
    model.add_callback('on_fit_epoch_end',log_epoch)
    start=time.perf_counter()
    try:
        model.train(data=str(args.comparison_dir/'resolved_dataset.yaml'),epochs=config.epochs,
                    imgsz=image_size,batch=config.batch_size,workers=0,device=args.device,amp=False,
                    freeze=config.freeze_backbone_layers,seed=config.seed,deterministic=True,
                    optimizer='AdamW',lr0=config.learning_rate,weight_decay=config.weight_decay,
                    warmup_epochs=1.0,mosaic=0.0,mixup=0.0,copy_paste=0.0,
                    fliplr=0.0,flipud=0.0,degrees=5.0,translate=0.05,scale=0.2,
                    hsv_h=0.01,hsv_s=0.15,hsv_v=0.2,close_mosaic=0,
                    project=str(args.comparison_dir.resolve()),name=name,exist_ok=False,
                    plots=True,save=True,verbose=False)
    finally:
        writer.close()
    seconds=time.perf_counter()-start
    best=YOLO(str(args.comparison_dir/name/'weights/best.pt'))
    validation=best.val(data=str(args.comparison_dir/'resolved_dataset.yaml'),split='val',
                        imgsz=image_size,batch=config.batch_size,workers=0,device='cpu',conf=.001,iou=.7,
                        project=str(args.comparison_dir.resolve()),name=name+'_validation',plots=True,verbose=False)
    record={'name':name,'image_size':image_size,'validation':metrics_record(validation),
            'training_seconds':seconds,'checkpoint':str((args.comparison_dir/name/'weights/best.pt').resolve()),
            'parameter_count':sum(p.numel() for p in best.model.parameters()),
            'phase':'additional validation-led fine-tuning, not a new architecture',
            'initialization':initialization,'settings':OmegaConf.to_container(config,resolve=True),
            'hardware':{'platform':platform.platform(),'training_device':args.device,'validation_device':'cpu','torch_threads':2}}
    comparison['experiments'].append(record)
    chosen=max(comparison['experiments'],key=lambda row:(row['validation']['mAP50_95'],-row['image_size']))
    comparison.update({'selected_experiment':chosen['name'],'selected_checkpoint':chosen['checkpoint'],
                       'selected_image_size':chosen['image_size'],
                       'refinement_note':'Initial resolution comparison completed first. Additional phase uses its validation-selected checkpoint. Final selection among recorded candidates uses validation mAP50:95 only.'})
    comparison_path.write_text(json.dumps(comparison,indent=2))
    print(json.dumps(record,indent=2),flush=True)
    print('Final validation-selected checkpoint: '+chosen['name'],flush=True)


if __name__=='__main__':
    main()
