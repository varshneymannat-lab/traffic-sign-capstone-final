from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

from omegaconf import OmegaConf
import torch
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parents[1]


def metrics_record(metrics):
    return {'precision': float(metrics.box.mp), 'recall': float(metrics.box.mr),
            'mAP50': float(metrics.box.map50), 'mAP50_95': float(metrics.box.map),
            'per_class_AP50': metrics.box.ap50.tolist(),
            'per_class_AP50_95': metrics.box.ap.tolist(),
            'class_indices': metrics.box.ap_class_index.tolist()}


def main():
    parser = argparse.ArgumentParser(description='Compare two lightweight detector resolutions on one fixed real-data split.')
    parser.add_argument('--data-dir', type=Path, default=PROJECT / 'data/mtsd')
    parser.add_argument('--config', type=Path, default=PROJECT / 'pipelines/detection/experiment.yaml')
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'outputs/mtsd_detection')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new output directory to preserve previous experiments.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.threads)
    os.environ['OMP_NUM_THREADS'] = str(config.threads)
    os.environ['MKL_NUM_THREADS'] = str(config.threads)
    data_config = OmegaConf.load(args.data_dir / 'dataset.yaml')
    data_config.path = str(args.data_dir.resolve())
    runtime_config = args.output_dir / 'resolved_dataset.yaml'
    OmegaConf.save(data_config, runtime_config)
    config_data = OmegaConf.to_container(config, resolve=True)
    dataset_summary = json.loads((args.data_dir / 'dataset_summary.json').read_text())
    records = []
    for image_size in config.image_sizes:
        name = f'yolo11n_{image_size}'
        print(f'Training {name}: {config.epochs} epochs, frozen first {config.freeze_backbone_layers} layers', flush=True)
        writer = SummaryWriter(str(args.output_dir / 'tensorboard' / name))
        def log_epoch(trainer):
            for key, value in trainer.metrics.items():
                writer.add_scalar(key, float(value), trainer.epoch)
            for key, value in trainer.label_loss_items(trainer.tloss, prefix='train').items():
                writer.add_scalar(key, float(value), trainer.epoch)
            writer.flush()
        model = YOLO(str(PROJECT / config.model))
        model.add_callback('on_fit_epoch_end', log_epoch)
        start = time.perf_counter()
        try:
            model.train(data=str(runtime_config), epochs=config.epochs, imgsz=image_size,
                        batch=config.batch_size, workers=0, device=config.device, amp=False,
                        freeze=config.freeze_backbone_layers, seed=config.seed, deterministic=True,
                        optimizer='AdamW', lr0=config.learning_rate, weight_decay=config.weight_decay,
                        warmup_epochs=1.0, mosaic=0.0, mixup=0.0, copy_paste=0.0,
                        fliplr=0.0, flipud=0.0, degrees=5.0, translate=0.05, scale=0.2,
                        hsv_h=0.01, hsv_s=0.15, hsv_v=0.2, close_mosaic=0,
                        project=str(args.output_dir.resolve()), name=name, exist_ok=False,
                        plots=True, save=True, verbose=False)
        finally:
            writer.close()
        seconds = time.perf_counter() - start
        checkpoint = args.output_dir / name / 'weights/best.pt'
        best = YOLO(str(checkpoint))
        validation = best.val(data=str(runtime_config), split='val', imgsz=image_size,
                              batch=config.batch_size, workers=0, device=config.device,
                              conf=0.001, iou=0.7, plots=True,
                              project=str(args.output_dir.resolve()), name=name+'_validation', verbose=False)
        record = {'name': name, 'image_size': image_size, 'validation': metrics_record(validation),
                  'training_seconds': seconds, 'checkpoint': str(checkpoint.resolve()),
                  'parameter_count': sum(p.numel() for p in best.model.parameters())}
        records.append(record)
        (args.output_dir / 'validation_comparison.json').write_text(json.dumps(records, indent=2))
    selected = max(records, key=lambda row: (row['validation']['mAP50_95'], -row['image_size']))
    result = {'selected_experiment': selected['name'], 'selected_checkpoint': selected['checkpoint'],
              'selected_image_size': selected['image_size'], 'selection_rule': 'Highest validation mAP50:95; smaller resolution breaks exact ties. No test predictions used.',
              'settings': config_data, 'dataset': dataset_summary, 'experiments': records,
              'hardware': {'platform': platform.platform(), 'device': config.device, 'threads': config.threads},
              'note': 'Same small detector at two input resolutions; not two different detector architectures. Three classifier architectures are compared separately.'}
    (args.output_dir / 'comparison.json').write_text(json.dumps(result, indent=2))
    print('Selected detector: ' + selected['name'], flush=True)


if __name__ == '__main__':
    main()
