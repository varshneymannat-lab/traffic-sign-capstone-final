import torch
from lightning import Trainer
from torch.utils.data import DataLoader

from src.models.sign_classifier import TrafficSignClassifierLitModule


def test_lightning_training_and_validation_execute(tmp_path):
    previous=torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        samples=[{'image':torch.randn(3,64,64),'sign_label':torch.tensor(i%5)} for i in range(4)]
        loader=DataLoader(samples,batch_size=2)
        model=TrafficSignClassifierLitModule(sign_classes=['stop','no_entry','yield','speed_limit_30','speed_limit_50'],pretrained=False)
        trainer=Trainer(accelerator='cpu',devices=1,fast_dev_run=True,logger=False,
                        enable_checkpointing=False,enable_progress_bar=False,
                        default_root_dir=tmp_path)
        trainer.fit(model,train_dataloaders=loader,val_dataloaders=loader)
        assert trainer.global_step==1
    finally:
        torch.set_num_threads(previous)
