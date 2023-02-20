#!/usr/bin/env/python3
"""
Authors
 * Ge Li 2022
"""
from whale.data_io.data_loader import WhaleDataModule
import torch
from whale.models import UNet
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

seed_everything(1234)

model = UNet(n_channels=1, n_classes=2)
whale_dm = WhaleDataModule(
    data_dir="/network/projects/aia/whale_call/LABELS/FWC_1CH", batch_size=32
)

trainer = Trainer(
    max_epochs=1,
    accelerator="gpu",
    devices=1 if torch.cuda.is_available() else None,
    logger=CSVLogger(save_dir="logs/"),
    callbacks=[LearningRateMonitor(logging_interval="step")],
)

trainer.fit(model, whale_dm)
trainer.test(model, datamodule=whale_dm)
