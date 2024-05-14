import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch_ema import ExponentialMovingAverage
from typing import Dict, Any

# Cell
class EMACallback(Callback):
    def __init__(self, decay=0.9999, use_ema_weights: bool = True):
        self.decay = decay
        self.ema = None
        self.use_ema_weights = use_ema_weights

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        self.ema = ExponentialMovingAverage(pl_module.parameters(), decay=self.decay)

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule,
        *args, **kwargs
    ):
        self.ema.update(pl_module.parameters())

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        self.store(pl_module.parameters())
        self.copy_to(pl_module.parameters())

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        self.restore(pl_module.parameters())

    def state_dict(self) -> Dict[str, Any]:
        return self.ema.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.ema.load_state_dict(state_dict)
        self.decay = self.ema.decay

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.use_ema_weights:
            self.copy_to(pl_module.parameters())

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema is not None:
            return {"state_dict_ema": self.ema.state_dict()}

    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> None:
        if 'callbacks' in checkpoint and 'EMACallback' in checkpoint['callbacks']:
            self.ema = ExponentialMovingAverage(pl_module.parameters(), 0)
            self.ema.load_state_dict(checkpoint['callbacks']['EMACallback'])

    def store(self, parameters):
        self.ema.store(parameters)

    def restore(self, parameters):
        self.ema.restore(parameters)

    def copy_to(self, parameters):
        self.ema.copy_to(parameters)