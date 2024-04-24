import os
from pathlib import Path

import hydra

import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from typing import Dict, List, Optional

from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy

from ranking_lib.lightning_wrappers.utils import load_pretrained_model
from ranking_lib.callbacks.ema_callback import EMACallback
from sys import exit


@hydra.main(version_base=None, config_path='configs', config_name='base')
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)

    if cfg.exp_name == 'None':
        raise ValueError("Specify experiment name - exp_name")

    yaml_cfg = OmegaConf.to_yaml(cfg)
    to_wandb_cfg = OmegaConf.to_container(cfg)
    print(yaml_cfg)

    wrapped_model = instantiate(cfg.lightning)
    if cfg.pretrained_ckpt_path != 'None':
        load_pretrained_model(
            wrapped_model.model,
            ckpt_path=cfg.pretrained_ckpt_path,
            strict=False
        )
    #wrapped_model.model = torch.compile(wrapped_model.model, mode='default')
    #wrapped_model.model = torch.compile(wrapped_model.model, dynamic=True, mode='max-autotune')

    exp_name = cfg.exp_name

    exp_folder = Path(os.environ['EXP_PATH'], exp_name)
    exp_folder.mkdir(parents=True, exist_ok=True)

    with open(Path(exp_folder, 'config.yaml'), 'w') as fout:
        print(yaml_cfg, file=fout)
    with open(Path(exp_folder, 'hparams.yaml'), 'w') as fout:
        print(OmegaConf.to_yaml(cfg.lightning), file=fout)
    #exit(0)

    callbacks=[
        ModelCheckpoint(
            dirpath=exp_folder,
            filename='ndcg-{epoch:02d}-{ndcg@10/valid:.4f}-{mrr@10/valid:.4f}',
            monitor='ndcg@10/valid',
            save_weights_only=False,
            save_top_k=10,
            save_last=True,
            mode='max',
            auto_insert_metric_name=False
        ),
        ModelCheckpoint(
            dirpath=exp_folder,
            filename='mrr-{epoch:02d}-{ndcg@10/valid:.4f}-{mrr@10/valid:.4f}',
            monitor='mrr@10/valid',
            save_weights_only=False,
            save_top_k=10,
            save_last=False,
            mode='max',
            auto_insert_metric_name=False
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    if cfg.ema_decay > 0:
        callbacks += [EMACallback(decay=cfg.ema_decay)]

    # if torch.cuda.device_count() > 1:
    #     strategy=DDPStrategy(
    #         find_unused_parameters=False,
    #     )
    # else:
    #     strategy = 'auto'


    trainer = instantiate(cfg.trainer)(
        logger=WandbLogger(
            project=cfg.project,
            name=exp_name,
            config=to_wandb_cfg
        ),
        log_every_n_steps=50,
        callbacks=callbacks,
        enable_checkpointing=True,
        gradient_clip_algorithm="norm",
        accelerator="gpu",
        val_check_interval=0.1,
        # strategy=strategy,
        sync_batchnorm=True
    )

    resume_ckpt_path = None
    if cfg.resume_ckpt_path != 'None':
        resume_ckpt_path = cfg.resume_ckpt_path

    datamodule=instantiate(cfg.datamodule)
    trainer.fit(
        wrapped_model,
        datamodule=datamodule,
        ckpt_path=resume_ckpt_path
    )

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['OPEN_PATH'] = str(Path('.'))
    os.environ['DATA_PATH'] = str(Path('data'))
    os.environ['EXP_PATH'] = str(Path('experiments/'))
    print(os.environ['DATA_PATH'])
    main()