import torch

from torch import nn, Tensor
from torch.nn import functional as F

import os

from typing import Any, Optional, Dict, List, Union
from lightning import LightningModule

from torchmetrics.retrieval import RetrievalNormalizedDCG, RetrievalMRR
from collections import Counter
from hydra.utils import instantiate
from omegaconf import DictConfig

class BiEncoderWrapper(LightningModule):
    def __init__(
            self,
            model: DictConfig,
            loss: DictConfig,
            optimizer: DictConfig,
            scheduler: DictConfig,
            cosine_scaler: int
    ) -> None:
        super().__init__()
        self.model = instantiate(model)
        self.loss = instantiate(loss)
        self.optimizer_cfg = optimizer
        self.lr_scheduler_cfg = scheduler
        self.valid_metrics = {
            "ndcg@3/valid": RetrievalNormalizedDCG(top_k=3),
            "mrr@3/valid": RetrievalMRR(top_k=3),
            "ndcg@10/valid": RetrievalNormalizedDCG(top_k=10),
            "mrr@10/valid": RetrievalMRR(top_k=10)
        }
        self.cosine_scaler = cosine_scaler

    def training_step(self, batch: Dict[str, Tensor], *args: Any, **kwargs: Any) -> Dict[str, Tensor]:
        query_embs = F.normalize(self.model(**batch["query"]))
        text_embs = F.normalize(self.model(**batch["text"]))
        cosine_similarity = torch.nn.functional.cosine_similarity(query_embs, text_embs, dim=-1)*self.cosine_scaler
        loss = self.loss(cosine_similarity, batch["label"])
        self.log('loss/train', loss)  
        return loss

    def validation_step(self, batch: Dict[str, Tensor], *args: Any, **kwargs: Any) -> Dict[str, Tensor]:
        with torch.no_grad():
            query_embs = F.normalize(self.model(**batch["query"]))
            text_embs = F.normalize(self.model(**batch["text"]))
            cosine_similarity = torch.nn.functional.cosine_similarity(query_embs, text_embs, dim=-1)*self.cosine_scaler
            for metric_name, metric in self.valid_metrics.items():
                if "mrr" in metric_name:
                    metric.update(cosine_similarity, torch.where(batch["label"] > 0.5, 1, 0), batch["qid"])
                else:
                    metric.update(cosine_similarity, batch["label"], batch["qid"])
            loss = self.loss(cosine_similarity, batch["label"])
            self.log('loss/valid', loss)

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.model(**batch)

    def on_validation_epoch_end(self) -> None:
        self.log_dict(
            {
                metric_name: metric.compute()
                for metric_name, metric in self.valid_metrics.items()
            }
        )

    def on_validation_start(self) -> None:
        for metric in self.valid_metrics.values():
            metric.reset()

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer_cfg)(self.model.parameters())
        scheduler = instantiate(self.lr_scheduler_cfg)(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
