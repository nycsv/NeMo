# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# salm_model_pt.py
#
# Wraps nemo.collections.speechlm2.SALM with ModelPT so that
# save_to() / restore_from() work as expected.

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch import Trainer

from nemo.collections.speechlm2 import SALM, DataModule, SALMDataset
from nemo.core.classes.modelPT import ModelPT


class SALMModelPT(ModelPT):
    """
    ModelPT wrapper around nemo.collections.speechlm2.SALM.

    Gives SALM full ModelPT serialisation (save_to / restore_from)
    while keeping the speechlm2 training loop untouched.

    Usage
    -----
    Build from config
    ~~~~~~~~~~~~~~~~~
    >>> model = SALMModelPT(cfg=OmegaConf.load("conf/salm.yaml"), trainer=trainer)

    Save
    ~~~~
    >>> model.save_to("my_salm.nemo")

    Restore
    ~~~~~~~
    >>> model = SALMModelPT.restore_from("my_salm.nemo", trainer=trainer)
    """

    def __init__(self, cfg: DictConfig, trainer: Optional[Trainer] = None):
        # ModelPT expects cfg to be a DictConfig that contains at least a
        # top-level key; we store the whole config under self._cfg.
        # If the caller passes the *full* training config (with model / data /
        # trainer sub-keys), pull out the model sub-config.
        if "model" in cfg:
            model_cfg = cfg.model
        else:
            model_cfg = cfg  # assume cfg is already the model sub-config

        # ModelPT.__init__ saves cfg to self._cfg
        super().__init__(cfg=model_cfg, trainer=trainer)

        # Build the inner SALM (the real implementation)
        with trainer.init_module() if trainer is not None else _nullctx():
            self.salm = SALM(OmegaConf.to_container(model_cfg, resolve=True))

    # ------------------------------------------------------------------
    # Forward – delegate to inner SALM
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        return self.salm(*args, **kwargs)

    # ------------------------------------------------------------------
    # Training / validation steps – delegate to inner SALM
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        return self.salm.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.salm.validation_step(batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self):
        return self.salm.on_validation_epoch_end()

    def configure_optimizers(self):
        # Delegate to SALM if it defines its own optimizer, otherwise fall
        # back to ModelPT's optimizer setup driven by cfg.optim.
        if hasattr(self.salm, "configure_optimizers"):
            return self.salm.configure_optimizers()
        return super().configure_optimizers()

    # ------------------------------------------------------------------
    # ModelPT abstract methods (required to instantiate the class)
    # ------------------------------------------------------------------
    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        """
        Called by ModelPT during restore_from / fit.
        Build and attach the training dataloader.
        Pass the 'data' sub-config here, or override as needed.
        """
        dataset = SALMDataset(tokenizer=self.salm.tokenizer)
        dm = DataModule(
            train_data_config,
            tokenizer=self.salm.tokenizer,
            dataset=dataset,
        )
        self._train_dl = dm.train_dataloader()

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        """
        Called by ModelPT during restore_from / fit.
        Build and attach the validation dataloader.
        """
        dataset = SALMDataset(tokenizer=self.salm.tokenizer)
        dm = DataModule(
            val_data_config,
            tokenizer=self.salm.tokenizer,
            dataset=dataset,
        )
        self._validation_dl = dm.val_dataloader()

    @classmethod
    def list_available_models(cls) -> List[str]:
        """Return pretrained model names (mirrors SALM.list_available_models)."""
        if hasattr(SALM, "list_available_models"):
            return SALM.list_available_models()
        return []

    # ------------------------------------------------------------------
    # Convenience: expose the inner SALM's generate / tokenizer
    # ------------------------------------------------------------------
    @property
    def tokenizer(self):
        return self.salm.tokenizer

    def generate(self, *args, **kwargs):
        return self.salm.generate(*args, **kwargs)

    # ------------------------------------------------------------------
    # State-dict helpers so save_to packs the inner SALM weights correctly
    # ------------------------------------------------------------------
    def state_dict(self, *args, **kwargs):
        # Store the SALM weights under the "salm" prefix so that
        # restore_from can load them back into self.salm.
        return {"salm." + k: v for k, v in self.salm.state_dict(*args, **kwargs).items()}

    def load_state_dict(self, state_dict, strict=True):
        # Strip the "salm." prefix added in state_dict() above.
        inner_sd = {}
        for k, v in state_dict.items():
            if k.startswith("salm."):
                inner_sd[k[len("salm."):]] = v
            else:
                inner_sd[k] = v
        return self.salm.load_state_dict(inner_sd, strict=strict)


# ---------------------------------------------------------------------------
# Tiny null-context helper so we can write
#   with trainer.init_module() if trainer else _nullctx():
# without importing contextlib at the call site.
# ---------------------------------------------------------------------------
from contextlib import contextmanager

@contextmanager
def _nullctx():
    yield
