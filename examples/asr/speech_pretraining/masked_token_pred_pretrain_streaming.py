# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import lightning.pytorch as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models.ssl_models import (
    EncDecDenoiseMaskedTokenPredModel,
    EncDecMaskedTokenPredDualModeModel,
    EncDecMaskedTokenPredModel,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

"""
Self-supervised masked-token-prediction (BEST-RQ) pre-training, with selectable model variant
for the streaming-SSL experiment. Pick the variant with the top-level config field
``ssl_model_class`` (default: masked_token_pred):

  masked_token_pred         -> EncDecMaskedTokenPredModel          (non-denoise; A offline / B0 streaming)
  masked_token_pred_dualmode-> EncDecMaskedTokenPredDualModeModel  (proposed streaming BEST-RQ)
  denoise_masked_token_pred -> EncDecDenoiseMaskedTokenPredModel   (original NEST denoise recipe)

Example (proposed streaming dual-mode):
  python masked_token_pred_pretrain_streaming.py \
    --config-path=../conf/ssl/nest --config-name=nest_fast-conformer_dualmode_streaming \
    model.train_ds.manifest_filepath=<libriheavy_train.json> \
    model.validation_ds.manifest_filepath=<dev.json>
"""

SSL_MODEL_CLASSES = {
    "masked_token_pred": EncDecMaskedTokenPredModel,
    "masked_token_pred_dualmode": EncDecMaskedTokenPredDualModeModel,
    "denoise_masked_token_pred": EncDecDenoiseMaskedTokenPredModel,
}


@hydra_runner(config_path="../conf/ssl/nest", config_name="nest_fast-conformer_streaming")
def main(cfg):
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    model_key = cfg.get("ssl_model_class", "masked_token_pred")
    if model_key not in SSL_MODEL_CLASSES:
        raise ValueError(f"Unknown ssl_model_class={model_key}; choose one of {list(SSL_MODEL_CLASSES)}")
    model_class = SSL_MODEL_CLASSES[model_key]
    logging.info(f"Instantiating SSL model: {model_class.__name__} (ssl_model_class={model_key})")

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = model_class(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(asr_model)


if __name__ == "__main__":
    main()
