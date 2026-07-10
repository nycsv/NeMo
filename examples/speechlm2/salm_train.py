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
import os
import tarfile
import tempfile

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import SALM, DataModule, SALMDataset
from nemo.core.config import hydra_runner
from nemo.lightning.pytorch.callbacks import PytorchProfilerCallback
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))



 
def _patch_save_to(model: SALM, cfg) -> None:
    """
    Monkey-patch save_to / restore_from onto a SALM instance so that
    NeMoModelCheckpoint (called at the end of every validation) doesn't crash.
 
    The produced .nemo file is a gzip-tar containing:
      - model_config.yaml   (cfg.model)
      - model_weights.pt    (state_dict)
    """
 
    def save_to(self, save_path: str):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path     = os.path.join(tmp, "model_config.yaml")
            weights_path = os.path.join(tmp, "model_weights.pt")
 
            OmegaConf.save(cfg, cfg_path)
            torch.save(self.state_dict(), weights_path)
 
            with tarfile.open(save_path, "w:gz") as tar:
                tar.add(cfg_path,     arcname="model_config.yaml")
                tar.add(weights_path, arcname="model_weights.pt")
 
    # Bind as an instance method
    import types
    model.save_to = types.MethodType(save_to, model)


def _maybe_enable_gradient_checkpointing(model: SALM, cfg) -> None:
    """
    Enable HuggingFace activation (gradient) checkpointing on the LLM.

    Trades ~20-30% extra compute for a large reduction in activation memory — the
    main lever for full fine-tuning on long audio. Gated by
    ``cfg.model.use_gradient_checkpointing``.

    Only the LLM is checkpointed: NeMo's ConformerEncoder has no native activation
    checkpointing, and the audio encoder is a single non-autoregressive forward that
    is rarely the memory bottleneck. The flag is read at forward time, so enabling it
    here (before Lightning's configure_model applies FSDP2) composes with sharding.
    """
    if not cfg.model.get("use_gradient_checkpointing", False):
        return
    llm = model.llm
    if not hasattr(llm, "gradient_checkpointing_enable"):
        logging.warning("LLM has no gradient_checkpointing_enable(); skipping activation checkpointing.")
        return
    # use_reentrant=False is the robust variant and composes with FSDP2.
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if getattr(llm, "config", None) is not None:
        llm.config.use_cache = False  # required whenever gradient checkpointing is on
    logging.info("Gradient checkpointing enabled on the LLM (use_reentrant=False, use_cache=False).")


@hydra_runner(config_path="conf", config_name="salm")
def train(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    with trainer.init_module():
        model = SALM(OmegaConf.to_container(cfg.model, resolve=True))

    _patch_save_to(model, cfg.model)
    _maybe_enable_gradient_checkpointing(model, cfg)

    profiler_cfg = cfg.get("profiler", None)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if profiler_cfg is not None and profiler_cfg.get("enabled", False) and local_rank == profiler_cfg.get("rank", 0):
        warmup_steps = profiler_cfg.get("warmup_steps", 1)
        active_steps = profiler_cfg.get("active_steps", 3)
        trace_dir = profiler_cfg.get("trace_dir", None) or os.path.join(os.getcwd(), "traces")
        os.makedirs(trace_dir, exist_ok=True)
        profiler_kwargs = OmegaConf.to_container(profiler_cfg.profiler_kwargs) if profiler_cfg.get("profiler_kwargs") else {}
        # repeat=1 prevents infinite schedule cycling (which spams "Disabling Execution Trace Observer")
        profiler_kwargs["schedule"] = torch.profiler.schedule(
            wait=0, warmup=warmup_steps, active=active_steps, repeat=1
        )
        profiler_kwargs["on_trace_ready"] = torch.profiler.tensorboard_trace_handler(trace_dir)
        profiler_callback = PytorchProfilerCallback(
            start_step=profiler_cfg.start_step,
            end_step=profiler_cfg.start_step + warmup_steps + active_steps,
            warmup_steps=warmup_steps,
            active_steps=active_steps,
            trace_dir=trace_dir,
            profiler_kwargs=profiler_kwargs,
        )
        trainer.callbacks.append(profiler_callback)

    dataset = SALMDataset(tokenizer=model.tokenizer)
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
