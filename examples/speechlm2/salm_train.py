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

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import SALM, DataModule, SALMDataset
from nemo.core.config import hydra_runner
from nemo.lightning.pytorch.callbacks import PytorchProfilerCallback
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


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
