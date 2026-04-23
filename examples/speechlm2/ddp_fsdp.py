from pathlib import Path
from typing import Optional, Union

import torch
from torch.profiler import (
    profile,
    schedule,
    ProfilerActivity,
    tensorboard_trace_handler,
)
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy


class TorchProfilerCallback(Callback):
    def __init__(
        self,
        log_dir: str,
        wait: int = 5,
        warmup: int = 2,
        active: int = 6,
        repeat: int = 1,
        profile_memory: bool = True,
        record_shapes: bool = True,
        with_stack: bool = False,
        rank0_only: bool = True,
    ):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.profile_memory = profile_memory
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.rank0_only = rank0_only

        self.prof = None
        self._enabled = True

    def setup(self, trainer, pl_module, stage=None):
        if stage is not None and stage != "fit":
            self._enabled = False
            return

        global_rank = getattr(trainer, "global_rank", 0)

        if self.rank0_only and global_rank != 0:
            self._enabled = False
            return

        out_dir = self.log_dir / f"rank{global_rank}"
        out_dir.mkdir(parents=True, exist_ok=True)

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        self.prof = profile(
            activities=activities,
            schedule=schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
            ),
            on_trace_ready=tensorboard_trace_handler(str(out_dir)),
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
        )

    def on_train_start(self, trainer, pl_module):
        if self._enabled and self.prof is not None:
            self.prof.__enter__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._enabled and self.prof is not None:
            self.prof.step()

    def _close_profiler(self):
        if self.prof is not None:
            self.prof.__exit__(None, None, None)
            self.prof = None

    def on_train_end(self, trainer, pl_module):
        if self._enabled:
            self._close_profiler()

    def on_exception(self, trainer, pl_module, exception):
        if self._enabled:
            self._close_profiler()


def build_strategy(
    strategy_name: str,
) -> Union[str, DDPStrategy, FSDPStrategy]:
    """
    Build a Lightning strategy object for DDP or FSDP.

    Args:
        strategy_name: "ddp" or "fsdp"

    Returns:
        Lightning strategy object or string

    Raises:
        ValueError: if unsupported strategy_name is given
    """
    strategy_name = strategy_name.lower()

    if strategy_name == "ddp":
        return DDPStrategy(find_unused_parameters=False)

    if strategy_name == "fsdp":
        return FSDPStrategy()

    raise ValueError(
        f"Unsupported strategy: {strategy_name}. Expected one of ['ddp', 'fsdp']."
    )


def build_trainer(
    strategy_name: str = "ddp",
    devices: int = 8,
    num_nodes: int = 1,
    precision: str = "bf16-mixed",
    profiler_log_dir: str = "profiler_traces",
) -> Trainer:
    """
    Build a Trainer that supports both DDP and FSDP.

    Notes:
        - DDP and FSDP both work with this interface.
        - For FSDP, do NOT manually inject generic plugins.
        - Let Lightning resolve the proper precision handling.

    Args:
        strategy_name: "ddp" or "fsdp"
        devices: number of GPUs per node
        num_nodes: number of nodes
        precision: e.g. "bf16-mixed", "16-mixed", "32-true"
        profiler_log_dir: directory for torch profiler traces
    """
    prof_cb = TorchProfilerCallback(
        log_dir=profiler_log_dir,
        wait=5,
        warmup=2,
        active=6,
        repeat=1,
        profile_memory=True,
        record_shapes=True,
        with_stack=False,
        rank0_only=True,
    )

    strategy = build_strategy(strategy_name)

    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        callbacks=[prof_cb],
    )
    return trainer


if __name__ == "__main__":
    # Example 1: DDP
    trainer_ddp = build_trainer(
        strategy_name="ddp",
        devices=8,
        num_nodes=1,
        precision="bf16-mixed",
        profiler_log_dir="profiler_traces_ddp",
    )

    # Example 2: FSDP
    trainer_fsdp = build_trainer(
        strategy_name="fsdp",
        devices=8,
        num_nodes=1,
        precision="bf16-mixed",
        profiler_log_dir="profiler_traces_fsdp",
    )
