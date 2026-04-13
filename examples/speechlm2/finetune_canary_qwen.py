"""
finetune_canary_qwen.py
───────────────────────
Fine-tuning script for nvidia/canary-qwen-2.5b using NeMo SpeechLM2.

Checkpoint loading flow:
    canary-qwen-2.5b.ckpt  (saved via torch.save with state_dict + config)
        └─▶  SALM model initialised from YAML config
        └─▶  state_dict loaded with strict=False
        └─▶  trainer.fit()

Usage:
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \
        finetune_canary_qwen.py \
        --config-path=. \
        --config-name=canary_qwen_finetune

Reference:
    https://huggingface.co/nvidia/canary-qwen-2.5b/discussions/13
"""

from pathlib import Path

import torch
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.speechlm2.models import SALM
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_state_dict_from_ckpt(ckpt_path: str) -> dict:
    """
    Load state_dict from a .ckpt file saved as:
        torch.save({"state_dict": model.state_dict(), "config": model.cfg}, path)

    Also handles plain state_dict files (torch.save(model.state_dict(), path)).
    """
    logging.info(f"Loading checkpoint: {ckpt_path}")

    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run save_checkpoint.py first to prepare canary-qwen-2.5b.ckpt"
        )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Case 1: {"state_dict": ..., "config": ...}  ← our preferred format
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        logging.info("  Format : {state_dict + config}")

    # Case 2: plain state_dict  ← torch.save(model.state_dict(), path)
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state_dict = ckpt
        logging.info("  Format : plain state_dict")

    else:
        raise ValueError(
            f"Unrecognised checkpoint format in {ckpt_path}.\n"
            "Expected: torch.save({'state_dict': ..., 'config': ...}, path)"
        )

    logging.info(f"  Keys   : {len(state_dict)}")
    return state_dict


def log_param_stats(model: SALM) -> None:
    """Print trainable vs frozen parameter counts, broken down per top-level module."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    logging.info("─" * 60)
    logging.info("Parameter summary")
    logging.info(f"  Total     : {total     / 1e9:.3f}B")
    logging.info(f"  Trainable : {trainable / 1e6:.1f}M  ({100 * trainable / total:.2f}%)")
    logging.info(f"  Frozen    : {frozen    / 1e9:.3f}B  ({100 * frozen    / total:.2f}%)")
    logging.info("─" * 60)
    logging.info("Per-module breakdown:")

    seen = set()
    for name, _ in model.named_parameters():
        top = name.split(".")[0]
        if top in seen:
            continue
        seen.add(top)

        mod_total     = sum(p.numel() for n, p in model.named_parameters() if n.startswith(top))
        mod_trainable = sum(p.numel() for n, p in model.named_parameters() if n.startswith(top) and p.requires_grad)
        tag = "TRAINABLE" if mod_trainable > 0 else "frozen"
        logging.info(f"  {top:<30} {mod_total / 1e6:>8.1f}M   [{tag}]")

    logging.info("─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

@hydra_runner(config_path=".", config_name="canary_qwen_finetune")
def main(cfg: DictConfig) -> None:

    logging.info("=" * 60)
    logging.info("  canary-qwen-2.5b  |  Fine-tuning with FSDP")
    logging.info("=" * 60)
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # ── 1. Trainer ────────────────────────────────────────────────────────
    trainer = pl.Trainer(**cfg.trainer)

    # ── 2. Experiment manager (checkpointing / TensorBoard / W&B) ─────────
    exp_manager(trainer, cfg.get("exp_manager", None))

    # ── 3. Build SALM model skeleton from YAML config ─────────────────────
    #
    #   pretrained_weights: False
    #     → Architecture initialised with random weights.
    #       canary-qwen-2.5b.ckpt is injected manually below.
    #       Use this for fine-tuning the released checkpoint.
    #
    #   pretrained_weights: True
    #     → SALM pulls pretrained_llm + pretrained_asr from HuggingFace Hub
    #       and assembles them automatically.
    #       Use this only when training a new model from scratch.
    #
    logging.info("Building SALM model skeleton from config ...")
    model = SALM(cfg=cfg.model, trainer=trainer)

    # ── 4. Inject canary-qwen-2.5b.ckpt weights ───────────────────────────
    ckpt_path = cfg.model.get("canary_qwen_pretrained_path", None)

    if ckpt_path is None:
        raise ValueError(
            "model.canary_qwen_pretrained_path is not set.\n"
            "Add it to canary_qwen_finetune.yaml or pass on the CLI:\n"
            "  model.canary_qwen_pretrained_path=/path/to/canary-qwen-2.5b.ckpt"
        )

    state_dict = load_state_dict_from_ckpt(ckpt_path)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # Report loading results
    logging.info(f"  Missing keys   : {len(missing)}")
    logging.info(f"  Unexpected keys: {len(unexpected)}")

    if missing:
        logging.warning("Missing keys — will be randomly initialised:")
        for k in missing[:20]:
            logging.warning(f"    {k}")
        if len(missing) > 20:
            logging.warning(f"    ... and {len(missing) - 20} more")

    if unexpected:
        logging.warning("Unexpected keys — ignored:")
        for k in unexpected[:10]:
            logging.warning(f"    {k}")

    # ── 5. Log trainable parameter summary ────────────────────────────────
    log_param_stats(model)

    # ── 6. Train ──────────────────────────────────────────────────────────
    logging.info("Starting training ...")
    trainer.fit(model)


if __name__ == "__main__":
    main()
