"""
finetune_canary_qwen.py
───────────────────────
Fine-tuning script for nvidia/canary-qwen-2.5b using NeMo SpeechLM2.

Checkpoint loading flow:
    canary-qwen-2.5b.nemo  (saved via convert.py)
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

import tarfile
import tempfile
from pathlib import Path

import torch
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.speechlm2.models import SALM
from nemo.collections.speechlm2.parts.pretrained import resolve_nemo_path
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_state_dict_from_ckpt(ckpt_path: str) -> dict:
    """
    Load state_dict from a checkpoint.

    Supported formats:
      - .nemo  : tarball produced by convert.py (model_weights.ckpt inside)
      - .ckpt  : plain torch.save(model.state_dict(), path)
    """
    logging.info(f"Loading checkpoint: {ckpt_path}")

    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run convert.py first to prepare canary-qwen-2.5b.nemo"
        )

    if path.suffix == ".nemo":
        return _load_state_dict_from_nemo(ckpt_path)

    # Plain .ckpt — raw state dict saved with torch.save
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unrecognised checkpoint format in {ckpt_path}")
    logging.info(f"  Format : plain state_dict  ({len(state_dict)} keys)")
    return state_dict


def _load_state_dict_from_nemo(nemo_path: str) -> dict:
    """Extract and load model_weights.ckpt from a .nemo tarball."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(nemo_path) as tar:
            tar.extract("model_weights.ckpt", path=tmpdir, filter="data")
        weights_path = Path(tmpdir) / "model_weights.ckpt"
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    logging.info(f"  Format : .nemo  ({len(state_dict)} keys)")
    return state_dict


def _load_encoder_keys_from_nemo(nemo_path_or_id: str) -> dict:
    """
    Load encoder.* and preprocessor.* keys from a .nemo file.

    Accepts:
      - Local .nemo path          : /ckpts/canary-1b-v2.nemo
      - HuggingFace Hub model ID  : nvidia/canary-1b-v2  (auto-downloaded)
      - NGC model name            : stt_en_fastconformer_hybrid_large_streaming_80ms

    Works with any NeMo ASR checkpoint, _asr_encoder.nemo (convert_modules.py),
    or SALM _perception.nemo (encoder/preprocessor keys are selected).
    """
    local_path = resolve_nemo_path(nemo_path_or_id)
    full_sd = _load_state_dict_from_nemo(local_path)
    encoder_sd = {
        k: v for k, v in full_sd.items()
        if k.startswith("encoder.") or k.startswith("preprocessor.")
    }
    if not encoder_sd:
        raise ValueError(
            f"No encoder.* / preprocessor.* keys found in {nemo_path_or_id}.\n"
            "Expected a NeMo ASR .nemo or a file produced by:\n"
            "  python convert_modules.py --from-asr <asr.nemo>"
        )
    return encoder_sd


def apply_module_weights(model: "SALM", module_weights_cfg) -> None:
    """
    Load per-module .nemo checkpoints and apply them on top of the model.

    Supported config keys and their loading behaviour:

      llm               → model.llm          (prefix-stripped keys)
      embed_tokens      → model.embed_tokens  (prefix-stripped keys)
      perception        → model.perception    (prefix-stripped keys, full module)
      perception_from_asr
                        → model.perception    (encoder.* + preprocessor.* only,
                                               from any NeMo ASR or extracted
                                               _asr_encoder.nemo file)

    Priority:
      module_weights.X  >  canary_qwen_pretrained_path  >  random init

    perception_from_asr is skipped when perception is also set
    (full module override takes precedence).
    """
    if module_weights_cfg is None:
        return

    # ── standard prefix-stripped modules ──────────────────────────────────────
    module_map = {
        "llm":          "llm",
        "embed_tokens": "embed_tokens",
        "perception":   "perception",
    }

    for cfg_key, attr_name in module_map.items():
        path = module_weights_cfg.get(cfg_key, None)
        if not path:
            continue

        logging.info(f"  Loading module '{cfg_key}' from: {path}")
        state_dict = _load_state_dict_from_nemo(path)
        submodule = getattr(model, attr_name)
        missing, unexpected = submodule.load_state_dict(state_dict, strict=False)

        if missing:
            logging.warning(f"    [{cfg_key}] missing keys  : {len(missing)}")
            for k in missing[:10]:
                logging.warning(f"      {k}")
        if unexpected:
            logging.warning(f"    [{cfg_key}] unexpected keys: {len(unexpected)}")
            for k in unexpected[:10]:
                logging.warning(f"      {k}")

        logging.info(f"    [{cfg_key}] OK  ({len(state_dict)} keys loaded)")

    # ── encoder from ASR checkpoint ───────────────────────────────────────────
    # Skipped when perception is already fully overridden above.
    asr_path = module_weights_cfg.get("perception_from_asr", None)
    if asr_path and not module_weights_cfg.get("perception", None):
        logging.info(f"  Loading encoder+preprocessor from ASR checkpoint: {asr_path}")
        encoder_sd = _load_encoder_keys_from_nemo(asr_path)

        missing, unexpected = model.perception.load_state_dict(encoder_sd, strict=False)

        if missing:
            # Expected: modality_adapter.* and proj.* are not in ASR checkpoints.
            adapter_missing = [k for k in missing if not (
                k.startswith("modality_adapter.") or k.startswith("proj.")
            )]
            if adapter_missing:
                logging.warning(f"    [perception_from_asr] unexpected missing keys: {len(adapter_missing)}")
                for k in adapter_missing[:10]:
                    logging.warning(f"      {k}")
        if unexpected:
            logging.warning(f"    [perception_from_asr] unexpected keys: {len(unexpected)}")
            for k in unexpected[:10]:
                logging.warning(f"      {k}")

        logging.info(f"    [perception_from_asr] OK  ({len(encoder_sd)} encoder/preprocessor keys loaded)")


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

    # ── 4. Load weights (full checkpoint → then per-module overrides) ────────
    #
    #  Priority per module:
    #    module_weights.X  >  canary_qwen_pretrained_path  >  random init
    #
    ckpt_path        = cfg.model.get("canary_qwen_pretrained_path", None)
    module_weights   = cfg.model.get("module_weights", None)
    any_module_set   = module_weights and any(
        module_weights.get(k) for k in ("llm", "embed_tokens", "perception", "perception_from_asr")
    )

    if ckpt_path is None and not any_module_set:
        raise ValueError(
            "No checkpoint configured.\n"
            "Set at least one of:\n"
            "  model.canary_qwen_pretrained_path=/path/to/canary-qwen-2.5b.nemo\n"
            "  model.module_weights.llm=/path/to/..._llm.nemo\n"
            "  model.module_weights.perception=/path/to/..._perception.nemo"
        )

    # Step 1 – full checkpoint as baseline
    if ckpt_path:
        logging.info(f"Loading full checkpoint: {ckpt_path}")
        state_dict = load_state_dict_from_ckpt(ckpt_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
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

    # Step 2 – per-module overrides (take priority over the full checkpoint)
    if any_module_set:
        logging.info("Applying per-module weight overrides ...")
        apply_module_weights(model, module_weights)

    # ── 5. Log trainable parameter summary ────────────────────────────────
    log_param_stats(model)

    # ── 6. Train ──────────────────────────────────────────────────────────
    logging.info("Starting training ...")
    trainer.fit(model)


if __name__ == "__main__":
    main()
