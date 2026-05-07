"""
convert_modules.py
──────────────────
Two modes:

  split   (default)
    Split a SALM .nemo checkpoint into per-module .nemo files.

    Accepts a local .nemo path, a HuggingFace Hub ID, or an NGC name.

    Output files (written next to the source file or to --output-dir):
      <stem>_llm.nemo          – model.llm  (Qwen backbone + lm_head)
      <stem>_embed_tokens.nemo – model.embed_tokens  (vocab embedding)
      <stem>_perception.nemo   – model.perception  (encoder + adapter)

    Usage:
      python convert_modules.py canary-qwen-2.5b.nemo
      python convert_modules.py canary-qwen-2.5b.nemo --output-dir /ckpts

  extract-encoder  (--from-asr)
    Extract the FastConformer encoder + preprocessor from any NeMo ASR
    checkpoint and save as a standalone .nemo file for use as
    model.module_weights.perception_from_asr during SALM training.

    Accepts a local .nemo / directory path, a HuggingFace Hub ID (e.g.
    nvidia/canary-1b-v2), or an NGC model name.

    Usage:
      python convert_modules.py --from-asr canary-1b-v2.nemo
      python convert_modules.py --from-asr nvidia/canary-1b-v2
      python convert_modules.py --from-asr nvidia/canary-1b-v2 --output-dir /ckpts

Each output .nemo is a tarball containing:
  model_weights.ckpt  – state_dict with the module prefix stripped
  module_config.yaml  – the matching sub-config slice (when available)
"""

import argparse
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

# Top-level SALM modules and their config key (None = no sub-config).
MODULES = {
    "llm":          None,
    "embed_tokens": None,
    "perception":   "perception",
}

# Keys to keep when extracting encoder from an ASR checkpoint.
_ASR_ENCODER_PREFIXES = ("encoder.", "preprocessor.")


# ─────────────────────────────────────────────────────────────────────────────
#  Low-level I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_source(source: str) -> str:
    """
    Resolve any model identifier to a local path.

    Accepted inputs:
      - Local .nemo file     → returned as-is after existence check
      - Local directory      → returned as-is after existence check
      - HuggingFace Hub ID   → downloaded via NeMo's caching pipeline
      - NGC model name       → downloaded via NeMo's caching pipeline

    Returns a local .nemo file path OR a local directory path.
    """
    p = Path(source)
    if p.exists():
        return str(p)

    # HF Hub ID or NGC name: use NeMo's download + cache mechanism.
    # return_model_file=True returns the local path without loading the model.
    from nemo.collections.asr.models import ASRModel
    print(f"  Downloading: {source} ...")
    return ASRModel.from_pretrained(source, return_model_file=True)


def _load_checkpoint(source: str, tmpdir: str) -> tuple[dict, Any]:
    """
    Load a state_dict and optional config from any checkpoint source.

    Handled formats:
      1. .nemo tarball          – extract → load model_weights.ckpt
      2. directory / model_weights.ckpt  – load directly (unpacked NeMo)
      3. directory / model.safetensors   – load with safetensors
    """
    p = Path(source)

    # ── Format 1: .nemo tarball ───────────────────────────────────────────────
    if p.is_file() and p.suffix == ".nemo":
        with tarfile.open(source) as tar:
            tar.extractall(tmpdir, filter="data")
        weights_path = Path(tmpdir) / "model_weights.ckpt"
        config_path  = Path(tmpdir) / "model_config.yaml"

        if not weights_path.exists():
            raise FileNotFoundError(f"model_weights.ckpt not found inside {source}")

        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        cfg = OmegaConf.load(config_path) if config_path.exists() else None
        return state_dict, cfg

    # ── Format 2 / 3: directory ───────────────────────────────────────────────
    if p.is_dir():
        ckpt_path = p / "model_weights.ckpt"
        st_path   = p / "model.safetensors"
        cfg_path  = p / "model_config.yaml"

        if ckpt_path.exists():
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        elif st_path.exists():
            try:
                from safetensors.torch import load_file
            except ImportError:
                raise ImportError(
                    "safetensors is required to load this model.\n"
                    "Install with: pip install safetensors"
                )
            state_dict = load_file(str(st_path), device="cpu")
        else:
            raise FileNotFoundError(
                f"No model_weights.ckpt or model.safetensors found in {source}"
            )

        cfg = OmegaConf.load(cfg_path) if cfg_path.exists() else None
        return state_dict, cfg

    raise FileNotFoundError(
        f"Unsupported source: {source}\n"
        "Expected a .nemo file, a model directory, a HF Hub ID, or an NGC name."
    )


def _pack_nemo(save_path: str, state_dict: dict, sub_cfg=None):
    """Pack state_dict (and optional sub-config) into a .nemo tarball."""
    with tempfile.TemporaryDirectory() as tmpdir:
        weights_path = os.path.join(tmpdir, "model_weights.ckpt")
        torch.save(state_dict, weights_path)

        arcnames = [("model_weights.ckpt", weights_path)]

        if sub_cfg is not None:
            cfg_path = os.path.join(tmpdir, "module_config.yaml")
            OmegaConf.save(sub_cfg, cfg_path)
            arcnames.append(("module_config.yaml", cfg_path))

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with tarfile.open(save_path, "w:") as tar:
            for arcname, fspath in arcnames:
                tar.add(fspath, arcname=arcname)

    print(f"  Saved : {save_path}  ({len(state_dict)} keys)")


# ─────────────────────────────────────────────────────────────────────────────
#  Mode 1 – split SALM .nemo into per-module files
# ─────────────────────────────────────────────────────────────────────────────

def split_nemo(source: str, output_dir: str | None = None):
    """Split a SALM .nemo checkpoint into per-module .nemo files."""
    local  = _resolve_source(source)
    stem   = Path(local).stem if Path(local).is_file() else Path(source).name
    out_dir = Path(output_dir) if output_dir else Path(local).parent

    print(f"Source : {source}")
    if local != source:
        print(f"Local  : {local}")
    print(f"Output : {out_dir}")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        full_state_dict, full_cfg = _load_checkpoint(local, tmpdir)

    print(f"Total keys: {len(full_state_dict)}")
    print()

    for module_name, cfg_key in MODULES.items():
        prefix = module_name + "."
        module_sd = {
            k[len(prefix):]: v
            for k, v in full_state_dict.items()
            if k.startswith(prefix)
        }

        if not module_sd:
            print(f"  [{module_name}] no keys found — skipping")
            continue

        sub_cfg = None
        if cfg_key is not None and full_cfg is not None:
            sub_cfg = full_cfg.get(cfg_key, None)

        out_path = str(out_dir / f"{stem}_{module_name}.nemo")
        _pack_nemo(out_path, module_sd, sub_cfg)

    print()
    print("Done.")


# ─────────────────────────────────────────────────────────────────────────────
#  Mode 2 – extract encoder + preprocessor from any ASR checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def extract_encoder_from_asr(source: str, output_dir: str | None = None):
    """
    Extract the FastConformer encoder + preprocessor from a NeMo ASR checkpoint
    and save as a standalone .nemo file.

    Accepted inputs:
      - Local .nemo / directory path
      - HuggingFace Hub ID : nvidia/canary-1b-v2  (auto-downloaded)
      - NGC model name

    The output can be used directly as:
        model.module_weights.perception_from_asr: /path/to/<stem>_asr_encoder.nemo
    """
    local   = _resolve_source(source)
    stem    = Path(local).stem if Path(local).is_file() else Path(source).name
    out_dir = Path(output_dir) if output_dir else Path(local).parent if Path(local).is_file() else Path(".")

    print(f"Source (ASR) : {source}")
    if local != source:
        print(f"Local        : {local}")
    print(f"Output       : {out_dir}")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        full_state_dict, full_cfg = _load_checkpoint(local, tmpdir)

    print(f"Total keys in ASR checkpoint: {len(full_state_dict)}")

    encoder_sd = {
        k: v for k, v in full_state_dict.items()
        if any(k.startswith(p) for p in _ASR_ENCODER_PREFIXES)
    }

    if not encoder_sd:
        raise ValueError(
            f"No encoder.* / preprocessor.* keys found in checkpoint from {source}.\n"
            "Make sure this is a NeMo ASR model (FastConformer-based)."
        )

    # Carry the encoder/preprocessor sub-config if available.
    sub_cfg = None
    if full_cfg is not None:
        encoder_cfg     = full_cfg.get("encoder", None)
        preprocessor_cfg = full_cfg.get("preprocessor", None)
        if encoder_cfg or preprocessor_cfg:
            sub_cfg = OmegaConf.create({})
            if encoder_cfg:
                sub_cfg.encoder = encoder_cfg
            if preprocessor_cfg:
                sub_cfg.preprocessor = preprocessor_cfg

    out_path = str(out_dir / f"{stem}_asr_encoder.nemo")
    _pack_nemo(out_path, encoder_sd, sub_cfg)

    print()
    print(f"Done.  Use as:  model.module_weights.perception_from_asr: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Split a SALM .nemo into modules, or extract an ASR encoder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # split SALM checkpoint
  python convert_modules.py canary-qwen-2.5b.nemo
  python convert_modules.py canary-qwen-2.5b.nemo --output-dir /ckpts

  # extract encoder from local ASR .nemo
  python convert_modules.py --from-asr canary-1b-v2.nemo

  # extract encoder from HF Hub (auto-download)
  python convert_modules.py --from-asr nvidia/canary-1b-v2
  python convert_modules.py --from-asr nvidia/canary-1b-flash --output-dir /ckpts
        """,
    )
    parser.add_argument(
        "nemo_path",
        nargs="?",
        help="SALM .nemo path or HF Hub ID (split mode).",
    )
    parser.add_argument(
        "--from-asr",
        metavar="SOURCE",
        default=None,
        help=(
            "Extract FastConformer encoder+preprocessor from a NeMo ASR checkpoint. "
            "Accepts a local .nemo path, a directory, a HF Hub ID "
            "(e.g. nvidia/canary-1b-v2), or an NGC model name."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output .nemo files (default: same as input).",
    )
    args = parser.parse_args()

    if args.from_asr:
        extract_encoder_from_asr(args.from_asr, args.output_dir)
    elif args.nemo_path:
        split_nemo(args.nemo_path, args.output_dir)
    else:
        parser.error("Provide a SALM .nemo path or use --from-asr <source>.")


if __name__ == "__main__":
    main()
