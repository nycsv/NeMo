"""
convert_modules.py
──────────────────
Two modes:

  split   (default)
    Split a SALM canary-qwen-2.5b.nemo into per-module .nemo files.

    Output files (written next to the source file by default):
      <stem>_llm.nemo          – model.llm  (Qwen backbone + lm_head)
      <stem>_embed_tokens.nemo – model.embed_tokens  (vocab embedding)
      <stem>_perception.nemo   – model.perception  (encoder + adapter)

    Usage:
      python convert_modules.py canary-qwen-2.5b.nemo
      python convert_modules.py canary-qwen-2.5b.nemo --output-dir /ckpts

  extract-encoder
    Extract the FastConformer encoder + preprocessor from any NeMo ASR
    checkpoint (.nemo) into a standalone .nemo file that can be used as
    model.module_weights.perception_from_asr during SALM training.

    Usage:
      python convert_modules.py --from-asr my_asr.nemo
      python convert_modules.py --from-asr my_asr.nemo --output-dir /ckpts

Each output .nemo is a tarball containing:
  model_weights.ckpt  – state_dict with the module prefix stripped
  module_config.yaml  – the matching sub-config slice (when available)
"""

import argparse
import os
import tarfile
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Top-level modules and which config key holds their sub-config (None = no sub-config).
MODULES = {
    "llm":          None,
    "embed_tokens": None,
    "perception":   "perception",
}


def _extract_nemo(nemo_path: str, tmpdir: str):
    """Unpack a .nemo tarball and return (state_dict, cfg_or_none)."""
    with tarfile.open(nemo_path) as tar:
        tar.extractall(tmpdir, filter="data")

    weights_path = Path(tmpdir) / "model_weights.ckpt"
    if not weights_path.exists():
        raise FileNotFoundError(f"model_weights.ckpt not found inside {nemo_path}")

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    config_path = Path(tmpdir) / "model_config.yaml"
    cfg = OmegaConf.load(config_path) if config_path.exists() else None

    return state_dict, cfg


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


def split_nemo(nemo_path: str, output_dir: str | None = None):
    source = Path(nemo_path)
    out_dir = Path(output_dir) if output_dir else source.parent

    print(f"Source : {source}")
    print(f"Output : {out_dir}")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        full_state_dict, full_cfg = _extract_nemo(nemo_path, tmpdir)

    print(f"Total keys in full checkpoint: {len(full_state_dict)}")
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

        out_path = str(out_dir / f"{source.stem}_{module_name}.nemo")
        _pack_nemo(out_path, module_sd, sub_cfg)

    print()
    print("Done.")


# Keys to keep when extracting encoder from an ASR checkpoint.
_ASR_ENCODER_PREFIXES = ("encoder.", "preprocessor.")


def extract_encoder_from_asr(asr_nemo_path: str, output_dir: str | None = None):
    """
    Extract the FastConformer encoder + preprocessor from a NeMo ASR .nemo
    and save them as a standalone .nemo file.

    The output can be used directly as:
        model.module_weights.perception_from_asr: /path/to/<stem>_asr_encoder.nemo

    Keys in the output file are the same as in the ASR checkpoint (e.g.
    ``encoder.layers.0.self_attn.linear_q.weight``), ready to be loaded
    into model.perception with strict=False.
    """
    source = Path(asr_nemo_path)
    out_dir = Path(output_dir) if output_dir else source.parent

    print(f"Source (ASR) : {source}")
    print(f"Output       : {out_dir}")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        full_state_dict, full_cfg = _extract_nemo(asr_nemo_path, tmpdir)

    print(f"Total keys in ASR checkpoint: {len(full_state_dict)}")

    encoder_sd = {
        k: v for k, v in full_state_dict.items()
        if any(k.startswith(p) for p in _ASR_ENCODER_PREFIXES)
    }

    if not encoder_sd:
        raise ValueError(
            f"No encoder.* / preprocessor.* keys found in {asr_nemo_path}.\n"
            "Make sure this is a NeMo ASR model checkpoint."
        )

    # Attach the encoder sub-config if the ASR config is available.
    sub_cfg = None
    if full_cfg is not None:
        encoder_cfg = full_cfg.get("encoder", None)
        preprocessor_cfg = full_cfg.get("preprocessor", None)
        if encoder_cfg or preprocessor_cfg:
            from omegaconf import OmegaConf as _OC
            sub_cfg = _OC.create({})
            if encoder_cfg:
                sub_cfg.encoder = encoder_cfg
            if preprocessor_cfg:
                sub_cfg.preprocessor = preprocessor_cfg

    out_path = str(out_dir / f"{source.stem}_asr_encoder.nemo")
    _pack_nemo(out_path, encoder_sd, sub_cfg)

    print()
    print(f"Done.  Use as:  model.module_weights.perception_from_asr: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Split a SALM .nemo into modules, or extract an ASR encoder."
    )
    parser.add_argument(
        "nemo_path",
        nargs="?",
        help="Path to canary-qwen-2.5b.nemo (split mode, default).",
    )
    parser.add_argument(
        "--from-asr",
        metavar="ASR_NEMO",
        default=None,
        help="Extract FastConformer encoder + preprocessor from a NeMo ASR .nemo.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output files (default: same directory as input).",
    )
    args = parser.parse_args()

    if args.from_asr:
        extract_encoder_from_asr(args.from_asr, args.output_dir)
    elif args.nemo_path:
        split_nemo(args.nemo_path, args.output_dir)
    else:
        parser.error("Provide a SALM .nemo path or use --from-asr <asr.nemo>.")


if __name__ == "__main__":
    main()
