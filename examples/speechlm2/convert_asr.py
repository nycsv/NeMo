"""
convert_asr.py
──────────────
Download a NeMo ASR model from HuggingFace Hub (or NGC) and save it as a
local .nemo file.

Handles two HF Hub layouts:
  Case A  repo contains <model>.nemo  → single-file download, copy to output
  Case B  repo is safetensors format  → snapshot_download, load, save_to()

Usage:
  python convert_asr.py nvidia/canary-1b-v2
  python convert_asr.py nvidia/canary-1b-flash
  python convert_asr.py nvidia/canary-1b-v2 --output /ckpts/canary-1b-v2.nemo
"""

import argparse
import os
import shutil
from pathlib import Path

from nemo.collections.asr.models import ASRModel
from nemo.collections.speechlm2.parts.pretrained import resolve_nemo_path
from nemo.utils import logging


def convert_asr(model_id: str, output: str | None = None) -> str:
    """
    Download ``model_id`` and save as a local .nemo file.

    Args:
        model_id: HuggingFace Hub ID (``nvidia/canary-1b-v2``), NGC name, or
                  local .nemo / directory path.
        output:   Destination .nemo path.  Defaults to ``<model-name>.nemo``
                  in the current directory.

    Returns:
        Absolute path of the saved .nemo file.
    """
    model_name = Path(model_id).name          # e.g. "canary-1b-v2"
    dest = Path(output) if output else Path(f"{model_name}.nemo")

    logging.info(f"Source : {model_id}")
    logging.info(f"Output : {dest}")

    # ── Step 1: resolve to a local path (download if needed) ──────────────────
    # resolve_nemo_path returns either:
    #   • a .nemo file path   (Case A: repo had a single .nemo)
    #   • a directory path    (Case B: repo was in safetensors / unpacked format)
    cached = resolve_nemo_path(model_id)
    logging.info(f"Cached : {cached}")

    # ── Step 2: copy or re-pack ────────────────────────────────────────────────
    if os.path.isfile(cached) and cached.endswith(".nemo"):
        # Case A: already a .nemo file — just copy to the desired output path
        logging.info("Case A: .nemo file found in cache — copying.")
        os.makedirs(dest.parent, exist_ok=True)
        shutil.copy2(cached, dest)

    elif os.path.isdir(cached):
        # Case B: downloaded as a directory (safetensors or unpacked NeMo)
        # Load the model and save_to() to produce a canonical .nemo tarball.
        logging.info("Case B: directory found — loading and re-packing as .nemo.")
        model = ASRModel.restore_from(cached)
        model.eval()
        os.makedirs(dest.parent, exist_ok=True)
        model.save_to(str(dest))

    else:
        raise RuntimeError(
            f"Unexpected cached path: {cached}\n"
            "Expected a .nemo file or a directory."
        )

    size_mb = dest.stat().st_size / 1024 / 1024
    logging.info(f"Saved  : {dest}  ({size_mb:.0f} MB)")
    return str(dest)


def main():
    parser = argparse.ArgumentParser(
        description="Download a NeMo ASR model and save as a local .nemo file."
    )
    parser.add_argument(
        "model_id",
        help=(
            "HuggingFace Hub ID (e.g. nvidia/canary-1b-v2), "
            "NGC name, or local .nemo path."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .nemo path (default: <model-name>.nemo in current dir).",
    )
    args = parser.parse_args()
    convert_asr(args.model_id, args.output)


if __name__ == "__main__":
    main()
