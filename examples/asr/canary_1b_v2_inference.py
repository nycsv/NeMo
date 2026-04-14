#!/usr/bin/env python3
"""
===========================================================================
 Canary-1B-v2 — Inference from a Local .nemo Checkpoint
===========================================================================

Usage:
    python canary_1b_v2_inference.py --model_path /path/to/your_model.nemo --audio file.wav
    python canary_1b_v2_inference.py --model_path ./my_canary.nemo --audio a.wav b.wav
    python canary_1b_v2_inference.py --model_path ./my_canary.nemo --audio file.wav --task translate --target_lang fr
    python canary_1b_v2_inference.py --model_path ./my_canary.nemo --audio file.wav --timestamps

Requirements:
    pip install -U "nemo_toolkit[asr]"
===========================================================================
"""

import argparse
import os
import sys


SUPPORTED_LANGS = [
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
    "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk",
    "sl", "es", "sv", "ru", "uk",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Canary-1B-v2 inference from a local .nemo checkpoint"
    )
    p.add_argument(
        "--model_path", type=str, required=True,
        help="Path to your trained .nemo checkpoint",
    )
    p.add_argument(
        "--audio", type=str, nargs="+", required=True,
        help="One or more audio file paths (.wav/.flac, 16 kHz mono)",
    )
    p.add_argument(
        "--task", choices=["transcribe", "translate"], default="transcribe",
        help="'transcribe' for ASR, 'translate' for AST (default: transcribe)",
    )
    p.add_argument(
        "--source_lang", type=str, default="en", choices=SUPPORTED_LANGS,
        help="Source language code (default: en)",
    )
    p.add_argument(
        "--target_lang", type=str, default="en", choices=SUPPORTED_LANGS,
        help="Target language code (default: en)",
    )
    p.add_argument(
        "--timestamps", action="store_true",
        help="Enable word-level and segment-level timestamps",
    )
    p.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for inference (default: 1)",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="'cuda', 'cpu', or None for auto-detect",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Load model from .nemo
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path, device=None):
    """
    Restore a Canary model from a local .nemo checkpoint.

    ASRModel.restore_from() unpacks the archive and rebuilds the full
    EncDecMultiTaskModel graph (config, weights, tokenizer).
    """
    import torch
    from nemo.collections.asr.models import ASRModel

    if not os.path.isfile(model_path):
        print(f"❌ Checkpoint not found: {model_path}")
        sys.exit(1)

    print(f"🔧 Restoring model from: {model_path}")
    model = ASRModel.restore_from(restore_path=model_path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✅ Model loaded on {device}  ({n_params:.1f}M params)\n")
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model, audio_paths, source_lang, target_lang,
                  batch_size=1, timestamps=False):
    """Single entry point — handles ASR, AST, and timestamps."""
    return model.transcribe(
        audio_paths,
        source_lang=source_lang,
        target_lang=target_lang,
        batch_size=batch_size,
        timestamps=timestamps,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Pretty-print
# ─────────────────────────────────────────────────────────────────────────────

def print_results(outputs, show_timestamps=False, is_translation=False):
    for i, out in enumerate(outputs):
        print(f"{'='*60}")
        print(f"  Sample {i+1}")
        print(f"{'='*60}")
        print(f"  Text: {out.text}")

        if show_timestamps and hasattr(out, "timestamp") and out.timestamp:
            if "segment" in out.timestamp:
                print(f"\n  ── Segment Timestamps ──")
                for s in out.timestamp["segment"]:
                    print(f"    {s['start']:.2f}s – {s['end']:.2f}s : {s['segment']}")

            if "word" in out.timestamp and not is_translation:
                print(f"\n  ── Word Timestamps ──")
                for w in out.timestamp["word"]:
                    print(f"    {w['start']:.2f}s – {w['end']:.2f}s : {w['word']}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Validate audio files exist
    for f in args.audio:
        if not os.path.isfile(f):
            print(f"❌ Audio file not found: {f}")
            sys.exit(1)

    model = load_model(args.model_path, device=args.device)

    is_translation = (args.task == "translate") or (args.source_lang != args.target_lang)
    target = args.target_lang if is_translation else args.source_lang

    label = f"Translating {args.source_lang} → {target}" if is_translation else f"Transcribing ({args.source_lang})"
    print(f"🎙  {label} {'with timestamps ' if args.timestamps else ''}...")

    outputs = run_inference(
        model,
        args.audio,
        source_lang=args.source_lang,
        target_lang=target,
        batch_size=args.batch_size,
        timestamps=args.timestamps,
    )

    print_results(outputs, show_timestamps=args.timestamps, is_translation=is_translation)


if __name__ == "__main__":
    main()
