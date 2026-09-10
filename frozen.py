# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Records the reproducibility contract of plan section 2 BEFORE any training, and fails
# loudly if the checkpoint is not the LSTM-predictor hybrid the compatibility route
# assumes.  This is item 1-2 of the plan's first implementation checklist.
#
# The artifact's identity in the contract is the SHA-256 of the ``.nemo`` file itself --
# that is what a reader can actually re-verify.  It is a second full read of the artifact
# (``restore_from`` reads it again), so ``--no-artifact-hash`` opts out on slow storage.
# Per-module tensor hashes are recorded alongside it (``--fingerprint {full,fast,none}``).
#
# Every phase is timed and reported.  Most of the wall clock here is usually NeMo's import
# and ``restore_from`` untarring the archive, neither of which this script controls -- the
# breakdown tells you which.
#
# Usage:
#   python recurrent_transducer/scripts/inspect_checkpoint.py \
#       --checkpoint "$RT_CHECKPOINT" \
#       --output exp/contract/checkpoint_contract.json

import argparse
import contextlib
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

TIMINGS = {}


@contextlib.contextmanager
def _phase(name: str, detail: str = ""):
    """Time one phase and report it as it completes, so a slow step is visible live.

    ``name`` is the stable key recorded in the contract; ``detail`` is display-only, so a
    label carrying a file size cannot make ``timings_seconds`` keys differ between two
    otherwise comparable runs.  ``finally`` ensures a failing phase still reports how long
    it ran before it died -- which is the whole point when a step is suspected of hanging.
    """
    label = f"{name} ({detail})" if detail else name
    print(f"[  ...  ] {label}", file=sys.stderr, flush=True)
    start = time.perf_counter()
    try:
        yield
    finally:
        TIMINGS[name] = round(time.perf_counter() - start, 2)
        print(f"[{TIMINGS[name]:7.2f}s] {label}", file=sys.stderr, flush=True)


def main() -> int:
    args = _parse_args()
    with _phase("import nemo.collections.asr"):
        from nemo.collections.asr.modules.rnnt import RNNTDecoder, RNNTJoint

        from recurrent_transducer.models.frozen_encoder_rnnt import (
            file_sha256,
            load_asr_model,
            module_fingerprint,
            resolve_checkpoint,
        )

    with _phase("resolve checkpoint path"):
        resolved = resolve_checkpoint(args.checkpoint)

    identity = {"name": args.checkpoint, "resolved": resolved}
    artifact = Path(resolved) if resolved.endswith(".nemo") else None
    if artifact is not None and artifact.is_file():
        identity["artifact_bytes"] = artifact.stat().st_size  # one stat, reused below
        if args.artifact_hash:
            # This is a second full read of the artifact: ``restore_from`` reads and untars
            # the same file moments later.  Worth it to pin the contract to exact bytes on a
            # local disk; on cold network storage it can dominate the run, hence the opt-out.
            gigabytes = identity["artifact_bytes"] / 1e9
            with _phase("sha256 of the .nemo artifact", detail=f"{gigabytes:.2f} GB"):
                identity["artifact_sha256"] = file_sha256(artifact)
        else:
            identity["artifact_sha256"] = None
            identity["artifact_note"] = "skipped via --no-artifact-hash"
    else:
        identity["artifact_sha256"] = None
        identity["artifact_note"] = (
            "not a local .nemo file, so there is no artifact hash to record. Pre-download the "
            "checkpoint and point --checkpoint at it to pin the contract to exact bytes."
        )

    with _phase("restore model (untar + torch.load)"):
        model = load_asr_model(resolved, map_location="cpu")
        model.eval()

    with _phase("read module configuration"):
        contract = {
            "checkpoint": {**identity, "class": type(model).__name__},
            "tokenizer": _tokenizer_contract(model),
            "encoder": _encoder_contract(model),
            "predictor": _predictor_contract(model),
            "joiner": _joiner_contract(model),
            "environment": _environment_contract(),
            "parameters": {
                "encoder": sum(p.numel() for p in model.encoder.parameters()),
                "predictor": sum(p.numel() for p in model.decoder.parameters()),
                "joiner": sum(p.numel() for p in model.joint.parameters()),
                "total": sum(p.numel() for p in model.parameters()),
            },
        }

    if args.fingerprint != "none":
        exact = args.fingerprint == "full"
        with _phase("module fingerprints", detail="exact tensor bytes" if exact else "per-tensor moments"):
            contract["module_fingerprints"] = {
                "mode": args.fingerprint,
                "encoder": module_fingerprint(model.encoder, exact=exact),
                "decoder": module_fingerprint(model.decoder, exact=exact),
                "joint": module_fingerprint(model.joint, exact=exact),
            }

    problems = []
    if not isinstance(model.decoder, RNNTDecoder):
        problems.append(
            f"decoder is {type(model.decoder).__name__}, not RNNTDecoder: this checkpoint does not use the "
            "LSTM predictor the compatibility route assumes -- stop the compatibility route (plan section 2)."
        )
    if not isinstance(model.joint, RNNTJoint):
        problems.append(f"joint is {type(model.joint).__name__}, not RNNTJoint")

    dec_rnn = predictor_rnn(model.decoder)
    if dec_rnn is None:
        problems.append("decoder has no prediction['dec_rnn']; this is not the LSTM predictor route")
    elif not isinstance(dec_rnn, torch.nn.LSTM) and not hasattr(dec_rnn, "lstm"):
        problems.append(f"predictor RNN is {type(dec_rnn).__name__}; expected an LSTM")
    contract["compatibility_problems"] = problems
    contract["timings_seconds"] = TIMINGS

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, indent=2, sort_keys=True, default=str))
    print(json.dumps(contract, indent=2, sort_keys=True, default=str))

    print(f"\nPhase timings: {json.dumps(TIMINGS, sort_keys=True)}", file=sys.stderr)
    if problems:
        print("\nCOMPATIBILITY PROBLEMS:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"\nContract written to {args.output}")
    return 0


def predictor_rnn(decoder):
    """The decoder's prediction RNN, or ``None`` if it has none.

    NeMo stores the prediction network in an ``nn.ModuleDict``, which supports ``in`` and
    ``[]`` but NOT ``.get()``: ``ModuleDict.get`` raises ``AttributeError`` rather than
    returning a default.  Using it here killed the whole audit at exactly the point that is
    meant to *report* an incompatible checkpoint, so membership is tested explicitly.
    """
    prediction = getattr(decoder, "prediction", None)
    if prediction is None or "dec_rnn" not in prediction:
        return None
    return prediction["dec_rnn"]


def _tokenizer_contract(model) -> dict:
    """Tokenizer bytes/hash, vocabulary size, blank index and SOS handling (plan 2.2)."""
    tokenizer = getattr(model, "tokenizer", None)
    contract = {
        "class": type(tokenizer).__name__ if tokenizer is not None else None,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "num_classes_with_blank": model.joint.num_classes_with_blank,
        "blank_index": model.decoder.blank_idx,
        "blank_as_pad": bool(getattr(model.decoder, "blank_as_pad", False)),
        # NeMo's predict(add_sos=True) prepends a ZERO VECTOR, not an embedding lookup.
        "sos_convention": "zero_vector_prepended_by_RNNTDecoder.predict(add_sos=True)",
    }
    model_path = getattr(getattr(tokenizer, "tokenizer", None), "model_file", None) or getattr(
        tokenizer, "model_path", None
    )
    if model_path and Path(model_path).is_file():
        contract["model_file"] = str(model_path)
        contract["model_file_sha256"] = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    return contract


def _encoder_contract(model) -> dict:
    """Encoder dimension, subsampling, attention/convolution context and cache shapes."""
    encoder = model.encoder
    contract = {
        "class": type(encoder).__name__,
        "d_model": getattr(encoder, "d_model", None),
        "num_layers": len(getattr(encoder, "layers", []) or []),
        "subsampling_factor": getattr(encoder, "subsampling_factor", None),
        "att_context_size": _plain(getattr(encoder, "att_context_size", None)),
        "att_context_sizes_all": _plain(getattr(encoder, "att_context_size_all", None)),
        "att_context_style": getattr(encoder, "att_context_style", None),
        "conv_context_size": _plain(getattr(encoder, "conv_context_size", None)),
        "conv_kernel_size": getattr(encoder, "conv_kernel_size", None),
        "streaming_cfg": _plain(getattr(encoder, "streaming_cfg", None)),
    }
    if hasattr(encoder, "get_initial_cache_state"):
        try:
            cache_channel, cache_time, cache_len = encoder.get_initial_cache_state(batch_size=1)
            contract["cache_shapes"] = {
                "last_channel": list(cache_channel.shape),
                "last_time": list(cache_time.shape),
                "last_channel_len": list(cache_len.shape),
            }
        except Exception as exc:  # pragma: no cover - depends on the restored config
            contract["cache_shapes_error"] = repr(exc)
    return contract


def _predictor_contract(model) -> dict:
    """Predictor type, layers, embedding/hidden widths, dropout, projections, state layout.

    Degrades instead of raising.  This runs *before* the compatibility check, and the whole
    point of that check is to report "this checkpoint does not use the LSTM predictor" as a
    finding -- so an unexpected predictor must not crash the audit on the way there.
    """
    decoder = model.decoder
    contract = {
        "class": type(decoder).__name__,
        "pred_hidden": getattr(decoder, "pred_hidden", None),
        "random_state_sampling": bool(getattr(decoder, "random_state_sampling", False)),
    }

    prediction = getattr(decoder, "prediction", None)
    if prediction is None:
        contract["unavailable"] = f"{type(decoder).__name__} has no `prediction` module dict"
        return contract

    embed = prediction["embed"] if "embed" in prediction else None
    if embed is not None:
        contract["embedding"] = {
            "num_embeddings": embed.num_embeddings,
            "dim": embed.embedding_dim,
            "padding_idx": embed.padding_idx,
        }

    rnn = predictor_rnn(decoder)
    if rnn is None:
        contract["unavailable"] = "no prediction['dec_rnn']"
        return contract

    inner = getattr(rnn, "lstm", rnn)  # NeMo may wrap nn.LSTM in LSTMDropout
    contract.update(
        {
            "rnn_class": type(rnn).__name__,
            "num_layers": getattr(inner, "num_layers", None),
            "input_size": getattr(inner, "input_size", None),
            "hidden_size": getattr(inner, "hidden_size", None),
            "proj_size": getattr(inner, "proj_size", 0),
            "dropout": getattr(inner, "dropout", None),
        }
    )
    try:
        h, c = decoder.initialize_state(torch.zeros(1, 1, decoder.pred_hidden))
        contract["state_layout"] = {"h": list(h.shape), "c": list(c.shape)}
    except Exception as exc:  # noqa: BLE001 - a non-LSTM state layout is a finding, not a crash
        contract["state_layout_error"] = repr(exc)
    return contract


def _joiner_contract(model) -> dict:
    """Joiner projections, hidden dim, activation, dropout, output proj, log-softmax rule."""
    joint = model.joint
    head = [type(m).__name__ for m in joint.joint_net]
    return {
        "class": type(joint).__name__,
        "enc_proj": [joint.enc.in_features, joint.enc.out_features],
        "pred_proj": [joint.pred.in_features, joint.pred.out_features],
        "joint_hidden": joint.joint_net[-1].in_features,
        "output_proj": [joint.joint_net[-1].in_features, joint.joint_net[-1].out_features],
        "joint_net_modules": head,
        "dropout": next((m.p for m in joint.joint_net if isinstance(m, torch.nn.Dropout)), 0.0),
        "log_softmax": joint.log_softmax,
        "temperature": getattr(joint, "temperature", 1.0),
        "fuse_loss_wer": bool(getattr(joint, "_fuse_loss_wer", False)),
    }


def _environment_contract() -> dict:
    """NeMo Speech commit, framework versions, precision/accelerator availability."""
    import nemo

    return {
        "nemo_version": getattr(nemo, "__version__", None),
        "nemo_git_sha": _git_sha(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "python": sys.version,
        "platform": platform.platform(),
    }


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown (not a git checkout: record the source archive hash instead)"


def _plain(value):
    """Convert OmegaConf / dataclass config fragments to plain JSON-serialisable data."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except ImportError:
        pass
    if hasattr(value, "__dict__") and not isinstance(value, (str, int, float, bool)):
        return {k: _plain(v) for k, v in vars(value).items() if not k.startswith("_")}
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="nvidia/stt_en_fastconformer_hybrid_large_streaming_multi")
    parser.add_argument("--output", type=Path, default=Path("exp/contract/checkpoint_contract.json"))
    parser.add_argument(
        "--no-artifact-hash",
        dest="artifact_hash",
        action="store_false",
        help=(
            "skip the SHA-256 of the .nemo file. It is a second full read of the artifact "
            "(restore_from reads it again straight after), which can dominate the run on "
            "cold network storage. Skipping it leaves the contract without exact-byte "
            "provenance for the checkpoint."
        ),
    )
    parser.add_argument(
        "--fingerprint",
        choices=["none", "fast", "full"],
        default="full",
        help=(
            "per-module tensor hashes, on top of the .nemo artifact hash. 'full' (default) "
            "hashes every byte (~2 s for a 114M-parameter encoder); 'fast' uses per-tensor "
            "moments (~0.2 s); 'none' skips them."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
