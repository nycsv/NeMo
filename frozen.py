

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from recurrent_transducer.decoding.looped_greedy import LoopedGreedyRNNTInfer
from recurrent_transducer.modules.looped_predictor_joint import (
    LoopedJointOnly,
    LoopedPredictorJoint,
    count_parameters,
)
from recurrent_transducer.modules.prefix_lstm import collect_prefix_cache

try:  # Lightning is only needed for actual training runs.
    from lightning.pytorch import LightningModule as _LightningModule
except ImportError:  # pragma: no cover - depends on the installed environment
    try:
        from pytorch_lightning import LightningModule as _LightningModule
    except ImportError:
        _LightningModule = None

_ModuleBase = _LightningModule if _LightningModule is not None else nn.Module

FEEDBACK_ALIASES = {"pre_vocabulary_hidden": "joint_hidden", "joint_hidden": "joint_hidden", "zero": "zero"}


def freeze_module(module: nn.Module) -> nn.Module:
    """Freeze ``module``: no gradients, and eval mode that survives ``parent.train()``.

    ``nn.Module.train()`` recurses into children, so simply calling ``.eval()`` once is
    not enough -- the next ``model.train()`` would re-enable dropout and batch-norm
    updates inside the "frozen" encoder.  Patching the bound ``train`` method keeps the
    module in eval mode permanently, which is what the plan's encoder-freeze gate checks.
    """
    module.eval()
    module.requires_grad_(False)
    if not getattr(module, "_rt_train_patched", False):
        original_train = module.train

        def _always_eval(mode: bool = True, _orig=original_train):
            return _orig(False)

        module.train = _always_eval
        module._rt_train_patched = True
    return module


def module_fingerprint(module: nn.Module, exact: bool = True) -> str:
    """Hash a module's parameters *and* mutable buffers.

    Buffers matter as much as parameters here: running statistics and caches can drift
    even when every parameter has ``requires_grad=False``, so a freeze check that looked
    only at parameters would miss it.

    Two modes, because the two callers want different things:

    - ``exact=True`` hashes the raw bytes.  Use it for provenance, where the digest is
      recorded and compared across machines and runs.  About 2 s for a 114M-parameter
      encoder, dominated by the host copy.
    - ``exact=False`` hashes per-tensor ``(numel, sum, L2 norm)`` reductions computed on the
      tensor's own device -- about 0.2 s for the same encoder, and no host transfer at all
      on GPU.  Use it to answer "did an optimizer step move this", which is all the
      in-training freeze check needs.  It detects accidental modification, not deliberate
      tampering: two genuinely different tensors could in principle share both moments.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(list(module.named_parameters()) + list(module.named_buffers()), key=lambda kv: kv[0]):
        digest.update(name.encode())
        digest.update(f"{tuple(tensor.shape)}{tensor.dtype}".encode())
        tensor = tensor.detach()
        if exact:
            digest.update(_raw_bytes(tensor))
        else:
            # ``dtype=`` accumulates in float64 without materialising a float64 copy, and
            # vector_norm avoids a full squared temporary.  Integer buffers (BatchNorm's
            # num_batches_tracked, for one) are rejected by vector_norm and are small
            # enough that an explicit cast costs nothing.
            total = float(tensor.sum(dtype=torch.float64))
            values = tensor.flatten()
            if values.is_floating_point():
                spread = float(torch.linalg.vector_norm(values, ord=2, dtype=torch.float64))
            else:
                spread = float(values.to(torch.float64).pow(2).sum())
            digest.update(f"{tensor.numel()}:{total!r}:{spread!r}".encode())
    return digest.hexdigest()


def _raw_bytes(tensor: torch.Tensor) -> memoryview:
    """Zero-copy view of a tensor's raw bytes, for hashing.

    Deliberately NOT ``bytes(tensor.untyped_storage())``: ``UntypedStorage`` has no buffer
    protocol, so ``bytes()`` falls back to iterating it one Python int per byte -- about
    4 seconds per megabyte, which turns a 114M-parameter encoder into a ~30 minute hash.
    Viewing as ``uint8`` and going through NumPy's buffer protocol is zero-copy and reaches
    roughly 0.2 GB/s, and the ``uint8`` view also covers dtypes NumPy cannot represent
    directly (bfloat16, float8).
    """
    flat = tensor.cpu().contiguous().flatten().view(torch.uint8)
    try:
        return memoryview(flat.numpy())
    except RuntimeError as exc:  # NumPy missing or ABI-mismatched
        raise RuntimeError(
            "exact fingerprints need a working torch<->NumPy bridge; "
            "use exact=False (per-tensor moments) or repair the NumPy install"
        ) from exc


def file_sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    """Streaming SHA-256 of a file, without reading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def build_scorer(decoder: nn.Module, joint: nn.Module, loop_cfg: Dict[str, Any]) -> Optional[nn.Module]:
    """Construct the looped scorer from a checkpoint's own predictor and joiner.

    Dimensions are read off the restored modules -- never chosen for convenience, per the
    plan's reproducibility contract.

    Args:
        decoder: NeMo ``RNNTDecoder`` (must expose ``prediction["embed"]`` / ``["dec_rnn"]``).
        joint: NeMo ``RNNTJoint`` (must expose ``enc``, ``pred``, ``joint_net``).
        loop_cfg: the ``loop`` block of the experiment config.

    Returns:
        The scorer, or ``None`` for ``variant: none`` (the B1 plain fine-tuning baseline).
    """
    variant = loop_cfg.get("variant", "predictor_joiner")
    if variant == "none":
        return None

    embed, dec_rnn = _require_lstm_predictor(decoder)
    joint_hidden = joint.joint_net[-1].in_features
    common = dict(
        joint_module=joint,
        frame_chunk=loop_cfg.get("frame_chunk"),
        gradient_checkpointing=bool(loop_cfg.get("gradient_checkpointing", False)),
        num_passes=int(loop_cfg.get("count", 2)),
    )

    if variant == "joint_only":
        return LoopedJointOnly(
            enc_proj=joint.enc,
            pred_proj=joint.pred,
            joint_net=joint.joint_net,
            joint_hidden=joint_hidden,
            block_hidden=int(loop_cfg.get("block_hidden") or joint_hidden),
            **common,
        )
    if variant != "predictor_joiner":
        raise ValueError(f"unknown loop.variant {variant!r}")

    feedback = loop_cfg.get("feedback", "pre_vocabulary_hidden")
    if feedback not in FEEDBACK_ALIASES:
        raise ValueError(f"unknown loop.feedback {feedback!r}; expected one of {sorted(FEEDBACK_ALIASES)}")
    return LoopedPredictorJoint(
        dec_rnn=dec_rnn,
        enc_proj=joint.enc,
        pred_proj=joint.pred,
        joint_net=joint.joint_net,
        embed_dim=embed.embedding_dim,
        joint_hidden=joint_hidden,
        retain_prefix_anchor=bool(loop_cfg.get("retain_prefix_anchor", True)),
        feedback_mode=FEEDBACK_ALIASES[feedback],
        share_depth_weights=bool(loop_cfg.get("share_depth_weights", True)),
        feedback_scale=float(loop_cfg.get("feedback_scale", 0.01)),
        **common,
    )


def resolve_checkpoint(checkpoint: str) -> str:
    """Resolve the configured checkpoint to a concrete local ``.nemo`` file when possible.

    Cloud runs use a pre-downloaded artifact on shared storage: every rank must open the
    same bytes, and compute nodes are often offline or rate-limited.  Accepts a ``.nemo``
    file, a directory holding exactly one (an unpacked HF snapshot, say), or a hub/NGC
    model name.  A local path that does not exist fails here with the path echoed, rather
    than silently falling through to a download on all eight ranks.
    """
    # A hub name also contains "/" (``nvidia/stt_en_...``), so a bare slash cannot decide
    # this. Local means: a .nemo suffix, an explicitly rooted/relative path, or something
    # that actually exists on disk.
    path = Path(checkpoint).expanduser()
    looks_local = checkpoint.endswith(".nemo") or checkpoint[:1] in ("/", "~", ".") or path.exists()
    if not looks_local:
        return checkpoint  # a hub/NGC name: left to NeMo's own resolution and cache
    if path.is_file():
        return str(path)
    if path.is_dir():
        candidates = sorted(path.glob("*.nemo"))
        if len(candidates) == 1:
            return str(candidates[0])
        raise FileNotFoundError(
            f"expected exactly one .nemo file in {path}, found {len(candidates)}: "
            f"{[c.name for c in candidates]}. Point `checkpoint` at the file itself."
        )
    raise FileNotFoundError(
        f"checkpoint not found: {path}\n"
        "Set it to the pre-downloaded artifact, e.g.\n"
        "  checkpoint=/shared/checkpoints/stt_en_fastconformer_hybrid_large_streaming_multi.nemo\n"
        "or export RT_CHECKPOINT=<path> (configs read it as a default)."
    )


def load_asr_model(checkpoint: str, map_location: str = "cpu"):
    """Restore the pretrained hybrid RNN-T/CTC model from a local ``.nemo`` or a hub name."""
    from nemo.collections.asr.models import ASRModel  # local import: heavy dependency

    resolved = resolve_checkpoint(checkpoint)
    if resolved.endswith(".nemo"):
        return ASRModel.restore_from(restore_path=resolved, map_location=map_location)
    return ASRModel.from_pretrained(model_name=resolved, map_location=map_location)


class RecurrentTransducerModule(_ModuleBase):
    """Training/eval module: frozen encoder + (optionally looped) predictor-joiner.

    Data loading, preprocessing and the tokenizer are delegated to the restored NeMo
    model, so every experiment in the matrix sees byte-identical batches.

    Args:
        asr_model: restored ``EncDecHybridRNNTCTCBPEModel`` (or plain RNN-T model).
        cfg: the experiment config (see ``configs/base.yaml``).
    """

    def __init__(self, asr_model: nn.Module, cfg: Dict[str, Any]):
        super().__init__()
        self.asr = asr_model
        self.cfg = cfg
        loop_cfg = cfg.get("loop", {})
        self.num_passes = int(loop_cfg.get("count", 1))

        if cfg.get("encoder", {}).get("frozen", True):
            freeze_module(self.asr.encoder)
            freeze_module(self.asr.preprocessor)
        # The auxiliary CTC head stays frozen and outside the primary loss: with the
        # encoder fixed it cannot train the proposed modules (plan section 3.4).
        for attr in ("ctc_decoder", "ctc_loss"):
            head = getattr(self.asr, attr, None)
            if isinstance(head, nn.Module):
                freeze_module(head)

        _set_dropout(self.asr.decoder, float(cfg.get("predictor", {}).get("dropout", 0.0)))
        _set_dropout(self.asr.joint, float(cfg.get("joiner", {}).get("dropout", 0.0)))

        self.scorer = build_scorer(self.asr.decoder, self.asr.joint, loop_cfg)
        if cfg.get("predictor", {}).get("frozen", False):  # F2 control
            self.asr.decoder.requires_grad_(False)

        self.inactive_module_names = self._freeze_inactive_modules()
        # Cheap device-side moments: this runs at construction and again after training on a
        # 114M-parameter encoder, and the question it answers is "did an optimizer step move
        # this", not "prove nobody swapped the weights". run_gates.py uses the exact hash.
        self._encoder_fingerprint = module_fingerprint(self.asr.encoder, exact=False)
        self._val_records: List[Dict[str, Any]] = []

    # --- forward pieces ---

    def encode(self, input_signal: torch.Tensor, input_signal_length: torch.Tensor):
        """Frozen-encoder forward.  Returns ``([B, T, D], [B])`` with gradients detached."""
        with torch.no_grad():
            outputs = self.asr.forward(input_signal=input_signal, input_signal_length=input_signal_length)
        encoded, encoded_len = outputs[0], outputs[1]
        return encoded.transpose(1, 2).detach(), encoded_len.detach()

    def compute_logits(
        self, encoded: torch.Tensor, transcript: torch.Tensor, transcript_len: torch.Tensor, K: Optional[int] = None
    ) -> torch.Tensor:
        """``[B, T, U + 1, V + 1]`` logits under the configured variant.

        ``variant: none`` reproduces NeMo's own training path exactly, so B1 differs from
        the proposal only in the scoring function -- not in data, exposure or optimizer.
        """
        if self.scorer is None:
            decoder_outputs, _, _ = self.asr.decoder(targets=transcript, target_length=transcript_len)
            return self.asr.joint(encoder_outputs=encoded.transpose(1, 2), decoder_outputs=decoder_outputs)

        prefix = collect_prefix_cache(
            self.asr.decoder.prediction["embed"],
            self.asr.decoder.prediction["dec_rnn"],
            transcript,
            transcript_len,
        )
        return self.scorer.score_cells(encoded, prefix, K=K if K is not None else self.num_passes)

    # --- Lightning hooks ---

    def training_step(self, batch, batch_idx):
        signal, signal_len, transcript, transcript_len = batch
        encoded, encoded_len = self.encode(signal, signal_len)
        logits = self.compute_logits(encoded, transcript, transcript_len)
        loss = self.asr.loss(
            log_probs=logits, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        self._log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        signal, signal_len, transcript, transcript_len = batch
        encoded, encoded_len = self.encode(signal, signal_len)
        logits = self.compute_logits(encoded, transcript, transcript_len)
        loss = self.asr.loss(
            log_probs=logits, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        hypotheses, cap_hits = self.transcribe_encoded(encoded, encoded_len)
        references = self.decode_references(transcript, transcript_len)
        errors, words = self._accumulate_errors(hypotheses, references, dataloader_idx)
        self._log("val_loss", loss)
        return {"val_loss": loss, "errors": errors, "words": words, "cap_hits": cap_hits}

    def on_validation_epoch_start(self):
        self._val_records = []

    def on_validation_epoch_end(self):
        """Pooled WER = pooled word errors / pooled reference words across dev sets.

        The plan selects checkpoints on the pooled figure while still reporting each dev
        set separately, so both are logged here.

        Counts are summed across ranks BEFORE dividing.  Each rank sees a different shard
        of the dev sets, so a per-rank ratio -- or a mean of per-rank ratios -- is not the
        pooled WER the plan selects on, and would make the monitored metric depend on how
        utterances happened to shard.
        """
        if not self._val_records:
            return
        indices = sorted({r["dataloader_idx"] for r in self._val_records})
        counts = torch.tensor(
            [
                [
                    sum(r["errors"] for r in self._val_records if r["dataloader_idx"] == idx),
                    sum(r["words"] for r in self._val_records if r["dataloader_idx"] == idx),
                ]
                for idx in indices
            ],
            dtype=torch.float64,
            device=self.device if hasattr(self, "device") else "cpu",
        )
        counts = self._reduce_counts(counts)
        self._log("val_wer_pooled", (counts[:, 0].sum() / counts[:, 1].sum().clamp(min=1)).float(), prog_bar=True)
        for row, idx in enumerate(indices):
            self._log(f"val_wer_{idx}", (counts[row, 0] / counts[row, 1].clamp(min=1)).float())

    def _reduce_counts(self, counts: torch.Tensor) -> torch.Tensor:
        """Sum ``[num_dev_sets, 2]`` (errors, words) counts across all ranks."""
        if _LightningModule is None or not isinstance(self, _LightningModule) or self.trainer is None:
            return counts
        if self.trainer.world_size <= 1:
            return counts
        return self.all_gather(counts).sum(dim=0)

    def configure_optimizers(self):
        """Two param groups: pretrained modules and newly initialised ones (plan section 7)."""
        opt_cfg = self.cfg.get("optimizer", {})
        pretrained, new = self.parameter_groups()
        groups = []
        if pretrained:
            groups.append({"params": pretrained, "lr": float(opt_cfg.get("pretrained_lr", 1e-4))})
        if new:
            groups.append({"params": new, "lr": float(opt_cfg.get("new_module_lr", 3e-4))})
        optimizer = torch.optim.AdamW(
            groups,
            betas=tuple(opt_cfg.get("betas", (0.9, 0.98))),
            weight_decay=float(opt_cfg.get("weight_decay", 1e-3)),
        )
        sched_cfg = opt_cfg.get("sched")
        if not sched_cfg:
            return optimizer

        # NeMo's own scheduler, so the warmup/anneal shape matches every other recipe here.
        from nemo.core.optim.lr_scheduler import get_scheduler

        scheduler = get_scheduler(sched_cfg.get("name", "CosineAnnealing"))(
            optimizer,
            max_steps=int(self.cfg.get("training", {}).get("max_updates", 10000)),
            warmup_ratio=sched_cfg.get("warmup_ratio"),
            min_lr=float(sched_cfg.get("min_lr", 0.0)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def setup_data(self) -> None:
        """Build train/validation dataloaders through the restored model's own setup.

        Reusing ``setup_training_data`` / ``setup_multiple_validation_data`` keeps the
        manifest parsing, bucketing, tokenizer and collation byte-identical to every other
        NeMo recipe -- and therefore identical across the experiment matrix.
        """
        data_cfg = self.cfg.get("data", {})
        if data_cfg.get("train_ds") is not None:
            self.asr.setup_training_data(data_cfg["train_ds"])
        if data_cfg.get("validation_ds") is not None:
            self.asr.setup_multiple_validation_data(data_cfg["validation_ds"])

    def train_dataloader(self):
        return self.asr._train_dl

    def val_dataloader(self):
        return getattr(self.asr, "_validation_dl", None)

    # --- helpers used by scripts, tests and the reporting tables ---

    def _freeze_inactive_modules(self) -> List[str]:
        """Freeze scorer submodules the configured variant can never reach.

        Without this, B2 (``K=1``) and N2 (zero feedback) carry trainable parameters that
        receive no gradient.  On one GPU that is merely dead weight in the optimizer; under
        DDP it hangs the all-reduce and crashes the first step.  Freezing is preferred over
        ``find_unused_parameters=True``, which would silence this for every variant and cost
        a full parameter traversal per step.
        """
        if self.scorer is None:
            return []
        names = []
        for name, module in self.scorer.inactive_modules(self.num_passes):
            module.requires_grad_(False)
            names.append(name)
        return names

    def decoder_infer(self, num_passes: Optional[int] = None) -> LoopedGreedyRNNTInfer:
        """Greedy decoder bound to this model's scorer and canonical predictor."""
        if self.scorer is None:
            raise RuntimeError("variant 'none' decodes through the stock NeMo decoding path")
        return LoopedGreedyRNNTInfer(
            scorer=self.scorer,
            embed=self.asr.decoder.prediction["embed"],
            dec_rnn=self.asr.decoder.prediction["dec_rnn"],
            blank_index=self.asr.decoder.blank_idx,
            max_symbols_per_step=self.cfg.get("decoding", {}).get("max_symbols_per_frame", 10),
            num_passes=num_passes if num_passes is not None else self.num_passes,
        )

    def parameter_groups(self):
        """``(pretrained_trainable, newly_initialised_trainable)`` parameter lists."""
        new_ids = {id(p) for p in self.scorer.extra_parameters()} if self.scorer is not None else set()
        pretrained, new = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (new if id(p) in new_ids else pretrained).append(p)
        return pretrained, new

    def parameter_report(self) -> Dict[str, Any]:
        """Stored / trainable / newly-added parameter counts for the results tables.

        Stored and trainable differ for B2 and N2, whose feedback module is allocated for
        architecture parity but frozen because the variant never reaches it.  Both numbers
        are reported so a parameter-efficiency claim cannot quietly use the wrong one.
        """
        pretrained, new = self.parameter_groups()
        return {
            "stored_total": count_parameters([self]),
            "trainable_pretrained": sum(p.numel() for p in pretrained),
            "trainable_new": sum(p.numel() for p in new),
            "encoder_frozen": count_parameters([self.asr.encoder]),
            "frozen_because_unused": self.inactive_module_names,
        }

    def assert_encoder_unchanged(self) -> None:
        """Raise if any encoder parameter or mutable buffer moved (plan section 5 gate)."""
        current = module_fingerprint(self.asr.encoder, exact=False)
        if current != self._encoder_fingerprint:
            raise AssertionError("frozen encoder changed: parameters or buffers were updated")

    def transcribe_encoded(self, encoded: torch.Tensor, encoded_len: torch.Tensor):
        """Decode encoder outputs to text.  Returns ``(hypotheses, cap_hits)``.

        ``variant: none`` goes through NeMo's own ``rnnt_decoder_predictions_tensor`` so the
        B1 baseline is decoded by the stock, optimized implementation; the looped variants
        use :class:`LoopedGreedyRNNTInfer`.  The plan requires reporting runtime for a
        common backend first and the native baseline separately -- never mixing the two in
        one comparison.
        """
        if self.scorer is None:
            best = self.asr.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded.transpose(1, 2), encoded_lengths=encoded_len, return_hypotheses=True
            )
            return [hyp.text for hyp in best], 0

        hyps, stats = self.decoder_infer().decode(encoded, encoded_len)
        return [self.asr.tokenizer.ids_to_text(h.y_sequence) for h in hyps], stats.cap_hits

    def decode_references(self, transcript: torch.Tensor, transcript_len: torch.Tensor):
        """Detokenize the reference targets with the checkpoint's own tokenizer."""
        return [
            self.asr.tokenizer.ids_to_text(transcript[i, : int(transcript_len[i])].tolist())
            for i in range(transcript.shape[0])
        ]

    def _accumulate_errors(self, hypotheses, references, dataloader_idx: int):
        # NeMo's own scorer, so WER here is defined exactly as everywhere else in the repo.
        from nemo.collections.asr.metrics.wer import word_error_rate_detail

        wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses, references)
        errors = int(round(wer * words))
        self._val_records.append(
            {
                "errors": errors,
                "words": words,
                "dataloader_idx": dataloader_idx,
                "ins": ins_rate * words,
                "del": del_rate * words,
                "sub": sub_rate * words,
            }
        )
        return errors, words

    def _log(self, name: str, value, **kwargs) -> None:
        if _LightningModule is not None and isinstance(self, _LightningModule) and self.trainer is not None:
            self.log(name, value, **kwargs)


def _require_lstm_predictor(decoder: nn.Module):
    """Fail loudly if the checkpoint is not the LSTM predictor the plan assumes.

    The plan explicitly says to stop the compatibility route rather than silently
    substitute a different predictor.
    """
    prediction = getattr(decoder, "prediction", None)
    if prediction is None or "embed" not in prediction or "dec_rnn" not in prediction:
        raise TypeError(
            f"expected a NeMo RNNTDecoder with prediction['embed'|'dec_rnn'], got {type(decoder).__name__}. "
            "This checkpoint does not use the LSTM predictor the compatibility route assumes."
        )
    return prediction["embed"], prediction["dec_rnn"]


def _set_dropout(module: nn.Module, p: float) -> None:
    """Override every ``nn.Dropout`` rate, and any RNN ``dropout`` attribute, in ``module``."""
    for sub in module.modules():
        if isinstance(sub, nn.Dropout):
            sub.p = p
        if isinstance(sub, nn.RNNBase):
            sub.dropout = p
