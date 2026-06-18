# Streaming BEST-RQ SSL experiment — run scripts

Self-contained scripts to run the experiment: **A** (offline SSL) vs **B0** (encoder-only
streaming) vs **B\*** (proposed streaming BEST-RQ with dual-mode consistency), pretrained on
**Libriheavy**, finetuned on **train-clean-100** (streaming Hybrid RNNT+CTC), evaluated across
chunk sizes.

## 1. Edit data paths — only `env.sh`
All data/compute knobs live in `env.sh` (manifests, tokenizer, output dir, GPUs, precision,
step counts, the eval chunk sweep). Fill in every `???`. Nothing else needs editing.

## 2. Run
```bash
cd examples/asr/speech_pretraining/streaming_ssl_experiment

# --- pretrain (Libriheavy) ---  (run on your cluster; long)
./01_pretrain_A_offline.sh
./01_pretrain_B0_streaming.sh
./01_pretrain_Bstar_dualmode.sh
# ./01_pretrain_FSQ_streaming.sh        # optional later arm

# --- finetune (train-clean-100) ---  auto-finds the arm's newest SSL .nemo
./02_finetune.sh A_offline
./02_finetune.sh B0_streaming
./02_finetune.sh Bstar_dualmode

# --- evaluate across chunk sizes ---  prints WER per att_context_size
./03_eval_sweep.sh A_offline
./03_eval_sweep.sh B0_streaming
./03_eval_sweep.sh Bstar_dualmode
```
Or drive all three arms end-to-end: `./run_all.sh`.

## Notes
- Every script sources `env.sh`; extra Hydra overrides can be appended, e.g.
  `./01_pretrain_Bstar_dualmode.sh model.dual_mode.beta=0.5 trainer.max_steps=200000`.
- `02_finetune.sh <ARM>` auto-locates the newest `exp/ssl/<ARM>/**/*.nemo`. Override with
  `SSL_NEMO=/path/to.nemo ./02_finetune.sh <ARM>`.
- `03_eval_sweep.sh <ARM>` auto-locates the newest `exp/ft/<ARM>/**/*.nemo`; or pass a `.nemo`
  path and optionally a test manifest: `./03_eval_sweep.sh model.nemo /data/test_other.json`.
- ARM names (must match across scripts): `A_offline`, `B0_streaming`, `Bstar_dualmode`,
  `FSQ_streaming`.

## Reading the result
For each finetuned model the eval prints WER (RNNT + CTC) per chunk size. Compare across arms:
- **B0 − A**: does streaming pretraining help at all?
- **B\* − B0**: does the proposed dual-mode design beat naive streaming? (the hypothesis)
Expected: `B* ≤ B0 ≤ A` at every chunk size, largest gains at the smallest chunks (`[70,1]`,
`[70,0]`), no regression in the offline column.

See `../../conf/ssl/nest/STREAMING_SSL_EXPERIMENT.md` for the full design write-up.

## Prereq
`pip install kaldialign` (optional NeMo dep currently missing in this env; required for
`import nemo.collections.asr`).
