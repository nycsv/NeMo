# Streaming BEST-RQ SSL (dual-mode) → Streaming Hybrid ASR — run guide

Research question: does pretraining the SSL encoder for streaming help downstream streaming ASR
**across all chunk sizes**, and does a *designed* streaming BEST-RQ (dual-mode consistency) beat a
naive encoder-only streaming baseline?

Data: **Libriheavy** for SSL pretraining, **train-clean-100** for finetuning. Non-denoise
masked-token-prediction (no noise corpus needed; transcripts ignored during SSL).
Encoder = Large (120M) FastConformer, subsampling 8 (1 enc frame = 80 ms). Shared dynamic
multi-context set `[[70,13],[70,6],[70,1],[70,0]]` (look-ahead 1.04s / 0.48s / 0.08s / 0s).

## Arms (A vs B0 vs B-proposed)
| Arm | SSL pretrain config | Encoder | Objective |
|-----|---------------------|---------|-----------|
| A  (offline baseline)        | `nest_fast-conformer_offline.yaml`            | offline/bidir | masked-token-pred |
| B0 (encoder-only streaming)  | `nest_fast-conformer_streaming.yaml`          | streaming dyn-chunk | masked-token-pred |
| B* (proposed)                | `nest_fast-conformer_dualmode_streaming.yaml` | streaming dyn-chunk | **+ dual-mode consistency** |
| (later) FSQ tokenizer        | `nest_fast-conformer_streaming_fsq.yaml`      | streaming dyn-chunk | masked-token-pred, FSQ targets |

Proposed method (B*): each step encodes the same masked input twice with the shared encoder —
teacher (largest context) + student (sampled smaller context) — and trains
`L = L_mlm(student) + alpha*L_mlm(teacher) + beta*KL(teacher||student)` on masked frames.
Code: `EncDecMaskedTokenPredDualModeModel` (ssl_models.py), `MaskedConsistencyLoss` (ssl_losses/mlm.py).

## New/changed files
- `nemo/collections/asr/models/ssl_models.py` — `EncDecMaskedTokenPredDualModeModel`
- `nemo/collections/asr/losses/ssl_losses/mlm.py` — `MaskedConsistencyLoss`
- `nemo/collections/asr/modules/ssl_modules/quantizers.py` — `FSQVectorQuantizer` (FSQ arm)
- `examples/asr/speech_pretraining/masked_token_pred_pretrain_streaming.py` — entry script (selects model via `ssl_model_class`)
- `examples/asr/conf/ssl/nest/nest_fast-conformer_{streaming,offline,dualmode_streaming,streaming_fsq}.yaml`
- `examples/asr/conf/fastconformer/hybrid_cache_aware_streaming/ssl_streaming_hybrid_finetune.yaml`

## Pretraining (Libriheavy)
```
# A (offline baseline)
python examples/asr/speech_pretraining/masked_token_pred_pretrain_streaming.py \
  --config-path=../conf/ssl/nest --config-name=nest_fast-conformer_offline \
  model.train_ds.manifest_filepath=<libriheavy_train.json> \
  model.validation_ds.manifest_filepath=<dev.json>

# B0 (encoder-only streaming)   -> --config-name=nest_fast-conformer_streaming
# B* (proposed dual-mode)       -> --config-name=nest_fast-conformer_dualmode_streaming
# FSQ arm (later)               -> --config-name=nest_fast-conformer_streaming_fsq
```
(`ssl_model_class` is set inside each config; the dual-mode config selects
`masked_token_pred_dualmode`. Tune `model.dual_mode.{alpha,beta}` if needed.)

## Finetuning (train-clean-100, streaming Hybrid RNNT+CTC)
```
python examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py \
  --config-path=../conf/fastconformer/hybrid_cache_aware_streaming \
  --config-name=ssl_streaming_hybrid_finetune \
  init_from_nemo_model.encoder.path=<ssl_arm.nemo> \
  model.tokenizer.dir=<tokenizer_dir> model.tokenizer.type=bpe \
  model.train_ds.manifest_filepath=<train_clean_100.json> \
  model.validation_ds.manifest_filepath=<dev_clean.json>
```
Run once per arm, swapping `init_from_nemo_model.encoder.path` (A / B0 / B* .nemo). The finetune
encoder uses the same dynamic context set so one model serves all chunk sizes. (train-clean-100 is
small — consider lowering `trainer.max_steps` and `optim.sched.warmup_steps`.)

## Evaluation sweep (per finetuned model)
```
for CTX in "[70,13]" "[70,6]" "[70,1]" "[70,0]"; do
  python examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
    model_path=<finetuned.nemo> dataset_manifest=<test.json> \
    att_context_size=$CTX batch_size=16 compare_vs_offline=true
done
```
Primary readouts (RNNT + CTC):
- **B0 − A** per chunk: does streaming pretraining help at all?
- **B* − B0** per chunk: does the proposed dual-mode design beat naive streaming? (key hypothesis)
Success = B* ≤ B0 ≤ A at every chunk size, largest gains at [70,1]/[70,0], no offline regression.

## Verification gates (smoke test before full runs)
- Short pretrain on a Libriheavy subset per arm; confirm val_loss decreases. For B*, confirm the
  three logged losses appear: train_loss_student / _teacher / _consistency, and consistency > 0.
- Finetune init: check the partial-load log (encoder.* loaded, RNNT/CTC fresh).
- Streaming infer on a tiny manifest first; finite WER + latency per CTX.

## Env note
`nemo.collections.asr` import needs optional dep `kaldialign` (`pip install kaldialign`) —
pre-existing, unrelated to these changes. Configs were validated via Hydra compose; the new loss
and FSQ math were validated via direct module import.
