# 모듈별 체크포인트 분리 및 사용

## 개요

`convert_modules.py`는 두 가지 기능을 제공합니다.

| 모드 | 입력 | 출력 |
|---|---|---|
| **split** (기본) | SALM `.nemo` | 모듈별 `.nemo` 3개 |
| **extract-encoder** (`--from-asr`) | ASR `.nemo` | encoder+preprocessor `.nemo` |

---

## SALM 모델의 모듈 구조

```
SALM
├── llm             ← Qwen transformer layers + lm_head
├── embed_tokens    ← token embedding (FSDP/TP 때문에 llm 밖으로 분리)
└── perception      ← 오디오 인식 모듈
    ├── preprocessor    ← mel spectrogram 전처리기
    ├── encoder         ← FastConformer encoder
    ├── modality_adapter
    └── proj            ← 차원 변환 (1024 → 2048)
```

---

## 모드 1: SALM .nemo → 모듈별 분리

### 실행

```bash
python convert_modules.py canary-qwen-2.5b.nemo
# 또는 출력 디렉터리 지정
python convert_modules.py canary-qwen-2.5b.nemo --output-dir /ckpts/modules
```

### 출력 파일

```
canary-qwen-2.5b_llm.nemo
canary-qwen-2.5b_embed_tokens.nemo
canary-qwen-2.5b_perception.nemo
```

### 파일 내부 구조

각 `.nemo`는 tarball이며 **모듈 prefix가 제거된** state dict를 담습니다.

예) `canary-qwen-2.5b_perception.nemo` 내 키:
```
preprocessor.featurizer.fb          # ← perception. prefix 없음
encoder.layers.0.self_attn.linear_q.weight
modality_adapter.linear.weight
proj.weight
```

예) `canary-qwen-2.5b_llm.nemo` 내 키:
```
model.layers.0.self_attn.q_proj.weight   # ← llm. prefix 없음
lm_head.weight
...
```

---

## 모드 2: ASR .nemo → encoder 추출

기존에 학습된 FastConformer ASR 모델에서 encoder와 preprocessor만 추출합니다.

### 실행

```bash
python convert_modules.py --from-asr my_finetuned_asr.nemo
# 또는
python convert_modules.py --from-asr my_finetuned_asr.nemo --output-dir /ckpts
```

### 출력 파일

```
my_finetuned_asr_asr_encoder.nemo
```

내부 키: `encoder.*`, `preprocessor.*` 만 포함 (decoder, joint 등 제외)

---

## 로딩 우선순위

학습 시 가중치 적용 순서:

```
module_weights.X  >  canary_qwen_pretrained_path  >  random init
```

| 설정 조합 | 결과 |
|---|---|
| `canary_qwen_pretrained_path` 만 | 전체 SALM 가중치 로드 |
| `+ module_weights.perception` | 전체 로드 후 perception 전체 교체 |
| `+ module_weights.perception_from_asr` | 전체 로드 후 encoder+preprocessor 만 교체 |
| `perception` + `perception_from_asr` 동시 | `perception` 우선 (`perception_from_asr` 무시) |

---

## 사용 예시 (train.sh / YAML CLI)

### 예시 A: perception 모듈만 교체

```bash
torchrun ... finetune_canary_qwen.py \
  model.canary_qwen_pretrained_path=canary-qwen-2.5b.nemo \
  model.module_weights.perception=/ckpts/canary-qwen-2.5b_perception.nemo
```

### 예시 B: LLM만 교체 (다른 버전의 LLM 가중치 사용)

```bash
torchrun ... finetune_canary_qwen.py \
  model.canary_qwen_pretrained_path=canary-qwen-2.5b.nemo \
  model.module_weights.llm=/ckpts/updated_llm.nemo \
  model.module_weights.embed_tokens=/ckpts/updated_embed_tokens.nemo
```

### 예시 C: 전체 checkpoint 없이 모듈 조합

```bash
torchrun ... finetune_canary_qwen.py \
  model.canary_qwen_pretrained_path=null \
  model.module_weights.llm=/ckpts/canary-qwen-2.5b_llm.nemo \
  model.module_weights.embed_tokens=/ckpts/canary-qwen-2.5b_embed_tokens.nemo \
  model.module_weights.perception=/ckpts/canary-qwen-2.5b_perception.nemo
```

---

## YAML 설정 (`canary_qwen_finetune.yaml`)

```yaml
model:
  module_weights:
    llm: null                  # e.g. /ckpts/canary-qwen-2.5b_llm.nemo
    embed_tokens: null         # e.g. /ckpts/canary-qwen-2.5b_embed_tokens.nemo
    perception: null           # e.g. /ckpts/canary-qwen-2.5b_perception.nemo
    perception_from_asr: null  # e.g. nvidia/canary-1b-v2  or  /ckpts/my_asr.nemo
```

---

## 관련 파일

| 파일 | 역할 |
|---|---|
| `convert_modules.py` | 분리/추출 스크립트 |
| `finetune_canary_qwen.py` | `apply_module_weights()` 로딩 로직 |
| `canary_qwen_finetune.yaml` | `module_weights` 설정 섹션 |
| `docs/04_pretrained_models.md` | `perception_from_asr`에서 HF Hub 사용 |
