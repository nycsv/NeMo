# Pretrained 모델 활용 가이드

## 지원 모델 소스

### `pretrained_asr` — NeMo ASR 모델

세 가지 형식으로 지정 가능합니다.

| 형식 | 예시 | 동작 |
|---|---|---|
| HuggingFace Hub ID | `nvidia/canary-1b-v2` | `.nemo` 자동 다운로드 및 캐시 |
| NGC 모델 이름 | `stt_en_fastconformer_hybrid_large_streaming_80ms` | NGC 카탈로그에서 다운로드 |
| 로컬 `.nemo` 경로 | `/ckpts/canary-1b-v2.nemo` | 직접 로드 |

### `pretrained_llm` — HuggingFace LLM

```yaml
pretrained_llm: Qwen/Qwen3-1.7B     # 기본값
pretrained_llm: Qwen/Qwen2.5-7B-Instruct
pretrained_llm: meta-llama/Llama-3.2-3B
```

---

## HuggingFace Hub 자동 다운로드 동작 원리

`pretrained_asr: nvidia/canary-1b-v2` 를 설정하면 NeMo는 다음 순서로 처리합니다.

```
1. HF Hub 캐시 확인  →  캐시 히트 시 즉시 반환
2. nvidia/canary-1b-v2 repo에서 canary-1b-v2.nemo 파일 검색
3. .nemo 파일 존재  →  hf_hub_download()로 단일 파일 다운로드
4. .nemo 파일 없음  →  snapshot_download()로 전체 repo 다운로드
5. 로컬 캐시 경로를 ASRModel.restore_from()에 전달
```

캐시 위치: `~/.cache/huggingface/hub/`

---

## 두 가지 학습 워크플로우

### 워크플로우 A: Fine-tuning (pretrained_weights: False)

기존 `canary-qwen-2.5b` 체크포인트를 시작점으로, 특정 모듈만 다른 가중치로 교체합니다.

**`pretrained_asr`의 역할**: 아키텍처 정의 전용 (가중치 미사용)  
→ `pretrained_weights: False` 이므로 ASR 모델 다운로드 없음

```yaml
model:
  pretrained_llm: Qwen/Qwen3-1.7B
  pretrained_asr: nvidia/canary-1b-flash   # 아키텍처 config만 사용
  pretrained_weights: False

  canary_qwen_pretrained_path: canary-qwen-2.5b.nemo  # 전체 가중치 로드

  module_weights:
    perception_from_asr: nvidia/canary-1b-v2  # encoder만 v2로 교체
```

### 워크플로우 B: Scratch 학습 (pretrained_weights: True)

`pretrained_llm`과 `pretrained_asr`에서 가중치를 직접 로드해 SALM을 조립합니다.

**`pretrained_asr`의 역할**: 아키텍처 config + 가중치 모두 사용  
→ 아키텍처 YAML 수동 작성 불필요 (ASR 모델 config로 자동 설정)

```yaml
model:
  pretrained_llm: Qwen/Qwen3-1.7B
  pretrained_asr: nvidia/canary-1b-v2   # config + 가중치 모두 로드
  pretrained_weights: True

  canary_qwen_pretrained_path: null     # 전체 ckpt 불필요
```

---

## canary-1b-v2 사용 패턴별 정리

### 패턴 1: encoder만 v2로 교체하고 fine-tuning

```yaml
pretrained_weights: False
canary_qwen_pretrained_path: canary-qwen-2.5b.nemo
module_weights:
  perception_from_asr: nvidia/canary-1b-v2
```

### 패턴 2: v2 encoder로 SALM 처음부터 조합

```yaml
pretrained_weights: True
pretrained_asr: nvidia/canary-1b-v2
pretrained_llm: Qwen/Qwen3-1.7B
canary_qwen_pretrained_path: null
```

### 패턴 3: 로컬 .nemo로 저장 후 오프라인 사용

`convert_asr.py`로 HF Hub 모델을 로컬 `.nemo` 파일로 변환합니다.

```bash
# canary-1b-v2 다운로드 → canary-1b-v2.nemo 저장
python convert_asr.py nvidia/canary-1b-v2

# 출력 경로 지정
python convert_asr.py nvidia/canary-1b-v2 --output /ckpts/canary-1b-v2.nemo
```

이후 YAML에서 로컬 경로 사용:
```yaml
pretrained_asr: /ckpts/canary-1b-v2.nemo       # scratch 학습
perception_from_asr: /ckpts/canary-1b-v2.nemo  # encoder 교체 fine-tuning
```

### 패턴 4: encoder 추출 후 재사용

`.nemo`로 저장한 뒤 encoder만 분리하면 더 작은 파일로 반복 활용 가능합니다.

```bash
python convert_asr.py nvidia/canary-1b-v2 --output canary-1b-v2.nemo
python convert_modules.py --from-asr canary-1b-v2.nemo --output-dir /ckpts
# → /ckpts/canary-1b-v2_asr_encoder.nemo  (encoder + preprocessor만 포함)
```

이후 학습에서 재사용:
```yaml
module_weights:
  perception_from_asr: /ckpts/canary-1b-v2_asr_encoder.nemo
```

---

## `resolve_nemo_path()` 유틸리티

`pretrained.py`에 추가된 함수로, 모든 소스 형식을 로컬 경로로 변환합니다.  
`perception_from_asr` 로딩 내부에서 자동으로 호출됩니다.

```python
from nemo.collections.speechlm2.parts.pretrained import resolve_nemo_path

# HF Hub ID → 로컬 .nemo 경로 (자동 다운로드)
path = resolve_nemo_path("nvidia/canary-1b-v2")

# 로컬 경로 → 그대로 반환
path = resolve_nemo_path("/ckpts/my_asr.nemo")

# NGC 이름 → 로컬 경로 (자동 다운로드)
path = resolve_nemo_path("stt_en_fastconformer_hybrid_large_streaming_80ms")
```

---

## 주요 ASR 모델 목록

| 모델 | HF Hub ID | 특징 |
|---|---|---|
| Canary-1B Flash | `nvidia/canary-1b-flash` | 기본값, 다국어 (en/fr/de/es) |
| Canary-1B v2 | `nvidia/canary-1b-v2` | 최신 버전 |
| Canary-180M Flash | `nvidia/canary-180m-flash` | 경량 버전 |
| FastConformer Large Streaming | `stt_en_fastconformer_hybrid_large_streaming_80ms` | 스트리밍 전용 |

---

## 관련 파일

| 파일 | 역할 |
|---|---|
| `convert_asr.py` | ASR 모델 다운로드 → 로컬 `.nemo` 저장 |
| `nemo/collections/speechlm2/parts/pretrained.py` | `resolve_nemo_path()`, `load_pretrained_nemo()`, `setup_speech_encoder()` |
| `canary_qwen_finetune.yaml` | `pretrained_asr`, `pretrained_llm`, `module_weights` 설정 |
| `finetune_canary_qwen.py` | `_load_encoder_keys_from_nemo()` — HF Hub ID 지원 |
| `docs/02_module_checkpoints.md` | 모듈별 분리 및 encoder 추출 |
| `docs/03_training.md` | 학습 전체 설정 |
