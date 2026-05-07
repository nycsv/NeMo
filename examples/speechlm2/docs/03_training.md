# 학습 설정 가이드

## 파일 구성

```
examples/speechlm2/
├── train.sh                   ← 실행 진입점
├── finetune_canary_qwen.py    ← 학습 스크립트 (Hydra + PTL)
└── canary_qwen_finetune.yaml  ← 전체 설정 파일
```

---

## 빠른 시작

### 1. 체크포인트 준비

```bash
python convert.py   # canary-qwen-2.5b.nemo 생성
```

### 2. 데이터 경로 설정 (`canary_qwen_finetune.yaml`)

```yaml
data:
  train_ds:
    input_cfg:
      - type: lhotse_as_conversation
        input_cfg: /path/to/input_cfg.yaml   # ← 수정

  validation_ds:
    datasets:
      my_devset:
        input_cfg:
          - type: lhotse_as_conversation
            cuts_path: /path/to/val_cuts.jsonl.gz  # ← 수정
```

### 3. 학습 실행

```bash
chmod +x train.sh
./train.sh
```

---

## 체크포인트 로딩 흐름

`finetune_canary_qwen.py`의 가중치 적용 순서:

```
Step 1. canary_qwen_pretrained_path  (전체 모델 baseline)
            ↓
Step 2. module_weights.llm           (LLM 교체, 선택)
        module_weights.embed_tokens  (임베딩 교체, 선택)
        module_weights.perception    (perception 전체 교체, 선택)
        module_weights.perception_from_asr  (encoder+preprocessor 교체, 선택)
```

각 단계는 앞 단계 결과 위에 덮어씁니다.  
`module_weights`가 하나도 설정되지 않으면 Step 1만 실행됩니다.  
`canary_qwen_pretrained_path`와 `module_weights` 중 적어도 하나는 반드시 설정해야 합니다.

---

## 주요 설정 옵션

### 체크포인트

```yaml
model:
  # 전체 모델 체크포인트 (baseline)
  canary_qwen_pretrained_path: /path/to/canary-qwen-2.5b.nemo

  # 모듈별 체크포인트 (선택, 각각 위 경로보다 우선 적용)
  module_weights:
    llm:                 null
    embed_tokens:        null
    perception:          null
    perception_from_asr: null
```

### 가중치 로딩 모드

```yaml
model:
  # False: canary_qwen_pretrained_path에서 가중치 로드 (fine-tuning 기본값)
  # True:  pretrained_llm + pretrained_asr에서 가중치 로드 (scratch 학습)
  pretrained_weights: False
```

### 동결 파라미터

```yaml
model:
  freeze_params:
    - "^llm\\..+$"          # LLM backbone 동결
    - "^embed_tokens\\..+$" # embedding 동결
  prevent_freeze_params: []  # 동결 예외 패턴
```

### LoRA

```yaml
model:
  lora:
    task_type: CAUSAL_LM
    r: 128
    lora_alpha: 256
    lora_dropout: 0.01
    target_modules: ["q_proj", "v_proj"]
```

### 트레이너

```yaml
trainer:
  devices: -1            # 전체 GPU 사용 (-1) 또는 숫자로 지정
  num_nodes: 1
  precision: bf16-true
  max_steps: 10000
  gradient_clip_val: 1.0

  strategy:
    _target_: lightning.pytorch.strategies.DDPStrategy
    find_unused_parameters: true   # LoRA + frozen 파라미터 조합 시 필수
```

### 실험 관리

```yaml
exp_manager:
  explicit_log_dir: ./results
  create_tensorboard_logger: true
  create_checkpoint_callback: true
  checkpoint_callback_params:
    monitor: val_wer
    mode: min
    save_top_k: 3
  resume_if_exists: true
```

---

## train.sh 주요 변수

```bash
CUDA_VISIBLE="0,1,2,3"          # 사용할 GPU ID
NUM_GPUS=4                       # GPU 수 (CUDA_VISIBLE와 일치해야 함)
CKPT="${SCRIPT_DIR}/canary-qwen-2.5b.nemo"   # 체크포인트 경로
TRAIN_INPUT_CFG="${SCRIPT_DIR}/input_cfg.yaml"
VAL_CUTS="/path/to/val_cuts.jsonl.gz"
RESULTS_DIR="${SCRIPT_DIR}/results"
```

---

## CLI 오버라이드 패턴

YAML 파일을 수정하지 않고 CLI에서 직접 설정을 변경할 수 있습니다.

```bash
torchrun ... finetune_canary_qwen.py \
  trainer.devices=2 \
  trainer.precision=bf16-true \
  model.pretrained_weights=False \
  model.canary_qwen_pretrained_path=/ckpts/canary-qwen-2.5b.nemo \
  model.module_weights.perception_from_asr=nvidia/canary-1b-v2 \
  "data.train_ds.input_cfg[0].input_cfg=/data/input_cfg.yaml" \
  exp_manager.explicit_log_dir=/results/run1
```

---

## 지원 체크포인트 포맷

`canary_qwen_pretrained_path` 및 `module_weights.*`에서 허용되는 포맷:

| 포맷 | 예시 | 비고 |
|---|---|---|
| `.nemo` (권장) | `canary-qwen-2.5b.nemo` | `convert.py`가 생성 |
| `.ckpt` | `model_weights.ckpt` | 순수 state dict |

`perception_from_asr`는 추가로 HF Hub ID와 NGC 이름도 지원합니다. → [04_pretrained_models.md](./04_pretrained_models.md)

---

## 관련 파일

| 파일 | 역할 |
|---|---|
| `train.sh` | torchrun 실행 진입점 |
| `finetune_canary_qwen.py` | 학습 스크립트 본체 |
| `canary_qwen_finetune.yaml` | 전체 설정 |
| `docs/01_convert_to_nemo.md` | 체크포인트 변환 |
| `docs/02_module_checkpoints.md` | 모듈별 체크포인트 |
| `docs/04_pretrained_models.md` | pretrained_asr / pretrained_llm 설정 |
