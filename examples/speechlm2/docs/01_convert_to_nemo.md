# HuggingFace 체크포인트 → .nemo 변환

## 배경

`nvidia/canary-qwen-2.5b` 모델은 HuggingFace Hub에 **safetensors + config.json** 형식으로 배포됩니다.  
NeMo 학습 파이프라인에서 사용하려면 **.nemo** 포맷으로 변환해야 합니다.

### .nemo 포맷이란?

`.nemo` 파일은 **단순 tarball** 로, 두 파일을 포함합니다.

```
canary-qwen-2.5b.nemo
├── model_config.yaml   ← OmegaConf로 직렬화된 모델 설정
└── model_weights.ckpt  ← torch.save(model.state_dict(), ...)  순수 state dict
```

> **PyTorch Lightning 버전 오류의 원인**  
> 이전 스크립트는 `torch.save({"state_dict": ..., "config": ...}, path)` 형식으로 저장했습니다.  
> 이 딕셔너리 구조를 PTL이 자신의 체크포인트로 오인해 `pytorch_lightning_version` 메타데이터를  
> 주입하고, 다른 버전에서 로드 시 불일치 오류가 발생했습니다.  
> 수정된 스크립트는 **순수 state dict** 만 저장하므로 이 문제가 없습니다.

---

## 변환 방법

### 1. 변환 실행

```bash
cd /path/to/NeMo/examples/speechlm2
python convert.py
```

출력 파일: `canary-qwen-2.5b.nemo` (현재 디렉터리)

### 2. 스크립트 내용 (`convert.py`)

```python
from nemo.collections.speechlm2.models import SALM
import torch, tarfile, tempfile, os
from omegaconf import OmegaConf

model = SALM.from_pretrained("nvidia/canary-qwen-2.5b")
model.eval()

save_path = "canary-qwen-2.5b.nemo"

with tempfile.TemporaryDirectory() as tmpdir:
    OmegaConf.save(model.cfg, os.path.join(tmpdir, "model_config.yaml"))
    torch.save(model.state_dict(), os.path.join(tmpdir, "model_weights.ckpt"))
    with tarfile.open(save_path, "w:") as tar:
        tar.add(tmpdir, arcname=".")

print(f"Saved: {save_path}")
```

---

## .nemo 파일 로드 방법

`SALM`은 NeMo의 `ModelPT`를 상속하지 않으므로 `restore_from()`을 사용할 수 없습니다.  
아래 패턴으로 직접 로드합니다.

```python
import tarfile, tempfile, os, torch
from omegaconf import OmegaConf
from nemo.collections.speechlm2.models import SALM

with tempfile.TemporaryDirectory() as tmpdir:
    with tarfile.open("canary-qwen-2.5b.nemo") as tar:
        tar.extractall(tmpdir)

    cfg = OmegaConf.to_container(
        OmegaConf.load(os.path.join(tmpdir, "model_config.yaml"))
    )
    cfg["pretrained_weights"] = False          # 아키텍처만 초기화
    model = SALM(cfg)
    model.load_state_dict(
        torch.load(os.path.join(tmpdir, "model_weights.ckpt"), map_location="cpu")
    )
```

---

## 관련 파일

| 파일 | 역할 |
|---|---|
| `convert.py` | HF Hub → `.nemo` 변환 스크립트 |
| `docs/02_module_checkpoints.md` | `.nemo` → 모듈별 분리 |
| `docs/03_training.md` | 학습 시 `.nemo` 체크포인트 사용 |
