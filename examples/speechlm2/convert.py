import os
import tarfile
import tempfile

import torch
from omegaconf import OmegaConf

from nemo.collections.speechlm2.models import SALM

model = SALM.from_pretrained("nvidia/canary-qwen-2.5b")
model.eval()

save_path = "canary-qwen-2.5b.nemo"

with tempfile.TemporaryDirectory() as tmpdir:
    OmegaConf.save(model.cfg, os.path.join(tmpdir, "model_config.yaml"))
    torch.save(model.state_dict(), os.path.join(tmpdir, "model_weights.ckpt"))
    with tarfile.open(save_path, "w:") as tar:
        tar.add(tmpdir, arcname=".")

print(f"Saved: {save_path}")
