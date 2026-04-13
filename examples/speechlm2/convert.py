from nemo.collections.speechlm2.models import SALM
import torch

model = SALM.from_pretrained("nvidia/canary-qwen-2.5b")
model.eval()

torch.save({
    "state_dict": model.state_dict(),
    "config":     model.cfg,
}, "canary-qwen-2.5b.ckpt")

print("Saved: canary-qwen-2.5b.ckpt")
