# Checkpoints

Model weights are hosted on HuggingFace, not in this repo.

Download: https://huggingface.co/AriaMF/anomaly-gpt

```python
from huggingface_hub import hf_hub_download
import torch

ckpt_path = hf_hub_download(repo_id="AriaMF/anomaly-gpt", filename="final_model.pt")
ckpt = torch.load(ckpt_path, map_location="cpu")
```
