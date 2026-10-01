import os, torch
from Chess_bot import load_checkpoint, CHECKPOINT

model, _ = load_checkpoint(CHECKPOINT, torch.device("cpu"))
model.eval()
os.makedirs("docs", exist_ok=True)
torch.onnx.export(
    model, torch.zeros(1, 17, 8, 8), "docs/model.onnx",
    input_names=["x"], output_names=["policy", "value"],
    dynamic_axes={"x": {0: "b"}, "policy": {0: "b"}, "value": {0: "b"}},
    opset_version=17)
print(os.path.getsize("docs/model.onnx") / 1e6, "MB")