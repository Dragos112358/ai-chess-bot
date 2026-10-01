import onnx
m = onnx.load("docs/model.onnx", load_external_data=True)
onnx.save_model(m, "docs/model.onnx", save_as_external_data=False)
import os
print(os.path.getsize("docs/model.onnx") / 1e6, "MB")