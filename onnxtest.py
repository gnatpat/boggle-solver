import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("boggle_cnn.onnx")
x = np.random.randn(1, 1, 32, 32).astype(np.float32)
outputs = session.run(None, {"input": x})
print(outputs[0].shape)  # should be (1, 26)