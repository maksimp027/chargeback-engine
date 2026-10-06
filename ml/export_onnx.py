import json
import os

import numpy as np
import onnxruntime as ort
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import xgboost as xgb

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)

model_path = os.path.join(project_dir, "model_artifacts", "xgboost_baseline.json")
schema_path = os.path.join(project_dir, "model_artifacts", "features_schema.json")
onnx_path = os.path.join(project_dir, "model_artifacts", "chargeback_model.onnx")

bst = xgb.Booster()
bst.load_model(model_path)

with open(schema_path, "r") as f:
    schema = json.load(f)

num_features = len(schema["features"])

# onnxmltools requires feature_names=None to map inputs to positional f0..fN indices.
bst.feature_names = None

initial_types = [("float_input", FloatTensorType([None, num_features]))]

onnx_model = onnxmltools.convert_xgboost(
    bst,
    initial_types=initial_types,
    target_opset=15,
)

onnxmltools.utils.save_model(onnx_model, onnx_path)
print(f"Model saved to {onnx_path}")

dummy_input = np.random.rand(5, num_features).astype(np.float32)

dmatrix = xgb.DMatrix(dummy_input)
xgb_preds = bst.predict(dmatrix)

session = ort.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name
label_name = session.get_outputs()[1].name

onnx_preds = session.run([label_name], {input_name: dummy_input})[0]
if isinstance(onnx_preds, list):
    onnx_prob = np.array([row[1] for row in onnx_preds])
else:
    onnx_prob = onnx_preds[:, 1] if onnx_preds.ndim > 1 else onnx_preds

max_diff = np.max(np.abs(xgb_preds - onnx_prob))
print(f"Max difference: {max_diff:.8f}")
assert max_diff < 1e-4, "Error: Discrepancy between Python and ONNX!"
print("Validation successful. Model is ready for Go service.")