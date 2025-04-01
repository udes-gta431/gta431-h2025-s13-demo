import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import onnxruntime as rt
import numpy as np


def save_model_to_onnx(model, X_sample, filename="model.onnx"):
    """
    Converts and saves a scikit-learn model to ONNX format.

    Parameters:
    - model: trained scikit-learn model
    - X_sample: a sample of the input features (e.g., X_train.head(1))
    - filename: output ONNX file path
    """
    initial_type = [('input', FloatTensorType([None, X_sample.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    with open(filename, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"✅ Model saved to {filename}")


def load_onnx_and_predict(filename, X_input):
    """
    Loads an ONNX model and runs prediction on input data.

    Parameters:
    - filename: path to ONNX file
    - X_input: NumPy array or DataFrame of input data

    Returns:
    - Numpy array of predictions
    """
    sess = rt.InferenceSession(filename)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    # Ensure input is float32
    if isinstance(X_input, pd.DataFrame):
        X_input = X_input.values.astype(np.float32)
    else:
        X_input = X_input.astype(np.float32)

    preds = sess.run([label_name], {input_name: X_input})[0]
    return preds
