import sys
import os
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
import io


def softmax(x):
    """Berechnet Softmax stabil über die letzte Achse."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def run_strict_inference():
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        src_path = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.dirname(src_path)

    model_path = os.path.join(base_path, "chess_model.onnx")

    try:
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        session = ort.InferenceSession(model_path, sess_options)

        img_bytes = sys.stdin.buffer.read()
        if not img_bytes:
            return

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img).astype(np.float32)

        img_np = img_np.transpose(2, 0, 1)
        img_np = np.expand_dims(img_np, axis=0)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img_np})

        logits = outputs[0][0]
        probs = softmax(logits)

        # JSON ausgeben
        print(json.dumps(probs.tolist()))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    run_strict_inference()
