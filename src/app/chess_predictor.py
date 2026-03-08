import sys
import numpy as np
import onnxruntime as rt
import io
import os
from PIL import Image


def softmax(x, temperature=4.0):
    x_scaled = x / temperature
    e_x = np.exp(x_scaled - np.max(x_scaled, axis=1, keepdims=True))

    return e_x / np.sum(e_x, axis=1, keepdims=True)


# Configure the exe file path to the model
def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def run():

    # Get the image bytes input.
    image_bytes = sys.stdin.buffer.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Necessary resize and transformations.
    img = img.resize((256, 256), Image.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)
    img_np = np.expand_dims(img_np, axis=0)

    model_path = resource_path("onnx_models/finetuned_chess_model_epoch43.onnx")
    sess = rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name

    # Run the inference.
    outputs = sess.run(None, {input_name: img_np})

    # Process results.
    logits = outputs[0][0]
    probs = softmax(logits)

    rows_str = []
    for row in probs:
        row_str = ", ".join([f"{val:.5f}" for val in row])
        rows_str.append(f"[{row_str}]")

    final_output = "[" + ", ".join(rows_str) + "]"
    print(final_output)


if __name__ == "__main__":
    run()
