import sys
import numpy as np
import onnxruntime as rt
import onnx
from PIL import Image
import io


def softmax(x):
    """Berechnet Softmax stabil über die letzte Achse."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def run():
    # Get the image bytes input.
    image_bytes = sys.stdin.buffer.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Necessary resize and transformations.
    img = img.resize((256, 256), Image.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)
    img_np = np.expand_dims(img_np, axis=0)

    # Create the inference session.
    sess = rt.InferenceSession("onnx_models/finetuned_chess_model.onnx")
    input_name = sess.get_inputs()[0].name

    # Run the inference.
    outputs = sess.run(None, {input_name: img_np})

    # Print results.
    logits = outputs[0][0]
    probs = softmax(logits)

    print(probs)


if __name__ == "__main__":
    run()
