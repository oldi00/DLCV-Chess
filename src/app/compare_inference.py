import subprocess
import torch
import os
import sys
import numpy as np
import json
from PIL import Image

# Configure Paths, so that utils can be imported.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.models import CustomChessCNN_v3

PATH_TO_EXE = "dist/InferenceModel.exe"

PATH_TO_TEST_IMAGE = r"C:/Users/olden/Desktop/programming/DLCV-Chess/data/image.png"
MODEL_PATH = r"G:/Meine Ablage/DLCV/models/finetuned (new)/epoch43_layer1_freeze_model/best_model.pth"

# Mapping for pieces
DEFAULT_ID_TO_PIECE = {
    0: "P",
    1: "R",
    2: "N",
    3: "B",
    4: "Q",
    5: "K",
    6: "p",
    7: "r",
    8: "n",
    9: "b",
    10: "q",
    11: "k",
    12: "1",
}


def softmax(x, temperature=4.0):
    x_scaled = x / temperature
    e_x = np.exp(x_scaled - np.max(x_scaled, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def probs_to_board_string(probs):
    probs_array = np.array(probs)
    predictions = np.argmax(probs_array, axis=1)
    predicted_pieces = [DEFAULT_ID_TO_PIECE[idx] for idx in predictions]
    return "".join(predicted_pieces)


def test_exe_model(image_path):
    print("=" * 50)
    print("TEST 1")
    print("=" * 50)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    try:
        result = subprocess.run(
            [PATH_TO_EXE],
            input=image_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stdout_str = result.stdout.decode().strip()
        print(stdout_str[:500] + ("..." if len(stdout_str) > 500 else ""))

        try:
            exe_probs = json.loads(stdout_str)
            board_str = probs_to_board_string(exe_probs)
            print("\nBOARD (EXE):")
            print(board_str)
        except json.JSONDecodeError:
            print("\nNo EXE could be found.")

        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr.decode())
    except FileNotFoundError:
        print(f"Error: The file {PATH_TO_EXE} could not be found.")


def test_raw_pytorch_model(image_path):
    print("\n" + "=" * 50)
    print("TEST 2: RUNNING RAW PYTORCH MODEL")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Liad Model
    model = CustomChessCNN_v3()
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Modelweights not found in: {MODEL_PATH}")
        return

    print("Load Modelweights")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Load and process Image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    img = img.resize((256, 256), Image.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW Format
    img_tensor = torch.tensor(img_np).unsqueeze(0).to(device)

    # Inference
    print("Inference:")
    with torch.no_grad():
        outputs = model(img_tensor)

    # Output has format (Batch, 64, Classes). Take first sample from batch.
    logits = outputs[0].cpu().numpy()

    # Softmax with temperature
    probs = softmax(logits, temperature=4.0)

    # 5. Output formaetted
    rows_str = []
    for row in probs:
        row_str = ", ".join([f"{val:.5f}" for val in row])
        rows_str.append(f"[{row_str}]")

    final_output = "[" + ", ".join(rows_str) + "]"

    print("MODEL STDOUT:")
    print(final_output[:500] + "...")

    # Convert probabilities to board
    board_str = probs_to_board_string(probs)
    print("\nBOARD (PYTORCH):")
    print(board_str)


if __name__ == "__main__":
    if not os.path.exists(PATH_TO_TEST_IMAGE):
        print(f"Error: Image not found in: {PATH_TO_TEST_IMAGE}")
    else:
        test_exe_model(PATH_TO_TEST_IMAGE)
        test_raw_pytorch_model(PATH_TO_TEST_IMAGE)
        print("\nDone. Tests finished")
