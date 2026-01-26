import cv2
import numpy as np
import argparse
import os
import sys
import onnxruntime as ort
import json
import io
from PIL import Image

# ============================
# 1. Helper & Preprocessing
# ============================
id_to_piece = {
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


def softmax(x):
    """
    Berechnet die Softmax-Funktion für jede Zeile einer Matrix x.
    Formel: exp(x) / sum(exp(x))
    Stabilisiert durch Subtraktion des Maximums (verhindert Overflow).
    """
    # x hat Form (64, 13)
    # Maximum pro Zeile abziehen für numerische Stabilität
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    # Teilen durch die Summe pro Zeile
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def order_points_robust(pts):
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    tl, bl = left_most[np.argsort(left_most[:, 1]), :]
    tr, br = right_most[np.argsort(right_most[:, 1]), :]
    return np.array([tl, tr, br, bl], dtype="float32")


def detect_board_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    image_area = img.shape[0] * img.shape[1]

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.1 * image_area:
            corners = np.float32([pt[0] for pt in approx])
            return order_points_robust(corners)
    return None


def warp_board(img, corners, output_size=400):
    src_pts = corners
    dst_pts = np.array(
        [
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1],
        ],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (output_size, output_size))


def preprocess_image_numpy(pil_image):
    pil_image = pil_image.resize((256, 256), Image.BILINEAR)
    img_np = np.array(pil_image).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)
    img_np = np.expand_dims(img_np, axis=0)
    return img_np


def logits_to_fen(pred_indices):
    board_rows = []
    for i in range(0, 64, 8):
        row_indices = pred_indices[i : i + 8]
        row_chars = [id_to_piece[idx] for idx in row_indices]
        board_rows.append(row_chars)

    fen_rows = []
    for row in board_rows:
        fen_row = ""
        empty_count = 0
        for char in row:
            if char == "1":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += char
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    return "/".join(fen_rows), board_rows


# ============================
# 2. Inference Logic
# ============================


def run_inference(input_data, model_path, is_preprocessed, raw_mode=False):
    if hasattr(sys, "_MEIPASS"):
        model_path = os.path.join(sys._MEIPASS, model_path)

    try:
        ort_session = ort.InferenceSession(model_path)
    except Exception as e:
        if not raw_mode:
            print(f"Error loading model: {e}")
        return

    # --- INPUT PROCESSING ---
    pil_image = None
    img_bgr = None

    try:
        # Case A: Bytes
        if isinstance(input_data, bytes):
            if is_preprocessed:
                pil_image = Image.open(io.BytesIO(input_data)).convert("RGB")
            else:
                nparr = np.frombuffer(input_data, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    return

        # Case B: File Path
        elif isinstance(input_data, str):
            input_data = input_data.strip().strip('"').strip("'")
            if not os.path.exists(input_data):
                return

            if is_preprocessed:
                pil_image = Image.open(input_data).convert("RGB")
            else:
                img_bgr = cv2.imread(input_data)
                if img_bgr is None:
                    return
        else:
            return

        # --- CORNER DETECTION ---
        if not is_preprocessed and pil_image is None:
            if not raw_mode:
                print("Detecting corners...")
            corners = detect_board_corners(img_bgr)
            if corners is None:
                if not raw_mode:
                    print("Error: Could not detect chessboard corners.")
                return

            warped_bgr = warp_board(img_bgr, corners)
            pil_image = Image.fromarray(cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB))

        # --- INFERENCE ---
        input_tensor = preprocess_image_numpy(pil_image)
        input_name = ort_session.get_inputs()[0].name

        # outputs[0] Shape: (1, 64, 13) - Logits
        outputs = ort_session.run(None, {input_name: input_tensor})

        # --- RAW MODE: Probabilities ---
        if raw_mode:
            logits = outputs[0][0]  # (64, 13)

            # WICHTIG: Hier Logits zu Wahrscheinlichkeiten umwandeln
            probs = softmax(logits)

            # Als Liste für JSON exportieren
            matrix = probs.tolist()
            print(json.dumps(matrix))
            return
        # -------------------------------

        pred_indices = np.argmax(outputs[0], axis=2)
        pred_indices_flat = pred_indices[0]
        fen, board_grid = logits_to_fen(pred_indices_flat)

        print("\n" + "=" * 30)
        print("      PREDICTION RESULT      ")
        print("=" * 30)
        print("\nVisual Board:")
        for row in board_grid:
            print(" ".join(["." if x == "1" else x for x in row]))
        print(f"\nPredicted FEN string:\n{fen}")
        print("\n" + "=" * 30)

    except Exception as e:
        if not raw_mode:
            print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChessPredictor")
    parser.add_argument("image_path", nargs="?", type=str, help="Path to input image")
    parser.add_argument(
        "--stdin", action="store_true", help="Read image bytes from standard input"
    )
    parser.add_argument(
        "--model", type=str, default="chess_model.onnx", help="Path to .onnx file"
    )
    parser.add_argument(
        "--preprocessed", action="store_true", help="Use if image is already cropped"
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Output only the 64x13 PROBABILITY matrix as JSON",
    )

    args = parser.parse_args()

    if args.stdin:
        image_bytes = sys.stdin.buffer.read()
        if image_bytes:
            run_inference(image_bytes, args.model, args.preprocessed, args.raw)
    elif args.image_path:
        run_inference(args.image_path, args.model, args.preprocessed, args.raw)
    else:
        print("################################################")
        print("#         ChessPredictor (Probabilities)       #")
        print("################################################")
        while True:
            print("\nEnter path to image file (or type 'q' to quit):")
            user_input = input(">> ")
            if user_input.lower() in ["q", "quit", "exit"]:
                break
            if not user_input.strip():
                continue
            run_inference(
                user_input, "chess_model.onnx", is_preprocessed=False, raw_mode=False
            )
