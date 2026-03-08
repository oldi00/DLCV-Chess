import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import torch
import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

# Detect if cluster or not.
def get_path_from_config_file(config, key):
    on_cluster = "SLURM_JOB_ID" in os.environ
    if on_cluster:
        return config.get(f"cluster_{key}", config.get(key))
    return config.get(key)

id_to_piece = {
    0: "P", 1: "R", 2: "N", 3: "B", 4: "Q", 5: "K",
    6: "p", 7: "r", 8: "n", 9: "b", 10: "q", 11: "k",
    12: "1",
}

piece_to_label = {v: k for k, v in id_to_piece.items()}

def converting_annotations_to_fen(data_dir=None, annotations_data=None):
    image_id_to_path = {
        img["id"]: os.path.join(data_dir, img["path"])
        for img in annotations_data["images"]
    }
    image_to_pieces = defaultdict(list)
    for ann in annotations_data["annotations"]["pieces"]:
        image_to_pieces[ann["image_id"]].append(ann)
    return image_id_to_path, image_to_pieces

def fen_to_label_vector(fen, empty_char="0"):
    squares = []
    for row in fen.split("/"):
        for ch in row:
            if ch == empty_char:
                squares.append(12)
            elif ch.isdigit():
                squares.extend([12] * int(ch))
            else:
                squares.append(piece_to_label[ch])
    assert len(squares) == 64, f"Expected 64 squares but got {len(squares)} in FEN: {fen}"
    return torch.tensor(squares, dtype=torch.long)

def pieces_to_fen(piece_list):
    board = [["1"] * 8 for _ in range(8)]
    def pos_to_index(pos):
        return (8 - int(pos[1]), ord(pos[0]) - ord("a"))
    for piece in piece_list:
        row, col = pos_to_index(piece["chessboard_position"])
        board[row][col] = id_to_piece[piece["category_id"]]
    fen_rows = []
    for row in board:
        fen_row = ""
        for cell in row:
            if cell == "1":
                fen_row += "0"
            else:
                fen_row += cell
        fen_rows.append(fen_row)
    return "/".join(fen_rows)

def label_vector_to_fen(label_vector):
    assert len(label_vector) == 64, f"Expected 64 squares, got {len(label_vector)}"
    fen_rows = []
    for i in range(0, 64, 8):
        row = label_vector[i : i + 8]
        fen_row = ""
        empty_count = 0
        for val in row:
            if val == 12:
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += id_to_piece[int(val)]
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    return "/".join(fen_rows)

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
            ordered = order_points_robust(corners)
            return ordered, dilated
    return None, dilated

def order_points_robust(pts):
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    tl, bl = left_most[np.argsort(left_most[:, 1]), :]
    tr, br = right_most[np.argsort(right_most[:, 1]), :]
    return np.array([tl, tr, br, bl], dtype="float32")

def draw_corners(img, corners):
    img_copy = img.copy()
    labels = ["TL", "TR", "BR", "BL"]
    for i, pt in enumerate(corners):
        pt_int = tuple(np.int32(pt))
        cv2.circle(img_copy, pt_int, 14, (0, 0, 255), -1)
        cv2.putText(img_copy, labels[i], (pt_int[0] + 10, pt_int[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 3)
    for i in range(4):
        pt1 = tuple(np.int32(corners[i]))
        pt2 = tuple(np.int32(corners[(i + 1) % 4]))
        cv2.line(img_copy, pt1, pt2, (0, 255, 0), 2)
    return img_copy

def warp_board(img, corners, output_size=400):
    src_pts = order_points_robust(corners)
    dst_pts = np.array([[0, 0], [output_size - 1, 0], [output_size - 1, output_size - 1], [0, output_size - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (output_size, output_size))
    return warped

def slice_squares(warped, square_size=50):
    squares = []
    for row in range(8):
        for col in range(8):
            x1 = col * square_size
            y1 = row * square_size
            square = warped[y1 : y1 + square_size, x1 : x1 + square_size]
            squares.append(square)
    return squares

def preprocess_chessboard(image_path, output_size=400, display=True):
    img = cv2.imread(image_path)
    corners, debug_dilated = detect_board_corners(img)
    if corners is None:
        print("no corners")
        return
    corner_overlay = draw_corners(img, corners)
    warped = warp_board(img, corners, output_size)
    squares = slice_squares(warped, square_size=output_size // 8)

    if display:
        titles = ["Original", "Dilated Edges", "Corner Overlay", "Warped Top-Down"]
        images = [img, debug_dilated, corner_overlay, warped]
        plt.figure(figsize=(60, 20))
        for i in range(4):
            plt.subplot(1, 4, i + 1)
            img_disp = images[i]
            if len(img_disp.shape) == 2:
                plt.imshow(img_disp, cmap="gray")
            else:
                img_disp = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
                plt.imshow(img_disp)
            plt.title(titles[i], fontsize=70)
            plt.axis("off")
        plt.tight_layout()
        plt.show()
    return warped, squares

def expand_fen(fen, empty_char="0"):
    board = []
    for row in fen.split("/"):
        expanded_row = []
        for ch in row:
            if ch.isdigit() and ch != empty_char:
                expanded_row.extend([empty_char] * int(ch))
            else:
                expanded_row.append(ch)
        board.append(expanded_row)
    return board

def compress_fen_row(row, empty_char="0"):
    compressed = ""
    count = 0
    for ch in row:
        if ch == empty_char:
            count += 1
        else:
            if count:
                compressed += str(count)
                count = 0
            compressed += ch
    if count:
        compressed += str(count)
    return compressed

def board_to_fen(board, empty_char="0"):
    return "/".join([compress_fen_row(row, empty_char=empty_char) for row in board])

def rotate_fen_90(fen, empty_char="0"):
    board = expand_fen(fen, empty_char)
    rotated = list(zip(*board[::-1]))
    return board_to_fen([list(row) for row in rotated], empty_char)

def rotate_fen_180(fen, empty_char="0"):
    board = expand_fen(fen, empty_char)
    rotated = [row[::-1] for row in board[::-1]]
    return board_to_fen(rotated, empty_char)

def rotate_fen_270(fen, empty_char="0"):
    board = expand_fen(fen, empty_char)
    rotated = list(zip(*board))[::-1]
    return board_to_fen([list(row) for row in rotated], empty_char)

# =====================================================
# PICKLE CREATION FUNCTIONS
# =====================================================

def expand_fen_string(fen_string):
    dense_rows = []
    for row in fen_string.split('/'):
        dense_row = ""
        for char in row:
            if char.isdigit():
                dense_row += "0" * int(char)
            else:
                dense_row += char
        dense_rows.append(dense_row)
    return "/".join(dense_rows)


def create_pickle(base_dir, image_dir, json_file_name, output_prefix="dataset", split_ratios=[0.7, 0.2, 0.1]):
    base_path = Path(base_dir)
    image_dir = Path(image_dir)
    json_path = base_path / json_file_name

    print(f"Loading metadata from: {json_path}")
    with open(json_path, "r") as f:
        label_data = json.load(f)

    print(f"Scanning images in: {image_dir}")
    files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    print(f"   Found {len(files)} files.")

    data_pairs = []
    missing_count = 0

    print("Matching images to metadata keys and expanding FEN strings")
    for file_name in tqdm(files):
        key = Path(file_name).stem 
        
        if key in label_data:
            file_path = str(image_dir / file_name)
            entry = label_data[key]
            
            # Extract FEN string
            if isinstance(entry, dict) and "fen" in entry:
                fen = entry["fen"]
            elif isinstance(entry, str):
                fen = entry
            else:
                print(f"Unexpected format for key {key}: {entry}")
                continue
            
            expanded_fen = expand_fen_string(fen)            
            data_pairs.append((file_path, expanded_fen))
        else:
            missing_count += 1
            if missing_count < 5:
                print(f"Key not found for file: {file_name} (Looked for key: '{key}')")

    if len(data_pairs) == 0:
        print("No data pairs created. Aborting.")
        return
    
    random.seed(42)
    random.shuffle(data_pairs)
    
    total = len(data_pairs)
    train_end = int(total * split_ratios[0])
    val_end = int(total * (split_ratios[0] + split_ratios[1]))

    splits = {
        "train": data_pairs[:train_end],
        "val": data_pairs[train_end:val_end],
        "test": data_pairs[val_end:],
    }

    for split_name, pairs in splits.items():
        if not pairs: continue # Skip empty splits
        X = [pair[0] for pair in pairs]
        y = [pair[1] for pair in pairs]
        
        pkl_filename = f"{output_prefix}_{split_name}_data.pkl"
        pkl_save_path = base_path / pkl_filename

        with open(pkl_save_path, "wb") as f:
            pickle.dump((X, y), f)

        print(f"Saved {len(X)} samples to: {pkl_save_path}")


def create_pickle_no_test(base_dir, image_dir, json_file_name, output_prefix="dataset", split_ratios=[0.8, 0.2]):
    base_path = Path(base_dir)
    image_dir = Path(image_dir)
    json_path = base_path / json_file_name

    print(f"Loading metadata from: {json_path}")
    with open(json_path, "r") as f:
        label_data = json.load(f)

    print(f"Scanning images in: {image_dir}")
    files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    print(f"   Found {len(files)} files.")

    data_pairs = []
    missing_count = 0

    print("Matching images to metadata keys and expanding FEN strings")
    for file_name in tqdm(files):
        key = Path(file_name).stem 
        
        if key in label_data:
            file_path = str(image_dir / file_name)
            entry = label_data[key]
            if isinstance(entry, dict) and "fen" in entry:
                fen = entry["fen"]
            elif isinstance(entry, str):
                fen = entry
            else:
                print(f"Unexpected format for key {key}: {entry}")
                continue
            data_pairs.append((file_path, fen))
        else:
            missing_count += 1
            if missing_count < 5:
                print(f"Key not found for file: {file_name} (Looked for key: '{key}')")

    if len(data_pairs) == 0:
        print("No data pairs created. Aborting.")
        return
    
    total = len(data_pairs)
    train_end = int(total * split_ratios[0])

    splits = {
        "train": data_pairs[:train_end],
        "val": data_pairs[train_end:],
    }

    # OUTPUT
    for split_name, pairs in splits.items():
        if not pairs: continue
        X = [pair[0] for pair in pairs]
        y = [pair[1] for pair in pairs]

        pkl_filename = f"{output_prefix}_{split_name}_data.pkl"
        pkl_save_path = base_path / pkl_filename

        with open(pkl_save_path, "wb") as f:
            pickle.dump((X, y), f)

        print(f"Saved {len(X)} samples to: {pkl_save_path}")

# ==============================================================
# MAIN EXECUTION
# ==============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Pickle datasets from images and metadata")
    
    # Required Paths
    parser.add_argument("--base_dir", type=str, required=True, help="Directory containing metadata.json and where to save .pkl")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the images")
    
    # Optional parameters
    parser.add_argument("--json_file", type=str, default="metadata.json", help="Name of the metadata file")
    parser.add_argument("--mode", type=str, choices=["with_test", "no_test"], default="no_test", help="Whether to create a test split")
    parser.add_argument("--prefix", type=str, default="dataset", help="Prefix for the generated pkl files")
    
    args = parser.parse_args()

    # Verify paths exist before running
    if not os.path.exists(args.base_dir):
        print(f"Base directory not found: {args.base_dir}")
        exit(1)
    if not os.path.exists(args.image_dir):
        print(f"Image directory not found: {args.image_dir}")
        exit(1)


    # INFO
    print("\n========================================")
    print("Starting Pickle Creation")
    print(f"Base Dir:   {args.base_dir}")
    print(f"Image Dir:  {args.image_dir}")
    print(f"Split Mode: {args.mode}")
    print(f"Prefix:     {args.prefix}_[train/val]_data.pkl")
    print("========================================\n")

    if args.mode == "with_test":
        create_pickle(args.base_dir, args.image_dir, args.json_file, output_prefix=args.prefix)
    else:
        create_pickle_no_test(args.base_dir, args.image_dir, args.json_file, output_prefix=args.prefix)