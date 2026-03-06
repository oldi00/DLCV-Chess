"""
Transform raw synthetic data from various sources into
400x400x3 top-down views for model training.
"""

import utils
import utils_chess_cv
import json
import csv
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

OUT_DIR = Path(r"C:\Users\miles\Documents\Pre-Training Dataset")
METADATA_DIR = OUT_DIR / "metadata.json"


def save_metadata(metadata: dict) -> None:
    """Save metadata to disk and prevent file corruption if the script crashes during write."""

    temp_path = METADATA_DIR.with_suffix(".tmp")
    with open(temp_path, "w") as file:
        json.dump(metadata, file, indent=4)
    if METADATA_DIR.exists():
        METADATA_DIR.unlink()
    temp_path.rename(METADATA_DIR)


def generate_fen_from_config(config: dict) -> str:
    """Generate FEN string from Kaggle SynthChess configuration dictionary."""

    # todo: merge with similar function in utils.py

    piece_map = {
        "pawn": "p", "rook": "r", "knight": "n",
        "bishop": "b", "queen": "q", "king": "k"
    }

    fen_rows = []
    for rank in range(8, 0, -1):
        empty_count = 0
        row_str = ""
        for file_idx in range(8):
            file_char = chr(ord('A') + file_idx)
            coord = f"{file_char}{rank}"

            if coord in config:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0

                piece_type, piece_color = config[coord].split('_')
                char = piece_map.get(piece_type, '?')
                if piece_color == 'w':
                    char = char.upper()
                row_str += char
            else:
                empty_count += 1

        if empty_count > 0:
            row_str += str(empty_count)
        fen_rows.append(row_str)

    return "/".join(fen_rows)


def process_and_save_entry(
        img_path: Path, corners: np.ndarray, out_name: str, fen: str, metadata: dict) -> None:
    """Core logic: check existence, warp image, save file, and update metadata."""

    out_path = OUT_DIR / out_name
    if out_path.exists() and out_name in metadata:
        return

    img = utils.load_image_RGB(img_path)
    img = utils_chess_cv.warp_board(img, corners)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(out_path), img)
    metadata[out_name] = {"fen": fen}

    # if len(metadata) % 1000 == 0:
    #     save_metadata(metadata)


def process_unity(dir: Path, metadata: dict) -> None:
    """Process Unity dataset images using corner data from metadata.json."""

    with open(dir / "metadata.json", "r") as file:
        image_data = json.load(file)

    image_paths = list(dir.glob("*.jpg"))
    image_paths = sorted(image_paths)

    for i, img_path in enumerate(tqdm(image_paths, desc="Unity", unit="img")):

        img_name = img_path.name

        # Unity specific: Points need robust ordering.
        corners = np.array(image_data[img_name]["corners"], dtype="float32")
        corners = utils_chess_cv.order_points_robust(corners)

        fen = image_data[img_name]["fen"]
        out_name = f"unity_{i}.jpg"

        process_and_save_entry(img_path, corners, out_name, fen, metadata)


def process_kaggle_synth(dir: Path, metadata: dict) -> None:
    """Process Kaggle SynthChess dataset calculating FEN from config."""

    image_paths = list(dir.glob("*.jpg"))
    for img_path in tqdm(image_paths, desc="Kaggle SynthChess", unit="img"):

        # Helper to read dimensions for normalization.
        img_temp = cv2.imread(str(img_path))
        h, w = img_temp.shape[:2]

        with open(dir / f"{img_path.stem}.json", "r") as file:
            data = json.load(file)

        # Kaggle Synth specific: Corners need denormalization and flip.
        corners = np.array(data["corners"], dtype="float32")[:, ::-1]
        corners[:, 0] *= w
        corners[:, 1] *= h
        corners = utils_chess_cv.order_points_robust(corners)

        fen = generate_fen_from_config(data["config"])
        out_name = f"kaggle_synth_{img_path.stem}.jpg"

        process_and_save_entry(img_path, corners, out_name, fen, metadata)


def process_kaggle_render(dir: Path, metadata: dict) -> None:
    """Process Kaggle Render dataset using FENs from CSV and corners from JSON."""

    rgb_dir = dir / "rgb"
    annotations_dir = dir / "annotations"

    with open(dir / "FENs.csv", "r") as file:
        reader = csv.reader(file)
        fens = [row[0].split(" ")[0] for row in reader]

    image_paths = list(rgb_dir.glob("*.jpeg"))
    for img_path in tqdm(image_paths, desc="Kaggle ChessRender360", unit="img"):

        img_num = img_path.stem.split("_")[-1]

        with open(annotations_dir / f"annotation_{img_num}.json", "r") as file:
            annotation = json.load(file)

        corners = np.array(list(annotation["board_corners"].values()), dtype="float32")
        corners = utils_chess_cv.order_points_robust(corners)
        fen = fens[int(img_num)]
        out_name = f"kaggle_render_{img_num}.jpg"

        process_and_save_entry(img_path, corners, out_name, fen, metadata)


def process_blender(dir: Path, metadata: dict) -> None:
    """Process Blender dataset where JSON shares the filename."""

    image_paths = list(dir.rglob("*.png"))
    for img_path in tqdm(image_paths, desc="Blender", unit="img"):

        img_num = img_path.stem

        with open(img_path.parent / f"{img_num}.json", "r") as file:
            img_json = json.load(file)

        corners = np.array(img_json["corners"], dtype="float32")
        corners = utils_chess_cv.order_points_robust(corners)
        fen = img_json["fen"]
        out_name = f"blender_{img_path.parent.name}_{img_num}.jpg"

        process_and_save_entry(img_path, corners, out_name, fen, metadata)


def main():

    metadata = {}
    if METADATA_DIR.exists():
        with open(METADATA_DIR, "r") as file:
            metadata = json.load(file)

    unity_dir = Path(r"C:\Users\miles\code\uni\dlcv\ChessDataGen\Snapshots")
    process_unity(unity_dir, metadata)

    kaggle_synth_dir = Path(r"C:\Users\miles\code\uni\dlcv\datasets\Kaggle_SynthChess")
    process_kaggle_synth(kaggle_synth_dir, metadata)

    kaggle_render_dir = Path(r"C:\Users\miles\code\uni\dlcv\datasets\Kaggle_ChessRender360")
    process_kaggle_render(kaggle_render_dir, metadata)

    blender_dir = Path(r"C:\Users\miles\code\uni\dlcv\datasets\Blender")
    process_blender(blender_dir, metadata)

    save_metadata(metadata)


if __name__ == "__main__":
    main()
