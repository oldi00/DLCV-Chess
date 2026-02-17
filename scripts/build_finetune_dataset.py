"""..."""

from board_detection import detect_board
from utils_chess_cv import pieces_to_fen
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import shutil
import os
import cv2
from tqdm import tqdm

CLOUD_DIR = Path(r"G:\.shortcut-targets-by-id\1kK41aQ1XMbjKKuinmDPfwUPClMvdcMvu\DLCV")
CHESSRED_DIR = Path(r"C:\Users\miles\Documents\ChessReD_Hough")
GAMES_DIR = Path(r"C:\Users\miles\Documents\Games")
OUT_DIR = Path(r"C:\Users\miles\Documents\Fine-Tuning Dataset")

METADATA_PATH = OUT_DIR / "metadata.json"
ANNOTATIONS_PATH = CLOUD_DIR / "annotations.json"


def load_metadata(path: Path) -> dict:
    """Load the metadata JSON file at the given path."""

    metadata = {}
    if path.exists():
        with open(path, 'r') as f:
            metadata = json.load(f)

    return metadata


def save_json(data: dict, path: Path) -> None:
    """Save the given dict as a JSON at the provided path."""

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_custom_data() -> list:
    """Fetch all paths of custom chess board images with their FEN and a unique name."""

    valid_formats = {'.jpg', '.jpeg', '.png'}
    files = [p for p in GAMES_DIR.rglob('*') if p.suffix.lower() in valid_formats]

    data = []
    for i, file_path in enumerate(files):
        name = f"{file_path.parts[-2]}_{i}"
        fen = "/".join(file_path.stem.split('_')[:-1])
        data.append((file_path, name, fen))

    return data


def get_chessred_data() -> list:
    """Fetch all ChessRed image paths with their FEN and a unique name."""

    annotations = load_metadata(ANNOTATIONS_PATH)

    image_to_pieces = defaultdict(list)
    for ann in annotations['annotations']['pieces']:
        image_to_pieces[ann['image_id']].append(ann)

    data = []
    for img_info in annotations['images']:

        full_path = CHESSRED_DIR / img_info['path']
        name = f"ChessReD_{img_info['id']}"

        pieces = image_to_pieces[img_info['id']]
        fen = pieces_to_fen(pieces)

        data.append((full_path, name, fen))

    return data


def process_image(args):
    """..."""

    img_path, name, fen = args

    img = cv2.imread(str(img_path))
    if img is None:
        return str(img_path), False, None, 0.0

    (board, score, _), _ = detect_board(img, debug=False)
    if board is None:
        return str(img_path), False, None, 0.0

    save_path = OUT_DIR / f"{name}.jpg"
    cv2.imwrite(str(save_path), board)

    return name, True, fen, score


def process_dataset(title: str, dataset: list, metadata: dict):
    """..."""

    to_process = []
    for item in dataset:
        img_path, name, _ = item
        if name not in metadata and str(img_path) not in metadata.get("fails", []):
            to_process.append(item)

    safe_workers = max(1, os.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=safe_workers) as executor:
        results = list(tqdm(
            executor.map(process_image, to_process),
            total=len(to_process),
            desc=title
        ))

    for name, success, fen, score in results:
        if success:
            metadata[name] = {"fen": fen, "score": float(score)}
        else:
            metadata["fails"].append(name)


def sync_chessred():
    """..."""

    metadata = load_metadata(METADATA_PATH)
    annotations = load_metadata(ANNOTATIONS_PATH)

    image_to_pieces = defaultdict(list)
    for ann in annotations['annotations']['pieces']:
        image_to_pieces[ann['image_id']].append(ann)

    images = list(CHESSRED_DIR.glob("*.png"))
    # count = 0
    for img_path in tqdm(images, unit="img", desc="ChessReD"):

        img_id = img_path.stem
        name = f"ChessReD_{img_id}"

        if name in metadata.keys():
            continue

        pieces = image_to_pieces[int(img_id)]
        fen = pieces_to_fen(pieces)

        metadata[name] = {"fen": fen}
        shutil.copy2(img_path, OUT_DIR / f"{name}.jpg")

        # count += 1
        # if count % 100:
        #     save_json(metadata, METADATA_PATH)

    save_json(metadata, METADATA_PATH)


def main():

    metadata = load_metadata(METADATA_PATH)

    if "fails" not in metadata.keys():
        metadata["fails"] = []

    custom_data = get_custom_data()
    # chessred_data = get_chessred_data()[:50]

    process_dataset("Custom", custom_data, metadata)
    # process_dataset("ChessReD", chessred_data, metadata)

    save_json(metadata, METADATA_PATH)


if __name__ == "__main__":
    # main()
    sync_chessred()
