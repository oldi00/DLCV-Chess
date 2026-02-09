"""..."""

from board_detection import detect_board
import utils
import utils_chess_cv
import gc
import cv2
import json
import uuid
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


# --- CONFIGURATION ---

BASE_DIR = Path(r"G:\.shortcut-targets-by-id\1kK41aQ1XMbjKKuinmDPfwUPClMvdcMvu\DLCV")

GAMES_DIR = BASE_DIR / "Games"
CHESSRED_DIR = BASE_DIR / "ChessReD"
OUTPUT_DIR = BASE_DIR / "Fine-Tuning"

ANNOTATIONS_FILE = BASE_DIR / "annotations.json"
METADATA_FILE = OUTPUT_DIR / "metadata.json"
FAILS_FILE = OUTPUT_DIR / "fails.json"


def load_json(path):

    metadata = {}
    if path.exists():
        with open(path, 'r') as f:
            metadata = json.load(f)

    return metadata


def save_json(data, path):

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_custom_data():

    valid_formats = {'.jpg', '.jpeg', '.png'}
    files = [p for p in GAMES_DIR.rglob('*') if p.suffix.lower() in valid_formats]

    data = []
    for file in files:

        fen = "/".join(file.stem.split('_')[:-1])
        data.append((file, fen))

    return data


def get_chessred_data():

    with open(ANNOTATIONS_FILE, 'r') as f:
        annotations = json.load(f)

    image_to_pieces = defaultdict(list)
    for ann in annotations['annotations']['pieces']:
        image_to_pieces[ann['image_id']].append(ann)

    data = []
    for img_info in annotations['images']:

        full_path = CHESSRED_DIR / img_info['path']

        pieces = image_to_pieces[img_info['id']]
        raw_fen = utils_chess_cv.pieces_to_fen(pieces)

        data.append((full_path, raw_fen))

    return data


def process_image(img_path, fen, metadata, fails, source_name):

    img_rgb = utils.load_image_RGB(img_path)
    board, debug = detect_board(img_rgb, debug=True)

    if board is None:
        fails[str(img_path)] = {"reason": debug["error"]}
        return

    image_guid = str(uuid.uuid4())

    save_name = f"{image_guid}.png"
    save_path = OUTPUT_DIR / save_name

    cropped_bgr = cv2.cvtColor(board, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), cropped_bgr)

    metadata[image_guid] = {
        "fen": fen,
        "source": source_name,
        "original_path": str(img_path),
        "confidence": debug["confidence"]
    }


def main():

    metadata = load_json(METADATA_FILE)
    fails = load_json(FAILS_FILE)

    processed_paths = set()
    for entry in metadata.values():
        processed_paths.add(entry["original_path"])
    for path in fails.keys():
        processed_paths.add(path)

    tasks = [
        ("Custom", get_custom_data()),
        ("ChessReD", get_chessred_data()),
    ]

    session_count = 0

    for src_name, dataset in tasks:

        for i, (img_path, fen) in enumerate(tqdm(dataset, desc=src_name, unit="img")):

            if str(img_path) in processed_paths:
                continue

            process_image(img_path, fen, metadata, fails, src_name)

            session_count += 1
            if session_count % 100 == 0:
                save_json(metadata, METADATA_FILE)
                save_json(fails, FAILS_FILE)
                gc.collect()

    save_json(metadata, METADATA_FILE)
    save_json(fails, FAILS_FILE)


if __name__ == "__main__":
    main()
