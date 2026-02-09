from board_detection import detect_board
import utils
import utils_chess_cv
import cv2
import json
import uuid
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


# --- CONFIGURATION---

SOURCE_CUSTOM = Path(r"G:\.shortcut-targets-by-id\1kK41aQ1XMbjKKuinmDPfwUPClMvdcMvu\DLCV\Games")
SOURCE_CHESSRED = Path(
    r"G:\.shortcut-targets-by-id\1kK41aQ1XMbjKKuinmDPfwUPClMvdcMvu\DLCV\ChessReD")

ANNOTATIONS_PATH = Path(
    r"G:\.shortcut-targets-by-id\1kK41aQ1XMbjKKuinmDPfwUPClMvdcMvu\DLCV\annotations.json")

DEST_DIR = Path(r"G:\.shortcut-targets-by-id\1kK41aQ1XMbjKKuinmDPfwUPClMvdcMvu\DLCV\Fine-Tuning")
DEST_HIGH = DEST_DIR / "HIGH"
DEST_LOW = DEST_DIR / "LOW"

METADATA_FILE = DEST_DIR / "metadata.json"

DEBUG_LIMIT = 10  # Set to None for a full run.


def process_image(img_path, fen, metadata, source_name):

    img_rgb = utils.load_image_RGB(img_path)
    board, debug = detect_board(img_rgb, debug=True)

    if board is None:
        return "FAIL"

    image_guid = str(uuid.uuid4())
    save_name = f"{image_guid}.png"

    save_folder = DEST_HIGH if debug["confidence"] == "HIGH" else DEST_LOW
    save_path = save_folder / save_name

    cropped_bgr = cv2.cvtColor(board, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), cropped_bgr)

    metadata[image_guid] = {
        "fen": fen,
        "source": source_name,
    }

    return debug["confidence"]


def get_custom_data(source_path):

    valid_exts = {'.jpg', '.jpeg', '.png'}
    files = [p for p in source_path.rglob('*') if p.suffix.lower() in valid_exts]

    data = []
    for file in files:
        fen = "/".join(file.stem.split('_')[:-1])
        data.append((file, fen))

    return data


def get_chessred_data(source_root, json_path):

    with open(json_path, 'r') as f:
        annotations = json.load(f)

    image_to_pieces = defaultdict(list)
    for ann in annotations['annotations']['pieces']:
        image_to_pieces[ann['image_id']].append(ann)

    data = []
    for img_info in annotations['images']:

        full_path = source_root / img_info['path']

        pieces = image_to_pieces[img_info['id']]
        raw_fen = utils_chess_cv.pieces_to_fen(pieces)

        data.append((full_path, raw_fen))

    return data


def main():

    DEST_HIGH.mkdir(parents=True, exist_ok=True)
    DEST_LOW.mkdir(parents=True, exist_ok=True)

    metadata = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)

    custom_list = get_custom_data(SOURCE_CUSTOM)
    chessred_list = get_chessred_data(SOURCE_CHESSRED, ANNOTATIONS_PATH)

    tasks = [
        (custom_list, "Custom"),
        (chessred_list, "ChessReD")
    ]

    stats = {"HIGH": 0, "LOW": 0, "FAIL": 0}

    for dataset, name in tasks:

        for i, (img_path, fen) in enumerate(tqdm(dataset, desc=f"Processing {name}", unit="img")):

            if DEBUG_LIMIT and i >= DEBUG_LIMIT:
                break

            res = process_image(img_path, fen, metadata, name)
            stats[res] += 1

    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)

    print("PROCESSING COMPLETE!")
    print("\n" + "="*30)
    print(f"High Quality: {stats['HIGH']}")
    print(f"Low Quality:  {stats['LOW']}")
    print(f"Failures:     {stats['FAIL']}")
    print("="*30)


if __name__ == "__main__":
    main()
