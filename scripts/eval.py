"""..."""

from board_detection import detect_board
from utils_chess_cv import order_points_robust, detect_board_corners
from utils import load_image_RGB
from pathlib import Path
import json
import cv2
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm


UNITY_DIR = Path(r"C:\Users\miles\code\uni\dlcv\ChessDataGen\Snapshots")
UNITY_METADATA_PATH = UNITY_DIR / "metadata.json"

CLOUD_DIR = Path(r"G:\.shortcut-targets-by-id\1kK41aQ1XMbjKKuinmDPfwUPClMvdcMvu\DLCV")
CHESSRED_DIR = CLOUD_DIR / "ChessReD2K/images"
CHESSRED_METADATA_PATH = CLOUD_DIR / "annotations.json"

CUSTOM_DIR = Path(r"C:\Users\miles\code\uni\dlcv\Board Detection Evaluation")
CUSTOM_METADATA_PATH = CUSTOM_DIR / "metadata.json"


def load_metadata(path: Path):
    """..."""

    metadata = {}
    if path.exists():
        with open(path, "r") as file:
            metadata = json.load(file)

    return metadata


def open_image_editor(img_path: Path):
    """..."""

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    corners = []

    def get_coords(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            corners.append((x, y))
            cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
            cv2.imshow("Window", img)

    target_height = 800
    aspect_ratio = w / h
    target_width = int(target_height * aspect_ratio)

    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)  # 1. Allow resizing
    cv2.resizeWindow("Window", target_width, target_height)
    cv2.moveWindow("Window", 50, 50)

    cv2.imshow("Window", img)
    cv2.setMouseCallback("Window", get_coords)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return corners


def annotate_corners():

    metadata = load_metadata()

    for img_path in CUSTOM_DIR.iterdir():

        img_key = img_path.stem
        if img_key in metadata.keys() or img_key == "metadata":
            continue

        corners = open_image_editor(img_path)
        if len(corners) != 4:
            print(f"Skipping {img_key}: Amounts of corners deviates from four.")
            continue

        metadata[img_key] = corners

    with open(CUSTOM_METADATA_PATH, "w") as file:
        json.dump(metadata, file, indent=4)


def compute_iou(pts_a, pts_b):
    """..."""

    poly_a = Polygon(pts_a)
    poly_b = Polygon(pts_b)

    # Handle self-intersecting polygons if necessary.
    if not poly_a.is_valid:
        poly_a = poly_a.buffer(0)
    if not poly_b.is_valid:
        poly_b = poly_b.buffer(0)

    inter_area = poly_a.intersection(poly_b).area
    union_area = poly_a.area + poly_b.area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def create_dataset(title, images, metadata):
    """..."""

    dataset = []

    if title == "Unity":
        for img_path in images:
            corners = np.array(metadata[img_path.name]["corners"])
            corners = order_points_robust(corners)
            dataset.append((img_path, corners))

    elif title == "ChessReD":
        corners_data = metadata["annotations"]["corners"]
        img_path_to_id = {img["path"]: img["id"] for img in metadata["images"]}
        img_id_to_corners = {img["image_id"]: img["corners"] for img in corners_data}
        for img_path in images:
            rel_path = Path(*img_path.parts[-3:]).as_posix()
            corners = img_id_to_corners[img_path_to_id[rel_path]]
            corners = order_points_robust(np.array(list(corners.values())))
            dataset.append((img_path, corners))

    elif title == "Custom":
        for img_path in images:
            corners = np.array(metadata[img_path.stem])
            corners = order_points_robust(corners)
            dataset.append((img_path, corners))

    else:
        print("Unknown dataset title.")

    return dataset


def run_benchmark(title, images, metadata, results):
    """..."""

    dataset = create_dataset(title, images, metadata)

    iou_scores_ours = []
    iou_scores_paper = []

    for img_path, target_corners in tqdm(dataset):

        img = load_image_RGB(img_path)

        # --- OUR PIPELINE ---

        (_, _, predicted_ours), _ = detect_board(img)

        if predicted_ours is None:
            iou_ours = 0.0
        else:
            predicted_ours = order_points_robust(predicted_ours)
            iou_ours = compute_iou(target_corners, predicted_ours)
            iou_scores_ours.append(iou_ours)

        # --- PAPER PIPELINE ---

        predicted_paper, _ = detect_board_corners(img)

        if predicted_paper is None:
            iou_paper = 0.0
        else:
            iou_paper = compute_iou(target_corners, predicted_paper)
            iou_scores_paper.append(iou_paper)

    results[title] = {}

    results[title]["total_images"] = len(dataset)
    results[title]["average_iou_ours"] = np.mean(iou_scores_ours)
    results[title]["average_iou_paper"] = np.mean(iou_scores_paper)

    success_ours_05 = sum(1 for x in iou_scores_ours if x > 0.5) / len(iou_scores_ours)
    success_paper_05 = sum(1 for x in iou_scores_paper if x > 0.5) / len(iou_scores_paper)
    results[title]["success_ours_05"] = success_ours_05
    results[title]["success_paper_05"] = success_paper_05

    success_ours_08 = sum(1 for x in iou_scores_ours if x > 0.8) / len(iou_scores_ours)
    success_paper_08 = sum(1 for x in iou_scores_paper if x > 0.8) / len(iou_scores_paper)
    results[title]["success_ours_08"] = success_ours_08
    results[title]["success_paper_08"] = success_paper_08


def main():

    # todo: store interesting cases

    results = {}

    unity_metadata = load_metadata(UNITY_METADATA_PATH)
    unity_images = list(UNITY_DIR.glob("*.jpg"))[:500]
    run_benchmark("Unity", unity_images, unity_metadata, results)

    chessred_metadata = load_metadata(CHESSRED_METADATA_PATH)
    chessred_images = list(CHESSRED_DIR.rglob("*.jpg"))[:250]
    run_benchmark("ChessReD", chessred_images, chessred_metadata, results)

    custom_metadata = load_metadata(CUSTOM_METADATA_PATH)
    custom_images = list(CUSTOM_DIR.glob("*.jpg"))
    run_benchmark("Custom", custom_images, custom_metadata, results)

    with open("eval.json", "w") as file:
        json.dump(results, file, indent=4)


if __name__ == "__main__":
    main()
