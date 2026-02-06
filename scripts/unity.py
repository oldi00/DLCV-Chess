"""Transforms the Unity snapshots into a 400x400x3 top-down view for model training."""

import utils
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

DIR_SNAPSHOTS = Path(r"C:\Users\miles\code\uni\dlcv\ChessDataGen\Snapshots")
DIR_CLOUD = Path(
    r"G:\.shortcut-targets-by-id\1kK41aQ1XMbjKKuinmDPfwUPClMvdcMvu\DLCV\Synthetic Data (Unity)")


def main():

    with open(DIR_SNAPSHOTS / "metadata.json", "r") as file:
        metadata = json.load(file)

    shutil.copy2(DIR_SNAPSHOTS / "metadata.json", DIR_CLOUD / "metadata.json")

    for img_path, img_data in tqdm(metadata.items(), desc="Transforming Images", unit="img"):

        img = utils.load_image_RGB(DIR_SNAPSHOTS / img_path)

        corners = np.array(img_data["corners"], dtype="float32")
        corners = utils.order_points(corners)

        img = utils.get_top_down_view(img, corners, (400, 400))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(DIR_CLOUD / Path(img_path).name), img)


if __name__ == "__main__":
    main()
