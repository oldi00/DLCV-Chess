import cv2
import numpy as np


def load_image_RGB(img_path):
    """Load the image at the given path in RGB color space."""

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def resize_image(img, target_width=640):
    """Resize the given image to the specified width and keep the ratio."""

    height, width = img.shape[:2]

    scale = target_width / width

    new_width, new_height = target_width, int(height * scale)

    # Change the interpolation based on whether the images needs to be
    # bigger or smaller to ensure quality.
    interpolation = cv2.INTER_AREA if new_width < width else cv2.INTER_CUBIC

    resized = cv2.resize(img, (new_width, new_height), interpolation=interpolation)

    return resized, scale


def order_points_robust(pts):
    """
    Attribution:
    -----------
    Source: Adapted from 'chess-cv'
    URL: https://github.com/Luthiraa/CVChess/blob/main/src/final_notebook.ipynb
    Authors: Luthira Abeykoon, Darshan Kasundra, Gawtham Senthilvelan, Ved Patel
    """

    # First, get bounding box
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    # sort by y-coord
    tl, bl = left_most[np.argsort(left_most[:, 1]), :]
    tr, br = right_most[np.argsort(right_most[:, 1]), :]

    return np.array([tl, tr, br, bl], dtype="float32")  # Clockwise


def warp_board(img, corners, output_size=400):
    """
    Attribution:
    -----------
    Source: Adapted from 'chess-cv'
    URL: https://github.com/Luthiraa/CVChess/blob/main/src/final_notebook.ipynb
    Authors: Luthira Abeykoon, Darshan Kasundra, Gawtham Senthilvelan, Ved Patel
    """

    src_pts = order_points_robust(corners)
    dst_pts = np.array([
        [0, 0],
        [output_size-1, 0],
        [output_size-1, output_size-1],
        [0, output_size-1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (output_size, output_size))

    return warped
