"""Detect the chess board in a given image."""

import scripts.utils as utils
from pathlib import Path
import cv2
import numpy as np
from rembg import remove, new_session

SESSION = new_session("isnet-general-use")


def detect_board(img_path, debug=False):
    """
    Detect and extract a top-down view of a chess board from the image.

    Pipeline:
    1. Isolate foreground using AI background removal.
    2. Extract the largest contour from the binary mask.
    3. Verify the shape is a valid convex quadrilateral.
    4. Apply perspective correction to crop the board.

    Args:
        img_path (Path): Path to the source image.
        debug (bool): If True, populates the debug dict and prints rejection reasons.

    Returns:
        tuple: (board_view, debug_info)
            - board_view (np.ndarray | None): The rectified board image, or None if failed.
            - debug_info (dict): Intermediate data for visualization.
    """

    debug_info = {}
    link = Path(img_path).resolve().as_uri()

    img = utils.load_image_RGB(img_path)

    # Resize to 320px for speed; we scale coordinates back up later.
    img_resized, scale = utils.resize_image(img, target_width=320)

    img_foreground = remove(img_resized, session=SESSION)
    alpha = img_foreground[:, :, 3]

    if debug:
        debug_info["img_foreground"] = img_foreground

    # If a pixel has an alpha value >= 1, we treat it as part of the foreground.
    _, img_thresh = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)

    # Remove "dust" (OPEN) and fill "holes" inside the board (CLOSE).
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_CLOSE, kernel)

    if debug:
        debug_info["img_morph"] = img_morph

    # RETR_EXTERNAL: We only care about the outer boundary, not internal details/holes.
    contours, _ = cv2.findContours(img_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        if debug:
            print(f"Skipping {link}: No contours detected.")
        return None, debug_info

    # Assumption: The board is the most prominent object after background removal.
    cnt = max(contours, key=cv2.contourArea)

    if cv2.contourArea(cnt) < (img_resized.shape[0] * img_resized.shape[1] * 0.1):
        if debug:
            print(f"Skipping {link}: Largest contour is too small.")
        return None, debug_info

    hull = cv2.convexHull(cnt)

    if debug:
        debug_info["hull"] = hull

    # Solidity Check: Ensure the shape isn't a "star" or "C-shape".
    # A board should be solid (Area roughly equals Hull Area).
    if cv2.contourArea(cnt) / cv2.contourArea(hull) < 0.7:
        if debug:
            print(f"Skipping {link}: Solidity is too far off.")
        return None, debug_info

    # 0.02 (2%) is a standard epsilon for geometrical shapes like rectangles.
    # It allows for slight curvature (lens distortion) while keeping corners sharp.
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    if debug:
        debug_info["approx"] = approx

    if len(approx) != 4:
        if debug:
            print(f"Skipping {link}: Found {len(approx)} corners instead of four.")
        return None, debug_info

    if not cv2.isContourConvex(approx):
        if debug:
            print(f"Skipping {link}: Approximation is not convex.")
        return None, debug_info

    # todo: angle sanity check
    # todo: check for boards that are not fully visible

    corners = np.float32([pt[0] for pt in approx])
    corners = utils.order_points_robust(corners)

    if debug:
        debug_info["corners"] = corners

    # Warp the original high-res image using scaled-up coordinates
    top_down_view = utils.warp_board(img.copy(), corners / scale)

    return top_down_view, debug_info
