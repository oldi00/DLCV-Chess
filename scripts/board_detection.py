"""
Detect the chess board in a given image.

An .exe of this script can be installed with the following command:
>>> pyinstaller --onedir --copy-metadata=pymatting --copy-metadata=tqdm --copy-metadata=rembg
    --hidden-import=onnxruntime --hidden-import=scipy scripts/board_detection.py
"""

import scripts.utils as utils
import scripts.utils_chess_cv as utils_chess_cv
import sys
import json
import base64
import cv2
import numpy as np
from rembg import remove, new_session

SESSION = new_session("isnet-general-use")


def get_confidence_score(cnt_area, hull_area, approx):
    """Computes a confidence score (0.0 - 1.0) based on solidity and quad fit."""

    if hull_area <= 0:
        return 0.0

    approx_area = cv2.contourArea(approx)

    solidity = cnt_area / hull_area
    quad_fit = approx_area / hull_area

    return round(solidity * quad_fit, 3)


def detect_board(img, debug=False):
    """
    Detect and extract a top-down view of a chess board from the image.

    Pipeline:
    1. Isolate foreground using AI background removal.
    2. Extract the largest contour from the binary mask.
    3. Verify the shape is a valid convex quadrilateral.
    4. Apply perspective correction to crop the board.

    Args:
        img (np.ndarray): The source image as a NumPy array (RGB).
        debug (bool): If True, populates the debug dict.

    Returns:
        tuple: (board_view, debug_info)
            - board_view (np.ndarray | None): The rectified board image, or None if failed.
            - debug_info (dict): Intermediate data for visualization.
    """

    debug_info = {}

    # Resize to 320px for speed; we scale coordinates back up later.
    img_resized, scale = utils.resize_image(img, target_width=320)

    if debug:
        debug_info["img_resized"] = img_resized

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
        debug_info["error"] = "No contours detected."
        return (None, None, None), debug_info

    # Assumption: The board is the most prominent object after background removal.
    cnt = max(contours, key=cv2.contourArea)
    cnt_area = cv2.contourArea(cnt)

    if cnt_area < (img_resized.shape[0] * img_resized.shape[1] * 0.1):
        debug_info["error"] = "Largest contour is too small."
        return (None, None, None), debug_info

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)

    if debug:
        debug_info["hull"] = hull

    # Solidity Check: Ensure the shape isn't a "star" or "C-shape".
    # A board should be solid (Area roughly equals Hull Area).
    if cnt_area / hull_area < 0.7:
        debug_info["error"] = "Solidity is too far off."
        return (None, None, None), debug_info

    potential_epsilons = np.linspace(0.02, 0.04, num=100)

    approx = None
    found_quad = False

    perimeter = cv2.arcLength(hull, True)

    for eps in potential_epsilons:

        epsilon = eps * perimeter
        approx = cv2.approxPolyDP(hull, epsilon, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            found_quad = True
            break

    if not found_quad:
        debug_info["error"] = f"Could not simplify to 4 corners (Last attempt: {len(approx)})"
        return (None, None, None), debug_info

    if debug:
        debug_info["approx"] = approx

    confidence_score = get_confidence_score(cnt_area, hull_area, approx)
    debug_info["confidence"] = confidence_score

    corners = np.float32([pt[0] for pt in approx])
    corners = utils_chess_cv.order_points_robust(corners)

    if debug:
        debug_info["corners"] = corners

    # Warp the original high-res image using scaled-up coordinates
    top_down_view = utils_chess_cv.warp_board(img.copy(), corners / scale)

    return (top_down_view, confidence_score, corners / scale), debug_info


def main():
    """
    Read raw image bytes from stdin, attempt to detect a chess board, and write
    the resulting PNG bytes or failure message to stdout.
    """

    # Read raw image bytes from standard input (stdin).
    try:

        input_bytes = sys.stdin.buffer.read()
        if not input_bytes:
            raise ValueError("Empty input.")

        # Decode bytes into an OpenCV image.
        array = np.frombuffer(input_bytes, np.uint8)
        img_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise ValueError("Failed to decode image bytes.")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    except Exception:
        sys.stdout.write("NO_BOARD_DETECTED")
        return

    (board, score, corners), _ = detect_board(img_rgb, debug=False)

    if board is None:
        sys.stdout.write("NO_BOARD_DETECTED")
    else:

        # Encode the result back to an image format.
        board_bgr = cv2.cvtColor(board, cv2.COLOR_RGB2BGR)
        success, encoded_img = cv2.imencode('.png', board_bgr)

        if success:

            # Convert bytes to Base64 String for JSON compatibility.
            img_base64 = base64.b64encode(encoded_img).decode('utf-8')

            response = {
                "warped_image_bytes": img_base64,
                "confidence_score": float(score),  # ensure native python float
                "detected_corners": corners.tolist()  # ensure list, not numpy array
            }

            sys.stdout.write(json.dumps(response))

        else:
            sys.stdout.write("NO_BOARD_DETECTED")


if __name__ == "__main__":
    main()
