"""Detect the chess board in a given image."""

import scripts.utils as utils
import scripts.utils_chess_cv as utils_chess_cv
import sys
import cv2
import numpy as np
from rembg import remove, new_session

SESSION = new_session("isnet-general-use")


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
        debug (bool): If True, populates the debug dict and prints rejection reasons.

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
        return None, debug_info

    # Assumption: The board is the most prominent object after background removal.
    cnt = max(contours, key=cv2.contourArea)

    if cv2.contourArea(cnt) < (img_resized.shape[0] * img_resized.shape[1] * 0.1):
        debug_info["error"] = "Largest contour is too small."
        return None, debug_info

    hull = cv2.convexHull(cnt)

    if debug:
        debug_info["hull"] = hull

    # Solidity Check: Ensure the shape isn't a "star" or "C-shape".
    # A board should be solid (Area roughly equals Hull Area).
    if cv2.contourArea(cnt) / cv2.contourArea(hull) < 0.7:
        debug_info["error"] = "Solidity is too far off."
        return None, debug_info

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
        return None, debug_info

    if debug:
        debug_info["approx"] = approx

    min_eps, max_eps = potential_epsilons[0], potential_epsilons[-1]

    raw_score = 1.0 - ((eps - min_eps) / (max_eps - min_eps))
    confidence_score = max(0.0, min(1.0, raw_score))

    debug_info["confidence"] = "HIGH" if confidence_score > 0.9 else "LOW"

    # todo: angle sanity check
    # todo: check for boards that are not fully visible

    corners = np.float32([pt[0] for pt in approx])
    corners = utils_chess_cv.order_points_robust(corners)

    if debug:
        debug_info["corners"] = corners

    # Warp the original high-res image using scaled-up coordinates
    top_down_view = utils_chess_cv.warp_board(img.copy(), corners / scale)

    return top_down_view, debug_info


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

    board, _ = detect_board(img_rgb, debug=False)

    if board is None:
        sys.stdout.write("NO_BOARD_DETECTED")
    else:

        # Encode the result back to an image format.
        board_bgr = cv2.cvtColor(board, cv2.COLOR_RGB2BGR)
        success, encoded_img = cv2.imencode('.png', board_bgr)

        if success:
            # Write raw image bytes to the stdout buffer.
            sys.stdout.buffer.write(encoded_img.tobytes())
        else:
            sys.stdout.write("NO_BOARD_DETECTED")


if __name__ == "__main__":
    main()
