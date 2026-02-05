"""
Detect the chess board in a given image.

Pipeline:
- Resize the image to a standardized size.
- Preprocess the image (grayscale and blur).
- Detect edges via canny edge detection.
- Detect hough lines from the edges.
- Cluster lines into two groups based on their angle.
- Remove outlier lines based on their theta angle.
- Find candidate lines using a smoothed histogram of rho values (distance).
#todo: finish
"""

import scripts.utils as utils
import cv2
import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN


def cluster_lines(lines):
    """Segment the given lines into two clusters based on their angles."""

    thetas = lines[:, 1]

    data = np.column_stack((np.cos(2 * thetas), np.sin(2 * thetas)))

    flags = cv2.KMEANS_RANDOM_CENTERS
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    _, labels, _ = cv2.kmeans(data, 2, None, criteria, 10, flags)

    labels = labels.ravel()

    return lines[labels == 0], lines[labels == 1]


def remove_outliers_by_angle(lines, threshold=0.3):
    """Remove outliers based on the theta angle and handle the 0/180 degree wrap-around."""

    thetas = lines[:, 1]

    # We double the angle (2*theta) to map [0, pi) to the full circle [0, 2*pi)
    # This ensures 0 radians and pi radians are treated as the same direction.
    sin_sum = np.sum(np.sin(2 * thetas))
    cos_sum = np.sum(np.cos(2 * thetas))

    # arctan2 returns values in [-pi, pi], so we divide by 2 to get back to [-pi/2, pi/2]
    avg_theta = 0.5 * np.arctan2(sin_sum, cos_sum)

    # Calculate the smallest angular difference with standard formula: min(|x - y|, pi - |x - y|)
    diffs = np.abs(thetas - avg_theta)

    shortest_diffs = np.minimum(diffs, np.pi - diffs)

    return lines[shortest_diffs < threshold]


def normalize_lines(lines):
    """Unifies lines that are split between 0 and 180 degrees."""

    thetas = lines[:, 1]

    has_low = np.any(thetas < np.pi / 4)
    has_high = np.any(thetas > 3 * np.pi / 4)

    if has_low and has_high:
        # We are likely looking at vertical lines split across the wrap-around.
        # We flip the "high" angle lines to align with the "low" angle ones.

        new_lines = lines.copy()

        # Mask for lines that need flipping (angles > 90 degrees).
        mask = thetas > np.pi / 2

        # Flip rho: positive becomes negative (or vice versa).
        new_lines[mask, 0] = -new_lines[mask, 0]

        # Flip theta: 179 deg becomes -1 deg (conceptually).
        # We don't strictly need the theta for the rho-gap logic, but it keeps the data consistent.
        new_lines[mask, 1] = new_lines[mask, 1] - np.pi

        return new_lines

    return lines


def find_candidates(lines):
    """Find candidate lines by looking for peaks in smoothed histogram."""

    rhos, thetas = lines[:, 0], lines[:, 1]

    min_rho = int(np.min(rhos))
    max_rho = int(np.max(rhos))

    bins = np.arange(min_rho - 10, max_rho + 10)
    hist, bin_edges = np.histogram(rhos, bins=bins)

    hist_img = hist.astype(np.float32).reshape(1, -1)
    smooth_hist = cv2.GaussianBlur(hist_img, (11, 1), sigmaX=3, sigmaY=0).flatten()

    peak_config = {
        "distance": 10,
        "height": 0.1 * np.max(smooth_hist),
        "prominence": 0.1 * np.max(smooth_hist),
    }

    peaks_indices, _ = find_peaks(smooth_hist, **peak_config)
    peak_rhos = bin_edges[peaks_indices]

    candidates = []
    for rho in peak_rhos:

        mask = np.abs(rhos - rho) < 10
        related_thetas = thetas[mask]

        avg_theta = np.median(related_thetas)
        candidates.append([rho, avg_theta])

    return np.array(candidates), (smooth_hist, bin_edges, peak_rhos)


def get_board_lines(lines):
    """..."""

    lines = lines[np.argsort(lines[:, 0])]

    sorted_rhos = lines[:, 0]
    gaps = np.diff(sorted_rhos)

    estimated_square_size = np.median(gaps)

    if np.isnan(estimated_square_size) or estimated_square_size <= 0:
        return np.array([])

    cluster_eps = estimated_square_size * 2.2

    rhos_clean = lines[:, 0].reshape(-1, 1)
    grid_clustering = DBSCAN(eps=cluster_eps, min_samples=2).fit(rhos_clean)
    grid_labels = grid_clustering.labels_

    unique_labels = set(grid_labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    best_label = -1
    max_count = 0
    for label in unique_labels:
        count = np.sum(grid_labels == label)
        if count > max_count:
            max_count = count
            best_label = label

    return lines[grid_labels == best_label]


def find_board(image_path, debug=False):
    """Detect the chess board in the given image."""

    img = utils.load_image_RGB(image_path)

    img_resized, scale = utils.resize_image(img, target_width=640)
    img_preprocessed = utils.preprocess_image(img_resized)

    img_edges = utils.detect_canny_edges(img_preprocessed)

    lines = utils.detect_hough_lines(img_edges, threshold=60)
    lines_a, lines_b = cluster_lines(lines)

    # We assume that most Hough lines belong to the chess board and
    # therefore pick only the strongest lines for each cluster.
    lines_a = lines_a[:100]
    lines_b = lines_b[:100]

    lines_a = normalize_lines(lines_a)
    lines_b = normalize_lines(lines_b)

    angle_threshold = 0.25
    lines_a = remove_outliers_by_angle(lines_a, angle_threshold)
    lines_b = remove_outliers_by_angle(lines_b, angle_threshold)

    candidates_a, hist_a = find_candidates(lines_a)
    candidates_b, hist_b = find_candidates(lines_b)

    board_lines_a = get_board_lines(candidates_a)
    board_lines_b = get_board_lines(candidates_b)

    if len(board_lines_a) < 2 or len(board_lines_b) < 2:
        return (None, None) if debug else None

    outer_lines = utils.get_outer_lines(board_lines_a, board_lines_b)

    corners = utils.find_intersections(*outer_lines)
    corners = utils.order_points(corners)

    corners_scaled = corners / scale
    chess_board = utils.get_top_down_view(img, corners_scaled, (400, 400))

    if not debug:
        return chess_board

    debug_info = {
        "img_resized": img_resized,
        "img_preprocessed": img_preprocessed,
        "img_edges": img_edges,
        "lines": (lines_a, lines_b),
        "candidates": (candidates_a, candidates_b),
        "hist": (hist_a, hist_b),
        "board_lines": (board_lines_a, board_lines_b),
        "outer_lines": outer_lines,
        "corners": corners,
        "corners_scaled": corners_scaled,
    }

    return chess_board, debug_info
