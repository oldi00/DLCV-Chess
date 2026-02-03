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
"""

import scripts.utils as utils
import cv2
import numpy as np
from scipy.signal import find_peaks


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


def find_candidates(lines):
    """Find candidate lines by looking for peaks in smoothed histogram."""

    rhos, thetas = lines[:, 0], lines[:, 1]

    min_rho = int(np.min(rhos))
    max_rho = int(np.max(rhos))

    bins = np.arange(min_rho - 10, max_rho + 10)
    hist, bin_edges = np.histogram(rhos, bins=bins)

    hist_img = hist.astype(np.float32).reshape(1, -1)
    smooth_hist = cv2.GaussianBlur(hist_img, (11, 1), sigmaX=3, sigmaY=0).flatten()

    peaks_indices, _ = find_peaks(smooth_hist)
    peak_rhos = bin_edges[peaks_indices]

    candidates = []
    for rho in peak_rhos:

        mask = np.abs(rhos - rho) < 10
        related_thetas = thetas[mask]

        avg_theta = np.median(related_thetas)
        candidates.append([rho, avg_theta])

    return candidates, (smooth_hist, bin_edges, peak_rhos)


def find_board(image_path, debug=False):
    """..."""

    img = utils.load_image_RGB(image_path)
    img_resized = utils.resize_image(img)
    img_preprocessed = utils.preprocess_image(img_resized)

    img_edges = utils.detect_canny_edges(img_preprocessed)

    lines = utils.detect_hough_lines(img_edges)
    lines_a, lines_b = cluster_lines(lines)

    # We assume that most Hough lines belong to the chess board and
    # therefore pick only the strongest lines for each cluster.
    lines_a = lines_a[:50]
    lines_b = lines_b[:50]

    lines_a = remove_outliers_by_angle(lines_a)
    lines_b = remove_outliers_by_angle(lines_b)

    candidates_a, hist_a = find_candidates(lines_a)
    candidates_b, hist_b = find_candidates(lines_b)

    if not debug:
        return None

    debug_info = {
        "img_resized": img_resized,
        "img_preprocessed": img_preprocessed,
        "img_edges": img_edges,
        "lines": (lines_a, lines_b),
        "candidates": (candidates_a, candidates_b),
        "hist": (hist_a, hist_b)
    }

    return None, debug_info
