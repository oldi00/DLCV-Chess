"""Plotting and drawing helpers for debugging chessboard CV pipelines."""

import cv2
import numpy as np
from matplotlib import pyplot as plt


def resize_and_crop(img: np.ndarray, target_size: tuple = (512, 512)) -> np.ndarray:
    """Resize the image to fit the target size while maintaining aspect ratio, then crop the center."""
    h, w = img.shape[:2]
    target_w, target_h = target_size

    scale = max(target_w / w, target_h / h)

    new_w = int(np.ceil(w * scale))
    new_h = int(np.ceil(h * scale))

    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA

    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    start_x = (new_w - target_w) // 2
    start_y = (new_h - target_h) // 2

    cropped = resized[start_y:start_y+target_h, start_x:start_x+target_w]

    return cropped


def plot_images(plot_title: str, images: list, cols: int = 3, normalize_img_size: bool = True) -> None:
    """Plot a list of images with optional titles in a grid layout."""
    n = len(images)
    rows = int(np.ceil(n / cols))

    row_height = 4
    fig_width = cols * 4
    fig_height = rows * row_height

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        facecolor='white'
    )

    physical_offset = 0.3  # Inches
    y_pos = 1 + (physical_offset / fig_height)

    fig.suptitle(
        plot_title,
        fontsize=24,
        fontweight='bold',
        fontfamily='sans-serif',
        color='#2c3e50',
        y=y_pos,
        va='bottom'
    )

    if n == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for ax, (title, img) in zip(axes_flat, images):

        img = img.astype(np.uint8)
        if normalize_img_size:
            img = resize_and_crop(img)

        ax.imshow(img, cmap="gray", interpolation='bicubic')
        ax.axis("off")

        if title is not None:
            ax.set_title(
                title,
                fontsize=14,
                fontfamily='sans-serif',
                fontweight='medium',
                color='#555555',
                pad=12,
            )

    # Hide empty slots.
    for ax in axes_flat[n:]:
        ax.axis("off")
        ax.set_visible(False)

    plt.show()


def draw_lines(img: np.ndarray, lines: list, color: tuple = (255, 255, 255), size: int = 2) -> np.ndarray:
    """Draw lines on the image given in polar coordinates (rho, theta)."""
    dist = max(img.shape) * 100

    for rho, theta in lines:

        a = np.cos(theta)
        b = np.sin(theta)

        x0, y0 = a * rho, b * rho

        x1 = int(x0 + dist * (-b))
        y1 = int(y0 + dist * a)
        x2 = int(x0 - dist * (-b))
        y2 = int(y0 - dist * a)

        cv2.line(img, (x1, y1), (x2, y2), color, size)

    return img


def draw_many_lines(img: np.ndarray, lines_list: list, colors: list) -> np.ndarray:
    """Draw multiple sets of lines on the image, each set with a different color."""
    img = img.copy()

    for lines, color in zip(lines_list, colors):
        draw_lines(img, lines, color)

    return img


def draw_board(img: np.ndarray, board_corners: list, color: tuple = (0, 255, 127), size: int = 3) -> np.ndarray:
    """Draw a polygon connecting the given board corners on the image."""
    board_corners = np.array(board_corners)

    center = np.mean(board_corners, axis=0)

    diff = board_corners - center
    angles = np.arctan2(diff[:, 1], diff[:, 0])

    board_corners = board_corners[angles.argsort()]

    points = np.array(board_corners, np.int32)

    points = points.reshape((-1, 1, 2))

    img_board = cv2.polylines(img.copy(), [points], True, color, size)

    return img_board


def draw_points(img: np.ndarray, points: list, color: tuple = (250, 20, 50)) -> np.ndarray:
    """Draw points on the image."""
    for x, y in points:
        cv2.circle(img, (int(x), int(y)), 5, color, -1)

    return img


def draw_contours(img: np.ndarray, contours: list, color: tuple = (0, 255, 127)) -> np.ndarray:
    """Draw contours on the image."""
    return cv2.drawContours(img.copy(), contours, -1, color, 2)


def plot_histograms(plot_title: str, *histograms: tuple, cols: int = 2) -> None:
    """Plot multiple histograms in a grid layout, each with its peaks highlighted."""
    n = len(histograms)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.15, hspace=0.1, wspace=0.1)

    fig.suptitle(
        plot_title,
        fontsize=24,
        fontweight='bold',
        fontfamily='sans-serif',
    )

    for i, hist_data in enumerate(histograms):
        ax = axes.flat[i]
        hist, bin_edges, peaks = hist_data
        ax.plot(bin_edges[:-1], hist, color='blue', label='Density')
        ax.fill_between(bin_edges[:-1], hist, color='blue', alpha=0.1)
        ax.set_xlabel("Distance (Rho)")
        ax.set_ylabel("Strength")
        ax.grid(True, alpha=0.3)
        peak_indices = (peaks - bin_edges[0]).astype(int)
        peak_indices = np.clip(peak_indices, 0, len(hist) - 1)
        peak_heights = hist[peak_indices]
        ax.plot(peaks, peak_heights, 'rx', markersize=8)

    plt.show()


def draw_grid_on_board(img: np.ndarray, color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draw a grid on the board."""
    img_grid = img.copy()

    h, w = img_grid.shape[:2]

    step_x = w / 8.0
    step_y = h / 8.0

    # Draw Vertical Lines (columns)
    for i in range(0, 9):
        x = int(round(i * step_x))
        if i == 8:
            x = w - 1

        cv2.line(img_grid, (x, 0), (x, h), color, thickness)

    # Draw Horizontal Lines (rows)
    for i in range(0, 9):
        y = int(round(i * step_y))
        if i == 8:
            y = h - 1

        cv2.line(img_grid, (0, y), (w, y), color, thickness)

    return img_grid
