import cv2
import numpy as np
from matplotlib import pyplot as plt


def resize_and_crop(img, target_size=(512, 512)):

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


def plot_images(plot_title, images, cols=3, normalize_img_size=True):

    n = len(images)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), constrained_layout=True)

    fig.suptitle(
        plot_title,
        fontsize=24,
        fontweight='bold',
        fontfamily='sans-serif',
        y=1.07,
    )

    for i, (title, img) in enumerate(images):
        ax = axes.flat[i]
        img = img.astype(np.uint8)
        if normalize_img_size:
            img = resize_and_crop(img)
        ax.imshow(img, cmap="gray")
        if title is not None:
            ax.set_title(
                f"{i+1}. {title}",
                fontsize=12,
                fontfamily='sans-serif',
                color='#444',
                pad=10,
            )

    for ax in axes.flat:
        ax.axis("off")

    plt.show()


def draw_lines(img, lines, color=(255, 255, 255), size=2):

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


def draw_many_lines(img, lines_list, colors):

    img = img.copy()

    for lines, color in zip(lines_list, colors):
        draw_lines(img, lines, color)

    return img


def draw_board(img, board_corners, color=(0, 255, 127), size=3):

    board_corners = np.array(board_corners)

    center = np.mean(board_corners, axis=0)

    diff = board_corners - center
    angles = np.arctan2(diff[:, 1], diff[:, 0])

    board_corners = board_corners[angles.argsort()]

    points = np.array(board_corners, np.int32)

    points = points.reshape((-1, 1, 2))

    img_board = cv2.polylines(img.copy(), [points], True, color, size)

    return img_board


def draw_points(img, points, color=(250, 20, 50)):

    for x, y in points:
        cv2.circle(img, (int(x), int(y)), 5, color, -1)

    return img


def plot_histograms(plot_title, *histograms, cols=2):

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


def draw_grid_on_board(img, color=(0, 255, 0), thickness=2):

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
