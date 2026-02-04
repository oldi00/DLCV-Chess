"""..."""

import cv2
import numpy as np


def load_image_RGB(image_path):
    """Load the image at the given path in RGB color space."""

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def resize_image(img, target_width=512):
    """Resize the given image to the specified width and keep the ratio."""

    height, width = img.shape[:2]

    scale = target_width / width

    new_width, new_height = target_width, int(height * scale)

    # Change the interpolation based on whether the images needs to be
    # bigger or smaller to ensure quality.
    interpolation = cv2.INTER_AREA if new_width < width else cv2.INTER_CUBIC

    resized = cv2.resize(img, (new_width, new_height), interpolation=interpolation)

    return resized


def preprocess_image(img):
    """Preprocess the given image by converting to grayscale and applying a blur."""

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), sigmaX=0)

    return blur


def detect_canny_edges(img):
    """Detect edges in the given image with canny edge detection."""

    sigma = 0.33
    v = np.median(img)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edges = cv2.Canny(img, lower, upper)

    return edges


def detect_hough_lines(edges, threshold=60):
    """Detect lines in the given image using hough lines."""

    hough_lines = cv2.HoughLines(edges, rho=1, theta=np.pi/360, threshold=threshold)

    return hough_lines.squeeze()


def find_intersections(lines_a, lines_b):
    """Find the (x, y) intersection points between two sets of lines."""

    intersections = []

    for rho1, theta1 in lines_a:
        for rho2, theta2 in lines_b:

            a = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([rho1, rho2])

            x0, y0 = np.linalg.solve(a, b)

            intersections.append([int(np.round(x0)), int(np.round(y0))])

    return np.array(intersections)
