import cv2


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
