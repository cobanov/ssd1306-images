import cv2


def _simple2D_dither(img):
    """
    Applies simple 2D dithering to the input image.

    Args:
        img: The input image to be dithered.

    Returns:
        The dithered image.
    """
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    rows, cols = img.shape
    out = cv2.normalize(img.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            err = out[i, j] - (out[i, j] > 0.5)
            out[i, j] = float(out[i, j] > 0.5)
            out[i, j + 1] += 0.5 * err
            out[i + 1, j] += 0.5 * err

    return out[1 : rows - 1, 1 : cols - 1]


def _floyd_steinberg_dither(img):
    """
    Applies the Floyd-Steinberg dithering algorithm to the input image.

    Parameters:
    img (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The dithered image.
    """
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    rows, cols = img.shape
    out = cv2.normalize(img.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            err = out[i, j] - (out[i, j] > 0.5)
            out[i, j] = float(out[i, j] > 0.5)
            out[i, j + 1] += 7 / 16 * err
            out[i + 1, j - 1] += 3 / 16 * err
            out[i + 1, j] += 5 / 16 * err
            out[i + 1, j + 1] += 1 / 16 * err

    return out[1 : rows - 1, 1 : cols - 1]


def _jarvis_judice_ninke_dither(img):
    """
    Applies the Jarvis-Judice-Ninke dithering algorithm to the input image.

    Parameters:
    img (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The dithered image.
    """
    img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    rows, cols = img.shape
    out = cv2.normalize(img.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)

    for i in range(2, rows - 2):
        for j in range(2, cols - 2):
            err = out[i, j] - (out[i, j] > 0.5)
            out[i, j] = float(out[i, j] > 0.5)
            out[i, j + 1] += 7 / 48 * err
            out[i, j + 2] += 5 / 48 * err
            out[i + 1, j - 2] += 3 / 48 * err
            out[i + 1, j - 1] += 5 / 48 * err
            out[i + 1, j] += 7 / 48 * err
            out[i + 1, j + 1] += 5 / 48 * err
            out[i + 1, j + 2] += 3 / 48 * err
            out[i + 2, j - 2] += 1 / 48 * err
            out[i + 2, j - 1] += 3 / 48 * err
            out[i + 2, j] += 5 / 48 * err
            out[i + 2, j + 1] += 3 / 48 * err
            out[i + 2, j + 2] += 1 / 48 * err

    return out[2 : rows - 2, 2 : cols - 2]


def dither(img, method="floyd-steinberg"):
    """
    Apply dithering to the input image using the specified method.

    Args:
        img: The input image to be dithered.
        method (str): The dithering method to be used. Available options are:
            - "simple2D": Simple 2D dithering.
            - "floyd-steinberg": Floyd-Steinberg dithering.
            - "jarvis-judice-ninke": Jarvis-Judice-Ninke dithering.

    Returns:
        The dithered image.

    Raises:
        ValueError: If the specified method does not exist.

    """
    if method == "simple2D":
        return _simple2D_dither(img)
    elif method == "floyd-steinberg":
        return _floyd_steinberg_dither(img)
    elif method == "jarvis-judice-ninke":
        return _jarvis_judice_ninke_dither(img)
    else:
        raise ValueError(
            'Specified method does not exist. Available methods: "simple2D", "floyd-steinberg", "jarvis-judice-ninke"'
        )
