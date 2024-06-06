import json

import cv2

import dithering as dth


def dither_image(image_path, method="floyd-steinberg"):
    """
    Dithers the given image using the specified dithering method.

    Args:
        image_path (str): The path to the image file.
        method (str, optional): The dithering method to use. Defaults to "floyd-steinberg".

    Returns:
        numpy.ndarray: The dithered image as a NumPy array.
    """
    img = cv2.imread(image_path, 0)
    d_img = dth.dither(img, method=method)
    return d_img


def save_dither(d_img, output_path="dithered.json", save_image=True):
    """
    Save the dithered image as a JSON file and optionally as a PNG image.

    Args:
        d_img (numpy.ndarray): The dithered image as a NumPy array.
        output_path (str, optional): The path to save the JSON file. Defaults to "dithered.json".
        save_image (bool, optional): Whether to save the dithered image as a PNG. Defaults to True.
    """

    out = cv2.normalize(d_img.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
    out = out.tolist()
    with open(output_path, "w") as f:
        json.dump(out, f)

    if save_image:
        cv2.imwrite("dithered2.png", d_img * 255)


def main():
    image_path = "./asdfg.png"
    d_img = dither_image(image_path)
    save_dither(d_img)


if __name__ == "__main__":
    main()
