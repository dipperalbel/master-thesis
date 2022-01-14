import numpy as np

def center_crop(img, size, offset=None):
    """
    Function used to crop the image to a square format, within a certain offset from the center

    Args:
        img: image to be center cropped
        size: desired size of the crop
        offset: the offset of the crop (can be used with a random function to diversify the dataset)
    """
    h, w, c = img.shape
    th, tw = size, size
    half_i = int(np.round((h - th) / 2.))
    half_j = int(np.round((w - tw) / 2.))

    if offset is not None:
        i = np.clip(half_i+offset[0], 0, th+h)
        j = np.clip(half_j+offset[1], 0, tw+w)
        image = img[i:i + th, j:j + tw, :]
    else:
        image = img[half_i:half_i + th, half_j:half_j + tw, :]
    return image