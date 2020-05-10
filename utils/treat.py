import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2

def centering(img, x, y, w, h):
    """ Center the img in the rectangle with the upper corner at x, y and with lenght w, h """
    n, p = img.shape[0], img.shape[1]
    res = np.zeros((h,w,3), dtype="uint8")
    #res = np.zeros((h, w, 3)), dtype = "uint8")
    for i in range(x, x + w):
        for j in range(y, y + h):
            if j<n-1 and i<p-1 and 0<=j-y<h and 0<=i-x<w:
                res[j - y][i - x] = img[j][i]
    return res


def rotation(img, theta):
    """Rotate img with an angle theta """
    rotated = ndimage.rotate(img, theta*180/np.pi)
    plt.imshow(rotated)
    plt.show()
