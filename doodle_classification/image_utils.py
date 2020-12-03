import base64
import io
import os
from datetime import date

import cv2
import numpy as np
from doodle_classification.logger import get_basic_logger, get_str_time_now
from doodle_classification.time_utils import timeit
from PIL import Image

log = get_basic_logger(__name__, 'DEBUG')

today = date.today()


@timeit
def save_image_history(image_raw, prediction: list):
    dir_name = f'./doodle_history/{today.strftime("%b-%d-%Y")}/'
    os.makedirs(dir_name, exist_ok=True)
    cv2.imwrite(
        '{}{}_{}.jpg'.format(
            dir_name,
            '_'.join(prediction),
            get_str_time_now()
        ),
        image_raw)


@timeit
def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


@timeit
def rgbToString(cv2img):
    _, buffer = cv2.imencode('.png', cv2img)
    imgdata = base64.b64encode(buffer)
    return imgdata


@timeit
def prepareImage(im):
    # gray
    im = cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # binary
    thresh = 127
    im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)[1]
    # see if need to invert
    n_white_pix = np.sum(im == 255)
    n_black_pix = np.sum(im == 0)
    if n_white_pix > n_black_pix:
        im = cv2.bitwise_not(im)
    # trim1, move content to the left-up corner;
    size = len(im[0])
    sum0 = im.sum(axis=0)
    sum1 = im.sum(axis=1)
    for i in range(len(sum0)):
        if sum0[i] == 0:
            im = np.delete(im, 0, 1)
            zero = np.zeros((size, 1))
            im = np.append(im, zero, 1)
        else:
            break
    for i in range(len(sum1)):
        if sum1[i] == 0:
            im = np.delete(im, 0, 0)
            zero = np.zeros((1, size))
            im = np.append(im, zero, 0)
        else:
            break
    # trim2 crop content
    sum3 = im.sum(axis=0)
    sum4 = im.sum(axis=1)
    x2 = 1
    y2 = 1
    while x2 < len(sum3) and sum3[-x2] == 0:
        x2 += 1
    while y2 < len(sum4) and sum4[-y2] == 0:
        y2 += 1
    w = size - x2
    h = size - y2
    contentSize = w if w > h else h
    # only crop if there is realy content
    if contentSize > 16:
        im = im[0:contentSize, 0:contentSize]
    return im


def getBw(img):
    im = img
    # gray
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.GaussianBlur(im, (25, 25), 0)
    # binary
    thresh = 200
    im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)[1]
#     see if need to invert
    n_white_pix = np.sum(im == 255)
    n_black_pix = np.sum(im == 0)
    if n_white_pix > n_black_pix:
        im = cv2.bitwise_not(im)
    return im


def getXYWH(bw):
    #     bw = cv2.bitwise_not(bw)
    sum0 = bw.sum(axis=0)
    sum1 = bw.sum(axis=1)
    w = len(sum0)
    h = len(sum1)
    x2, y2, x1, y1 = 0, 0, 0, 0

    while x1 < w and sum0[x1] == 0:
        x1 += 1
    while y1 < h and sum1[y1] == 0:
        y1 += 1
    while x2 < w and sum0[-x2-1] == 0:
        x2 += 1
    while y2 < h and sum1[-y2-1] == 0:
        y2 += 1

    x2 = w - x2
    y2 = h - y2

    return x1, y1, x2, y2


def getRGBAimg(img):
    bw = getBw(img)
    # get a gray version, to process contour
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Remove noise
    ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(
        thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # creat an empty mask, draw contour on it
    mask = np.zeros(img.shape, np.uint8)
    for c in contours:
        acc = 0.03 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, acc, True)
        cv2.fillPoly(mask, [approx], color=(255, 255, 255))
    # get alpha from mask
    h = img.shape[0]
    w = img.shape[1]
    mb, mg, mr = cv2.split(mask)
    b, g, r = cv2.split(img)
    alpha = np.zeros(b.shape, np.uint8)
    for x in range(0, w):
        for y in range(0, h):
            if mb[x, y] != 255 and bw[x, y] == 0:
                alpha[x, y] = 0
            else:
                alpha[x, y] = 255
    # merge img with alpha
    rgba = cv2.merge((b, g, r, alpha))
    # cut empty part
    x1, y1, x2, y2 = getXYWH(bw)
    rgba = rgba[y1:y2, x1:x2]

    h = rgba.shape[0]
    fy = 480 / h
    rgba = cv2.resize(rgba, None, fx=(fy+1)/2, fy=(fy+1)/2)
    rgba = cv2.threshold(rgba, 127, 255, cv2.THRESH_BINARY)[1]
    return rgba
