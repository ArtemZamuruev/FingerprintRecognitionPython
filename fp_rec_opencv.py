from os.path import join, isfile
from os import listdir
import cv2 as cv
import numpy as np
from time import time


def openImagesInFolder(folder):
    img_paths = [join(folder, fname)
                 for fname in listdir(folder)
                 if isfile(join(folder, fname))]
    try:
        return [cv.imread(img_path) for img_path in img_paths]
    except Exception:
        print "Something went wrong during opening images in folder: %s" % folder
        return False


def showOneImage(img, winname="Fingerprints"):
    cv.imshow(winname, img)


def getKey(delay=0):
    keycode = cv.waitKey(delay)
    if keycode != -1:
        return keycode & 0xFF


def drawRects(img, rects, color=(20, 20, 250)):
    if rects is None:
        return
    for rect in rects:
        x = rect[0]
        y = rect[1]
        w = rect[2]
        h = rect[3]
        cv.rectangle(img, (x, y), (x + w, y + h), color)


def writeImg(img, folder="./", ext="png"):
    nf_name = "res_%.5f.%s" % (time(), ext)
    nf_path = join(folder, nf_name)
    try:
        cv.imwrite(nf_path, img)
    except Exception:
        print "Error writing image. Path: %s" % nf_path
        return False

imgs = openImagesInFolder("./Fingerprints/")
for img in imgs:
    showOneImage(img, winname="Original")
    kernel = np.array([
        [1, 1, 1, 1, 1],
        [1, 3, 3, 3, 1],
        [1, 3, 5, 3, 1],
        [1, 3, 3, 3, 3],
        [1, 1, 1, 1, 1]
    ])
    closed = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    showOneImage(closed, "Closed")
    getKey()