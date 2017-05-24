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
        return [(cv.imread(img_path), img_path) for img_path in img_paths]
    except Exception:
        print "Something went wrong during opening images in folder: %s" % folder
        return False


def showImage(img, winname="Fingerprints", trackbars=None):
    cv.imshow(winname, img)
    if trackbars is not None:
        for tr in trackbars:
            cv.createTrackbar(tr["Name"],
                              winname,
                              tr["Value"],
                              tr["Count"],
                              tr["Update"])


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


def imgToGray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def applyBlur(img, method="GAUSSIAN", kernsize=7, sigma=2):
    result = None
    if method == "GAUSSIAN":
        result = cv.GaussianBlur(img, (kernsize, kernsize), sigma)
    elif method == "MEDIAN":
        result = cv.medianBlur(img, kernsize)
    else:
        return False
    return result


def otsuThreshold(img):
    ret, thr = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
    return thr


def adaptiveThreshold(img, method="MEAN", bsize=9, c=1):
    result = None
    if method == "GAUSSIAN":
        result = cv.adaptiveThreshold(
            img,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,
            bsize,
            c)
    elif method == "MEAN":
        result = cv.adaptiveThreshold(
            img,
            255,
            cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,
            bsize,
            c)
    else:
        return False
    return result


def sharpenImage(img, kcenter=9):
    kernel = np.array([
        [-1, -1, -1],
        [-1, kcenter, -1],
        [-1, -1, -1]
    ])
    return cv.filter2D(img, cv.CV_8U, kernel)


def sumImages(images):
    res = images[0]
    for i in range(1, len(images)):
        res = addTwoImages(res, images[i])
    return res


def addTwoImages(img1, img2, alpha=0.5):
    beta = 1.0 - alpha
    return cv.addWeighted(img1, alpha, img2, beta, 0)


def adaptiveBinarization(img):
    adapt_thresholds = [
        adaptiveThreshold(
            img,
            method="MEAN",
            bsize=9,
            c=i)
        for i in range(0, 4)]

    approx_res = sumImages(adapt_thresholds)

    approx_blured = applyBlur(
        approx_res,
        method="GAUSSIAN",
        kernsize=5,
        sigma=3.3)

    res_threshold = otsuThreshold(approx_blured)
    return res_threshold


# Make a list of images stored in given folder
imgs = openImagesInFolder("./Fingerprints/")


# Files for sotring pathes to
# good and bad examples of fingerprints
good_imgs = open("good.txt", 'w')
bad_imgs = open("bad.txt", 'w')

# Loop this images
for img_t in imgs:

    img = img_t[0]

    # Show original image to comapare transitions
    showImage(img, winname="Original")

    # Convert image to grayscale
    grayed = imgToGray(img)

    # Blur it to remove some noise
    blured = applyBlur(grayed, kernsize=5, sigma=3.4)

    # Binarize image
    binarized = adaptiveBinarization(blured)
    showImage(binarized, winname="Binarized")

    # Wait for any key before switch to another image
    k_code = getKey()

    # G key was pressed:
    if k_code == 103:
        try:
            good_imgs.write(img_t[1] + "\n")
        except Exception:
            print "Error while writing GOOD image path"
            continue
    # B key was pressed:
    elif k_code == 98:
        try:
            bad_imgs.write(img_t[1] + "\n")
        except Exception:
            print "Erorr while writing BAD image path"
    # ESC was pressed:
    elif k_code == 27:
        break
    else:
        continue
