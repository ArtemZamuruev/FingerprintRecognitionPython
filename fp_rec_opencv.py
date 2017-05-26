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


def sumImages(images):
    res = images[0]
    for i in range(1, len(images)):
        res = addTwoImages(res, images[i])
    return res


def addTwoImages(img1, img2, alpha=0.5):
    beta = 1.0 - alpha
    return cv.addWeighted(img1, alpha, img2, beta, 0)


def adaptiveBinarization(img, bsize=9):
    adapt_thresholds = [
        adaptiveThreshold(
            img,
            method="GAUSSIAN",
            bsize=bsize,
            c=i)
        for i in range(0, 4)]

    approx_res = sumImages(adapt_thresholds)

    approx_blured = applyBlur(
        approx_res,
        method="GAUSSIAN",
        kernsize=3,
        sigma=3.3)

    res_threshold = otsuThreshold(approx_blured)
    return res_threshold


def removeSurroundNoise(denoised, binarized):
    eroding_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    closing_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
    shape = cv.morphologyEx(denoised, cv.MORPH_ERODE, eroding_kernel, iterations=2)
    shape = cv.bitwise_not(otsuThreshold(shape))
    shape = cv.morphologyEx(shape, cv.MORPH_CLOSE, closing_kernel)
    shape = cv.bitwise_not(shape)
    return binarized - shape


def thinningGuoHallIteration(img, iter):
    marker = np.zeros(img.shape, np.uint8)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            p2 = img[i - 1, j]
            p3 = img[i - 1, j + 1]
            p4 = img[i, j + 1]
            p5 = img[i + 1, j + 1]
            p6 = img[i + 1, j]
            p7 = img[i + 1, j - 1]
            p8 = img[i, j - 1]
            p9 = img[i - 1, j - 1]

            C = 0
            C += (int(not p2) and (p3 or p4))
            C += (int(not p4) and (p5 or p6))
            C += (int(not p6) and (p7 or p8))
            C += (int(not p8) and (p9 or p2))

            N1 = (p9 or p2) + (p3 or p4) + (p5 or p6) + (p7 or p8)
            N2 = (p2 or p3) + (p4 or p5) + (p6 or p7) + (p8 or p9)

            if N1 < N2:
                N = N1
            else:
                N = N2

            m = 0
            if iter == 0:
                m = ((p6 or p7 or int(not p9)) and p8)
            else:
                m = ((p2 or p3 or int(not p5)) and p4)
            if C == 1 and (N >= 2 and N <= 3) and m == 0:
                marker[i, j] = 1

    return np.bitwise_and(img, np.bitwise_not(marker) / 255)


def thinningGuoHall(imgIn):
    img = imgIn / 255

    prev = np.zeros(img.shape, np.uint8)

    done = False
    print "\n\nThinning start:"

    iter_counter = 1
    while not done:
        print "Iteration:%d" % iter_counter
        img = thinningGuoHallIteration(img, 0)
        img = thinningGuoHallIteration(img, 1)
        diff = cv.absdiff(img, prev)
        prev = np.copy(img)

        if np.count_nonzero(diff) <= 0:
            done = True
        iter_counter += 1

    return img * 255


# Make a list of images stored in given folder
imgs = openImagesInFolder("./Fingerprints/NIST/")


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
    showImage(grayed, winname="Grayed")

    grayed = cv.resize(grayed, (1024, 1024), interpolation=cv.INTER_CUBIC)

    denoised = cv.fastNlMeansDenoising(grayed, h=7)
    showImage(denoised, winname="Denoised")

    ker = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    denoised = cv.morphologyEx(denoised, cv.MORPH_DILATE, ker)

    # Blur it to remove some noise
    blured = applyBlur(denoised, kernsize=5, sigma=3.4)
    blured = cv.fastNlMeansDenoising(blured, h=7)
    showImage(blured, winname="Blured")

    # # Binarize image
    binarized = adaptiveBinarization(blured, bsize=45)
    binarized = cv.bitwise_not(binarized)
    binarized = removeSurroundNoise(denoised, binarized)
    binarized = cv.resize(binarized, (768, 768), interpolation=cv.INTER_CUBIC)
    showImage(binarized, winname="Binarized")

    # # Image thinning
    # binarized = cv.resize(binarized, (256, 256), interpolation=cv.INTER_CUBIC)
    # thinned = thinningGuoHall(binarized)
    # thinned = cv.resize(thinned, (768, 768), interpolation=cv.INTER_CUBIC)
    # showImage(thinned, winname="Thinned")

    # Write thinned to specific folder
    # writeImg(thinned, folder="./Fingerprints/thinned/")


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
