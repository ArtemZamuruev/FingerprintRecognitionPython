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


def showImage(img, winname="Fingerprints", trackbar=None):
    cv.imshow(winname, img)
    if trackbar is not None:
        cv.createTrackbar(trackbar["Name"],
                          winname,
                          trackbar["Value"],
                          trackbar["Count"],
                          trackbar["Update"])


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


def applyMedianFilter(img, kernsize=5):
    return cv.medianBlur(img, kernsize)


def otsuThreshold(img):
    ret, thr = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
    return thr


def binaryThreshold(img, thresh=100):
    return cv.threshold(img, thresh, 255, cv.THRESH_BINARY)[1]


def adaptiveThreshold_1(img):
    thr = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    return thr


def adaptiveThreshold_2(img):
    return cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)


def sobelFilter(img):
    return (cv.Sobel(img, cv.CV_8U, 1, 0), cv.Sobel(img, cv.CV_8U, 0, 1))

def addTwoImages(img1, img2, alpha):
    beta = float(1 - alpha)
    return cv.addWeighted(img1, alpha, img2, beta, 0)


def equalHist(img):
    return cv.equalizeHist(img)


def sharpenImage(img, kcenter=9):
    kernel = np.array([
        [-1, -1, -1],
        [-1, kcenter, -1],
        [-1, -1, -1]
    ])
    return cv.filter2D(img, cv.CV_8U, kernel)


def smoothUpdate(x):
    global grayed
    track_position = int(
        cv.getTrackbarPos(
            "Kernel Size",
            "Smoothed"
        )
    )

    if track_position < 0:
        return

    newKernelSize = track_position * 2 + 1
    newSmoothedImg = applyMedianFilter(grayed, newKernelSize)
    showImage(newSmoothedImg, winname="Smoothed")
    newBinarized = adaptiveThreshold_1(newSmoothedImg)
    showImage(newBinarized, winname="Adaptive 1")
    newBinarized_2 = adaptiveThreshold_2(newSmoothedImg)
    showImage(newBinarized_2, winname="Adaptive 2")

def thresholdUpdate(x):
    global hist
    track_position = int(
        cv.getTrackbarPos(
            "Threshold",
            "Threshold Binarized"
        )
    )

    if track_position < 0:
        return

    newBinarizedThreshold = binaryThreshold(
        hist,
        thresh=track_position
    )
    showImage(newBinarizedThreshold, winname="Threshold Binarized")


# Make a list of images stored in given folder
imgs = openImagesInFolder("./Fingerprints/")

# Loop this images
for img in imgs:

    # Show original image to comapare transitions
    showImage(img, winname="Original")

    # Convert image to grayscale
    grayed = imgToGray(img)

    # Use histogram equalization:
    # hist = equalHist(grayed)
    # showImage(hist, winname="Histogram")

    # sharpened = sharpenImage(grayed)
    # showImage(sharpened, winname="Sharpened")

    # Apply sobel filter:
    # sobeled = sobelFilter(hist)
    # showImage(sobeled[0], winname="Sobeled X")
    # showImage(sobeled[1], winname="Sobeled Y")
    # showImage(addTwoImages(sobeled[0], sobeled[1], 0.5), winname="Sobeled Together")

    # Apply median filter to smooth the image
    smoothed = applyMedianFilter(grayed, kernsize=3)
    smoothTrackbar = {
        "Name": "Kernel Size",
        "Value": 1,
        "Count": 5,
        "Update": smoothUpdate
    }
    showImage(smoothed, winname="Smoothed", trackbar=smoothTrackbar)

    # Binarization
    # binarized_otsu = otsuThreshold(hist)
    # showImage(binarized_otsu, winname="OTSU Binarized")

    thr_1 = adaptiveThreshold_1(smoothed)
    showImage(thr_1, winname="Adaptive 1")

    thr_2 = adaptiveThreshold_2(smoothed)
    showImage(thr_2, winname="Adaptive 2")


    # sobeled2 = addTwoImages(sobeled[0], sobeled[1], 0.5)
    # binotsu2 = otsuThreshold(sobeled2)
    # showImage(binotsu2, winname="Fffff")


    # canny = cv.Canny(grayed, 100, 200)
    # showImage(canny, winname="Canny")

    # binarized_threshold = binaryThreshold(hist, thresh=100)
    # thresholdTrackbar = {
    #     "Name": "Threshold",
    #     "Value": 100,
    #     "Count": 255,
    #     "Update": thresholdUpdate
    # }
    # showImage(binarized_threshold,
    #           winname="Threshold Binarized",
    #           trackbar=thresholdTrackbar)

    # Wait for any key before switch to another image
    getKey()