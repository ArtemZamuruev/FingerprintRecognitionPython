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


def main():
    # Make a list of images stored in given folder
    imgs = openImagesInFolder("./Fingerprints/DB1/")

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


if __name__ == "__main__":
    main()
