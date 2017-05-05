from PIL import Image
import sys
import math
import copy
import progressbar as pb
# import os

# A dictionary which contains info about a handling image
img_data = {
    "mode": "",
    "size": "",
}

# Number of color channels works with
col_channels = 3

# Coefficients to make a pixel gray
gray_coef = (0.299, 0.587, 0.114)


def getPMap():
    """ Get linear list of image pixels

    Method opens an image and returns a list of
    pixel tuples. Path to image extracts from 1st console
    argument.

    Returns:
        List of tupples, where every tuple reperesnts
        a pixel of image. List length is H x W of image.pixel
        Example of return [(233, 100, 12), (203, 203, 2), ...]
    """
    try:
        img = Image.open(sys.argv[1])
        img_data["mode"] = img.mode
        img_data["size"] = img.size
    except:
        print "Error opening image!\n"
        return False
    pixel_map = list(img.getdata())
    return pixel_map


# Method to make an image from linear list of pixel tuples
def ImageFromPixMap(pix_map):
    new_img = Image.new(img_data["mode"], img_data["size"])
    new_img.putdata(pix_map)
    return new_img


# Method which saves an image represented as linear list of pixel tuples
def saveImageFromMap(pix_map, img_name):
    img = ImageFromPixMap(pix_map)
    img.save(img_name)


def showImageFromMap(pix_map):
    new_img = ImageFromPixMap(pix_map)
    new_img.show()


def grayscalePixMap(pix_map):
    gray_pix_map = []
    for pixel in pix_map:
        gray_pix_map.append(getGrayPixel(pixel))
    return gray_pix_map


def getGrayPixel(pixel_tuple, method="average"):
    pix_val = 0

    if method == "average":
        pix_val = int((pixel_tuple[0] + pixel_tuple[1] + pixel_tuple[2]) / 3.0)

    if method == "luminosity":
        pix_val = int(pixel_tuple[0] * gray_coef[0] + pixel_tuple[1] * gray_coef[1] + pixel_tuple[2] * gray_coef[2])

    if method == "lightness":
        pix_val = int((max(pixel_tuple) + min(pixel_tuple)) / 2)

    return (pix_val, pix_val, pix_val)


def listToLofL(in_list):
    res_list = [in_list[i * img_data["size"][0]:i * img_data["size"]
                        [0] + img_data["size"][0]] for i in range(img_data["size"][1])]
    return res_list


def LofLtolist(lofl):
    newlist = []
    for inlist in lofl:
        newlist.extend(inlist)
    return newlist


def Gaussian(x, y, om):
    f1 = float(1.0 / (2.0 * math.pi * math.pow(om, 2)))
    xy = (math.pow(x, 2) + math.pow(y, 2)) * (-1.0)
    f2 = math.exp(float(xy) / float(2.0 * math.pow(om, 2)))
    return float(f1 * f2)


def GaussKernelGenerator(kern_size, omega):
    kernel = [
        [0 for col in range(kern_size[0])] for row in range(kern_size[1])
    ]

    kr = len(kernel)
    kc = len(kernel[0])
    hkr = int(kr / 2)
    hkc = int(kc / 2)

    for r in range(hkr + 1):
        for c in range(hkc + 1):
            kernel[r][c] = kernel[kr - r - 1][c] = kernel[r][kc - c - 1] = kernel[kr - r - 1][kc - c - 1] = Gaussian(r, c, omega)

    return kernel


def gaussianBlur(pix_map, kern_size, omega):
    kernel = GaussKernelGenerator(kern_size, omega)
    lofl_in = listToLofL(pix_map)
    lofl_out = list(lofl_in)

    rows = len(lofl_in)
    cols = len(lofl_in[0])

    hkr = int(len(kernel) / 2)
    hkc = int(len(kernel[0]) / 2)

    for row in range(hkr, rows - hkr):
        print "Row = %d" % row
        for col in range(hkc, cols - hkc):
            surr_pixels = [
                [
                    lofl_in[r][c] for c in range(-1 * hkc + col, hkc + col + 1)
                ] for r in range(-1 * hkr + row, hkr + row + 1)
            ]
            pix_sums = [0, 0, 0]

            for r in range(len(surr_pixels)):
                for c in range(len(surr_pixels[0])):
                    for ch_val_ind in range(3):
                        pix_sums[ch_val_ind] += float(surr_pixels[r][c][ch_val_ind]) * kernel[r][c]
            pix_val = [int(pix_sums[chn]) for chn in range(3)]
            pix_val = tuple(pix_val)

            lofl_out[row][col] = pix_val
    return lofl_out


def MedianFilter(pix_map, aperture_size, color_channel):
    sq_map = listToLofL(pix_map)
    sq_map_out = list(sq_map)

    ah = aperture_size[0]
    aw = aperture_size[1]
    ah2 = int(ah / 2.0)
    aw2 = int(aw / 2.0)

    central = int((ah * aw) / 2.0) + 1

    surr_pixels = [[0 for c in range(aw)] for r in range(ah)]

    iter_counter = 0
    iter_total = (len(sq_map) - ah2 * 2) * (len(sq_map[0]) - aw2 * 2)

    for row in range(ah2, len(sq_map_out) - ah2):
        for col in range(aw2, len(sq_map_out[0]) - aw2):
            surr_pixels = [
                [
                    sq_map_out[r][c] for c in range(-1 * aw2 + col, aw2 + col + 1)
                ] for r in range(-1 * ah2 + row, ah2 + row + 1)
            ]
            surr_pixels_linear = LofLtolist(surr_pixels)

            pixels_r = [pix[0] for pix in surr_pixels_linear]
            pixels_g = [pix[1] for pix in surr_pixels_linear]
            pixels_b = [pix[2] for pix in surr_pixels_linear]

            pixels_r.sort()
            pixels_g.sort()
            pixels_b.sort()

            # surr_pixels_linear.sort(key=lambda pixel: pixel[color_channel])

            iter_counter += 1

            print "\r\bMedian Filter: %.2f percents completed" % (iter_counter * 100 / iter_total)
            sq_map_out[row][col] = tuple([pixels_r[central], pixels_g[central], pixels_b[central]])
            # sq_map_out[row][col] = surr_pixels_linear[central]

    pix_map_out = LofLtolist(sq_map_out)
    return pix_map_out


def ApplyMatrixFilter(lin_pmap, matrix):
    sq_map = listToLofL(lin_pmap)
    sq_map_out = copy.copy(sq_map)

    imh = len(sq_map)
    imw = len(sq_map[0])

    mah = len(matrix)
    maw = len(matrix[0])

    hmah = int(mah / 2.0)
    hmaw = int(maw / 2.0)

    i_counter = 0
    i_total = (imh - mah) * (imw - maw)

    bar = pb.ProgressBar().start()

    for row in range(hmah, imh - hmah):
        for col in range(hmaw, imw - hmaw):
            pix_matrix = [
                [sq_map_out[r][c] for c in range(-1 * hmaw + col, hmaw + col + 1)] for r in range(-1 * hmah + row, hmah + row + 1)
            ]
            # Some actions with matrix and result
            sq_map_out[row][col] = getConvolution(pix_matrix, matrix)

            i_counter += 1
            bar.update(int((i_counter * 100.0) / i_total))

    bar.finish()
    return sq_map_out


def showGaussBlur():
    in_img = getPMap()
    gMatrix = GaussKernelGenerator((5, 5), 1)
    gra_img = grayscalePixMap(in_img)
    gResult = ApplyMatrixFilter(gra_img, gMatrix)
    lin_gResult = LofLtolist(gResult)
    showImageFromMap(lin_gResult)


def getConvolution(in_matr, conv_matr, div=1.0):
    # res = 0.0
    red, green, blue = 0, 0, 0

    for r in range(len(in_matr)):
        for c in range(len(in_matr[0])):
            red += (in_matr[r][c][0] * conv_matr[r][c] * (1.0 / div))
            green += (in_matr[r][c][1] * conv_matr[r][c] * (1.0 / div))
            blue += (in_matr[r][c][2] * conv_matr[r][c] * (1.0 / div))
    return tuple([int(red), int(green), int(blue)])


# Bradley - Roth algorythm implementation
def Threshold(pix_map, koef=0.15):
    h = img_data["size"][1]
    w = img_data["size"][0]
    s = int(w / 8)
    s2 = s / 2
    t = float(koef)
    integral_image = [0 for x in range(w * h)]
    summa = 0
    count = 0
    index = 0
    x1, y1, x2, y2 = 0, 0, 0, 0

    res_pix_map = [0 for x in range(w * h)]

    for i in range(w):
        summa = 0
        for j in range(h):
            index = j * w + i
            summa += pix_map[index][0]
            if i == 0:
                integral_image[index] = summa
            else:
                integral_image[index] = integral_image[index - 1] + summa

    for i in range(w):
        for j in range(h):
            index = j * w + i
            x1 = i - s2
            x2 = i + s2
            y1 = j - s2
            y2 = j + s2

            if x1 < 0:
                x1 = 0
            if x2 >= w:
                x2 = w - 1
            if y1 < 0:
                y1 = 0
            if y2 >= h:
                y2 = h - 1

            count = (x2 - x1) * (y2 - y1)

            summa = (
                integral_image[y2 * w + x2] -
                integral_image[y1 * w + x2] -
                integral_image[y2 * w + x1] +
                integral_image[y1 * w + x1]
            )

            if (pix_map[index][0] * count < summa * (1.0 - t)):
                res_pix_map[index] = (0, 0, 0)
            else:
                res_pix_map[index] = (255, 255, 255)

    return res_pix_map

# showImageFromMap(Threshold(grayscalePixMap(getPMap()), koef=0.15))


showGaussBlur()