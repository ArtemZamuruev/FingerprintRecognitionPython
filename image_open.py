from PIL import Image
import sys
import math


img_data = {
    "mode": "",
    "size": "",
}

gray_coef = (0.299, 0.587, 0.114)
def getPMap():
    try:
        img = Image.open(sys.argv[1])
        img_data["mode"] = img.mode
        img_data["size"] = img.size
    except:
        print "Error opening image!\n"

col_channels = 3


def getPMap():
    try:
        img = Image.open(sys.argv[1])
        img_data["mode"] = img.mode
        img_data["size"] = img.size
    except:
        print "Error opening image!\n"
        return False
    pixel_map = list(img.getdata())
    return pixel_map


def ImageFromPixMap(pix_map):
    new_img = Image.new(img_data["mode"], img_data["size"])
    new_img.putdata(pix_map)
    return new_img


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


def getGrayPixel(pixel_tuple, method="lightness"):
    pix_val = 0

    if method == "average":
        pix_val = int((pixel_tuple[0] + pixel_tuple[1] + pixel_tuple[2]) / 3.0)

    if method == "luminosity":
        pix_val = int(pixel_tuple[0] * gray_coef[0] + pixel_tuple[1] * gray_coef[1] + pixel_tuple[2] * gray_coef[2])

    if method == "lightness":
        pix_val = int((max(pixel_tuple) + min(pixel_tuple)) / 2)

    return (pix_val, pix_val, pix_val)


def listToLofL(in_list):
    res_list = [in_list[i * img_data["size"][0]:i * img_data["size"][0] + img_data["size"][0]] for i in range(img_data["size"][1])]
    return res_list


def LofLtolist(lofl):
    newlist = []
    for inlist in lofl:
        newlist.extend(inlist)
    return newlist


# def calculatePixValue(kernel, pixels):


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

            # surr_pixels_number = len(surr_pixels) * len(surr_pixels[0])

            for r in range(len(surr_pixels)):
                for c in range(len(surr_pixels[0])):
                    # for ch_val_ind in range(len(surr_pixels[0][0])):
                    for ch_val_ind in range(3):
                        pix_sums[ch_val_ind] += float(surr_pixels[r][c][ch_val_ind]) * kernel[r][c]
            # pix_val = [int(pix_sums[chn]) for chn in range(len(surr_pixels[0][0]))]
            pix_val = [int(pix_sums[chn]) for chn in range(3)]
            pix_val = tuple(pix_val)

            lofl_out[row][col] = pix_val
    return lofl_out


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


in_map = getPMap()
ttt = Threshold(in_map, koef=0.07)
showImageFromMap(ttt)
saveImageFromMap(ttt, "finger_original_tr.png")
lofl_map = gaussianBlur(in_map, (5, 5), 2)
list_map = LofLtolist(lofl_map)
showImageFromMap(list_map)
saveImageFromMap(list_map, "finger_blur.png")
tmpa = Threshold(list_map, koef=0.07)
showImageFromMap(tmpa)
saveImageFromMap(tmpa, "finger_tr_after_gauss.png")
