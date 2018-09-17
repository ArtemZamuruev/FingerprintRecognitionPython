from os.path import join, isfile
from os import listdir
import os
import cv2 as cv
import numpy as np
from time import time
import math


def openImagesInFolder(folder):
    print("Reading Images")
    img_paths = [join(folder, fname)
                 for fname in listdir(folder)
                 if isfile(join(folder, fname))]
    try:
        return [(cv.imread(img_path), img_path) for img_path in img_paths]
    except Exception:
        print("Something went wrong during opening images in folder: %s" % folder)
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
        print("Error writing image. Path: %s" % nf_path)
        return False


def imgToGray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def getBlockVariation(block):
    summa = 0
    mean = float(np.sum(block)) / float(block.size)
    for r in range(block.shape[0]):
        for c in range(block.shape[1]):
            summa += ((float(block[r, c]) - mean) ** 2)
    return float(summa) / float(block.size)


def segmentation(img):
    result_image = np.copy(img)
    # Calculate global grayscale variation
    global_var = getBlockVariation(img)

    # Calculate variation of every (10, 10) block
    bsize = 10
    for r in range(0, img.shape[0] - bsize, bsize):
        for c in range(0, img.shape[1] - bsize, bsize):
            block = img[r: r + bsize, c: c + bsize]
            local_var = getBlockVariation(block)
            # if float(local_var) / float(global_var) < 0.6:
            if local_var < global_var:
                result_image[r: r + bsize, c: c + bsize] = np.zeros(block.shape, np.uint8)
    return result_image


def coherence_filter(img, sigma=15, str_sigma=15, blend=0.7, iter_n=5):
    h, w = img.shape[:2]

    img = np.float32(img)

    for i in range(iter_n):
        gray = img
        eigen = cv.cornerEigenValsAndVecs(gray, str_sigma, 3)
        eigen = eigen.reshape(h, w, 3, 2)  # [[e1, e2], v1, v2]
        x, y = eigen[:, :, 1, 0], eigen[:, :, 1, 1]

        gxx = cv.Sobel(gray, cv.CV_32F, 2, 0, ksize=sigma)
        gxy = cv.Sobel(gray, cv.CV_32F, 1, 1, ksize=sigma)
        gyy = cv.Sobel(gray, cv.CV_32F, 0, 2, ksize=sigma)
        gvv = x * x * gxx + 2 * x * y * gxy + y * y * gyy
        m = gvv < 0

        ero = cv.erode(img, None)
        dil = cv.dilate(img, None)
        img1 = ero
        img1[m] = dil[m]
        img = np.uint8(img * (1.0 - blend) + img1 * blend)

    return img


def morphoSegmentation(img):
    print("Stage:\tMorphological Segmentation")

    coherenced = coherence_filter(img)

    blured = cv.GaussianBlur(coherenced, (5, 5), 5)
    blured = cv.fastNlMeansDenoising(blured, h=13)
    binary = cv.threshold(blured, 0, 255, cv.THRESH_OTSU)[1]
    binary = cv.bitwise_not(binary)

    close_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        (9, 9))
    dilate_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        (5, 5))
    erosion_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        (3, 3))

    morphology = cv.morphologyEx(
        binary,
        cv.MORPH_DILATE, dilate_kernel,
        iterations=4)
    morphology = cv.morphologyEx(
        morphology,
        cv.MORPH_CLOSE,
        close_kernel,
        iterations=1)
    morphology = cv.morphologyEx(
        morphology,
        cv.MORPH_ERODE,
        erosion_kernel,
        iterations=3)

    segmask = np.bitwise_not(morphology)

    segmented = np.copy(img)

    rc, cc = img.shape
    for r in range(rc):
        for c in range(cc):
            if segmask[r, c] == 255:
                segmented[r, c] = 255

    print("\t...done")
    return segmented


def calcNormPixel(pix, gmean, gvar, dmean, dvar):
    v_1 = math.sqrt(((dvar * (pix - gmean) ** 2)) / float(gvar))
    if pix > gmean:
        return float(dmean + float(v_1))
    else:
        return float(dmean - float(v_1))


def normalization(img, d_mean=100, d_var=255):
    print("Stage:\tNormaliztion")
    g_var = getBlockVariation(img)
    g_mean = np.sum(img) / img.size

    new_img = np.copy(img)

    rc, cc = img.shape

    for r in range(rc):
        for c in range(cc):
            new_img[r, c] = calcNormPixel(
                img[r, c],
                g_mean,
                g_var,
                d_mean,
                d_var)
    print("\t...done")
    return new_img


def computeOrientationAngle(gr_x, gr_y):
    brx, bcx = gr_x.shape

    g_xx = 0.0
    g_xy = 0.0
    g_yy = 0.0

    for u in range(brx):
        for v in range(bcx):

            g_xy += float(gr_x[u, v]) * float(gr_y[u, v])
            g_xx += float(gr_x[u, v]) * float(gr_x[u, v])
            g_yy += float(gr_y[u, v]) * float(gr_y[u, v])

    ksi = float(np.arctan2(float(2.0 * g_xy), float(g_xx - g_yy))) / 2.0
    # ksi += np.pi / 2.0

    return ksi


def orientionalComputing(img, b_shape=(15, 15), step=None):

    print("Stage:\tComputing ridge angles")

    if step is None:
        step = max(b_shape)

    # img = cv.resize(img, (img.shape[1]*2, img.shape[0] * 2), interpolation=cv.INTER_CUBIC)
    # img = np.bitwise_not(img)
    img64 = np.float64(img)

    sobel_x = cv.Sobel(img64, cv.CV_64F, 1, 0)
    sobel_y = cv.Sobel(img64, cv.CV_64F, 0, 1)

    orient_mask = np.zeros(img64.shape, np.float64)

    br, bc = b_shape
    rc, cc = img64.shape

    for i in range(br // 2, rc - br // 2, step):
        for j in range(bc // 2, cc - bc // 2, step):

            row_r = (i - br // 2, i + br // 2 + 1)
            col_r = (j - bc // 2, j + bc // 2 + 1)

            block_x = sobel_x[row_r[0]:row_r[1], col_r[0]:col_r[1]]
            block_y = sobel_y[row_r[0]:row_r[1], col_r[0]:col_r[1]]

            orient_mask[i, j] = computeOrientationAngle(block_x, block_y)

    # new_img = np.zeros(img64.shape)
    # new_img = np.copy(img)
    # new_img = cv.cvtColor(new_img, cv.COLOR_GRAY2BGR)

    # for r in range(rc):
    #     for c in range(cc):
    #         if orient_mask[r, c] != 0:
    #             drawLineFromPoint(new_img, (c, r), orient_mask[r, c], 4, two_dir=True)

    # showImage(np.concatenate((sobel_x, sobel_y, orient_mask, new_img), axis=1), winname="Sobels")

    print("\t...done")
    return orient_mask


def drawLineFromPoint(img, c_point, angle_rad, length, color=255, two_dir=False):
    c_x, c_y = c_point

    x2_f = c_x + length * math.cos(angle_rad + np.pi / 2)
    y2_f = c_y + length * math.sin(angle_rad + np.pi / 2)
    x2 = int(round(x2_f))
    y2 = int(round(y2_f))
    cv.line(img, (c_x, c_y), (x2, y2), color)

    if two_dir:
        x3 = c_x + (-1) * (x2 - c_x)
        y3 = c_y + (-1) * (y2 - c_y)
        cv.line(img, (c_x, c_y), (x3, y3), color)
#    return img


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

    marker_divided = np.bitwise_not(marker) / 255

    img = img.astype(np.uint8)
    marker_divided = marker_divided.astype(np.uint8)

    # print(type(img))
    # print(img.dtype)
    # print(type(marker_divided))
    # print(marker_divided.dtype)


    mul = np.bitwise_and(img, marker_divided)

    return mul


def shockAndBinarization(img):
    print("Stage:\tBinarization")
    bin_1 = cv.threshold(img, 0, 255, cv.THRESH_OTSU)[1]
    coh_filtered = coherence_filter(
        bin_1,
        sigma=13,
        str_sigma=13,
        blend=0.7,
        iter_n=10)
    bin_2 = cv.threshold(coh_filtered, 0, 255, cv.THRESH_OTSU)[1]
    bin_res = np.bitwise_not(bin_2)
    print("\t...done")
    return bin_res


def thinningGuoHall(imgIn):
    print("Stage:\tGuo-Hall lines thinning")
    img = imgIn / 255

    prev = np.zeros(img.shape, np.uint8)

    done = False

    iter_counter = 1
    while not done:
        print("\t...iteration %d" % iter_counter)
        img = thinningGuoHallIteration(img, 0)
        img = thinningGuoHallIteration(img, 1)
        diff = cv.absdiff(img, prev)
        prev = np.copy(img)

        if np.count_nonzero(diff) <= 0:
            done = True
        iter_counter += 1

    print("\t...done")
    return img * 255


def morphologicalSkeleton(img):
    skel = np.zeros(img.shape, np.uint8)
    temp = np.zeros(img.shape, np.uint8)

    kern = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

    done = False

    while not done:
        temp = cv.morphologyEx(img, cv.MORPH_OPEN, kern)
        temp = cv.bitwise_not(temp)
        temp = cv.bitwise_and(img, temp)
        skel = cv.bitwise_or(temp, skel)
        img = cv.erode(img, kern)

        if np.count_nonzero(img) == 0:
            done = True

    return skel


def calcCrossNumber(n_hood):
    s = 0
    for i in range(0, len(n_hood) - 1):
        s += abs(int(n_hood[i + 1]) - int(n_hood[i]))
    return s


def minutaeDetection(img, postprocessing=True):
    print("Stage:\tMinutaes Detection")
    minutaes_figure = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    minutaes_figure = np.float64(minutaes_figure)
    img = img.astype(np.float64)
    img /= 255
    minutaeMap = np.zeros(img.shape, np.uint16)

    rc, cc = img.shape
    for r in range(1, rc - 1):
        for c in range(1, cc - 1):

            if img[r, c] == 0:
                continue

            nbhood = (
                img[r, c - 1],
                img[r - 1, c - 1],
                img[r - 1, c],
                img[r - 1, c + 1],
                img[r, c + 1],
                img[r + 1, c + 1],
                img[r + 1, c],
                img[r + 1, c - 1],
                img[r, c - 1]
            )

            cn = calcCrossNumber(nbhood)

            # Bifurcation point
            if cn == 6:
                minutaeMap[r, c] = cn

            # Termination point
            if cn == 2:
                minutaeMap[r, c] = cn


    if not postprocessing:
        print("\t...Minutiaes found: %d" % np.count_nonzero(minutaeMap))
        print("\t...done")
        return minutaeMap

    # Minutiaes postprocessing
    false_terminations = np.zeros(minutaeMap.shape, np.int16)

    ns = 20
    rc, cc = minutaeMap.shape

    for r in range(ns, rc - ns):
        for c in range(ns, cc - ns):
            if minutaeMap[r, c] == 2:
                nh_block = minutaeMap[r - ns: r + ns, c - ns: c + ns]
                count_twos = 0
                br_c, bc_c = nh_block.shape
                for br in range(br_c):
                    for bc in range(bc_c):
                        if nh_block[br, bc] == 2:
                            count_twos += 1
                if count_twos < 4:
                    false_terminations[r, c] = -2

    minutaeMap = minutaeMap + false_terminations


    print("\t...Minutiaes found: %d" % np.count_nonzero(minutaeMap))
    print("\t...done")
    return minutaeMap


def drawMinutiaesOnImage(img, m_map, a_map):
    print("Stage:\tDrawing minutiaes")
    figure_img = np.copy(img)
    figure_img = np.float32(img)
    figure_img = cv.cvtColor(figure_img, cv.COLOR_GRAY2BGR)
    rc, cc = m_map.shape
    for r in range(rc):
        for c in range(cc):
            if m_map[r, c] == 2:
                # Draw termination point
                angle = a_map[r, c]
                cv.circle(figure_img, (c, r), 4, (0, 255, 0))
                drawLineFromPoint(figure_img, (c, r), angle, 7, color=(0, 0, 255), two_dir=True)
            if m_map[r, c] == 6:
                # Draw bifurcation point
                angle = a_map[r, c]
                cv.circle(figure_img, (c, r), 4, (0, 0, 255))
                drawLineFromPoint(figure_img, (c, r), angle, 7, color=(0, 255, 0), two_dir=True)
    print("\t...done")
    return figure_img


def calculateHoughTransform(m1, m2, fi_range, fi_tol, fi_err, dx_err, dy_err):
    transforms = []

    x_1, y_1, fi_1 = m1
    x_2, y_2, fi_2 = m2

    dx_err = int(dx_err)
    dy_err = int(dy_err)

    # Each value of rotation angle in range
    for dfi in fi_range:
        # Compute direction differnce
        abs_dd = float(abs((fi_2 + dfi) - fi_1))
        dd = min(abs_dd, 2.0 * np.pi - abs_dd)

        if dd < fi_tol:
            dx = x_1 - (math.cos(dfi) * x_2 - y_2 * math.sin(dfi))
            dy = y_1 - (math.sin(dfi) * x_2 + y_2 * math.cos(dfi))

            dx = int(round(dx))
            dy = int(round(dy))

            dfi_d = int(round(math.degrees(dfi)))

            tr = (dx, dy, dfi_d)
            transforms.append(tr)

            # for e_x in range(dx - dx_err, dx + dx_err + 1):
            #     for e_y in range(dy - dy_err, dy + dy_err + 1):
            #         tr = (e_x, e_y, dfi_d)
            #         transforms.append(tr)

            # for e_fi in range(int(math.degrees(dfi - fi_err)), int(math.degrees(dfi + fi_err))):
            #     for e_x in range(dx - dx_err, dx + dx_err + 1):
            #         for e_y in range(dy - dy_err, dy + dy_err + 1):
            #             tr = (e_x, e_y, e_fi)
            #             transforms.append(tr)
    return transforms


def minutiaesMatching(m_set1, ms_set_2, fi_range=0, fi_tol=10, fi_err=0, dx_err=0, dy_err=0):
    # Fi_range, fi_err, fi_tol sets in degrees
    fi_range = [float(math.radians(f)) for f in fi_range]
    fi_tol = float(math.radians(float(fi_tol)))
    fi_err = float(math.radians(float(fi_err)))

    accumulator = {}

    print("Stage:\tMinutiaes matching")
    print("\tSubstage: registration")

    pair_counter = 0

    for m1 in m_set1:
        for m2 in ms_set_2:
            pair_counter += 1
            # print("\t\t_Pair #%d" % pair_counter
            rates = calculateHoughTransform(m1, m2, fi_range, fi_tol, fi_err, dx_err, dy_err)
            for rate in rates:
                rate_key = str(rate)
                if rate_key in accumulator.keys():
                    accumulator[rate_key] += 1
                else:
                    accumulator[rate_key] = 1

    ac_sorted = sorted(accumulator.items(), key=lambda x: x[1], reverse=True)

    correct_transform = ac_sorted[0][0]

    print("\tCorrect transform:")
    print(correct_transform)
    print("\t...done")


def getMinutiesDescriptors(minutiaes_map, angles_map):
    minutiae_set = []
    rc, cc = minutiaes_map.shape
    for r in range(rc):
        for c in range(cc):
            if minutiaes_map[r, c] != 0:
                minutae = (c, r, angles_map[r, c])
                minutiae_set.append(minutae)
    return minutiae_set


def filterImage(img):
    segmented = morphoSegmentation(img)
    normalized = normalization(segmented, d_mean=100, d_var=100)
    binaryzed = shockAndBinarization(normalized)
    thinned_gh = thinningGuoHall(binaryzed)
    return thinned_gh


def compareTwoFingerprints(img_1, img_2):

    img_1 = imgToGray(img_1)
    img_2 = imgToGray(img_2)

    originals = np.concatenate((img_1, img_2), axis=1)
    showImage(originals, winname="Fingerprint(images")

    print("\n")
    print("=" * 20)
    print("Image 1:")
    filtered_1 = filterImage(img_1)
    min_map_1 = minutaeDetection(filtered_1)
    angle_map_1 = orientionalComputing(filtered_1, step=1)
    min_set_1 = getMinutiesDescriptors(min_map_1, angle_map_1)
    # Extras: drawing
    min_fig_1 = drawMinutiaesOnImage(img_1, min_map_1, angle_map_1)

    print("\n")
    print("Image 2:")
    filtered_2 = filterImage(img_2)
    min_map_2 = minutaeDetection(filtered_2)
    angle_map_2 = orientionalComputing(min_map_2, step=1)
    min_set_2 = getMinutiesDescriptors(min_map_2, angle_map_2)
    # Extras: drawing
    min_fig_2 = drawMinutiaesOnImage(img_2, min_map_2, angle_map_2)

    print("\n")
    print("Comparing two images:")
    minutiaesMatching(min_set_1, min_set_2, fi_range=range(-2, 3), fi_tol=10, fi_err=1, dx_err=1, dy_err=1)

    min_total_figure = np.concatenate((min_fig_1, min_fig_2), axis=1)
    showImage(min_total_figure, winname="Minutaes drawed on fingerprints")

    print("=" * 20)
    print("\n")


def main():
    # Make a list of images stored in given folder
    imgs = openImagesInFolder("./Fingerprints/TEST_1/")

    # Files for sotring pathes to
    # good and bad examples of fingerprints
    # good_imgs = open("good.txt", 'w')
    # bad_imgs = open("bad.txt", 'w')

    compareTwoFingerprints(imgs[0][0], imgs[1][0])

    # Wait for any key before switch to another image
    k_code = getKey(delay=0)

    # G key was pressed:
    if k_code == 103:
        try:
            good_imgs.write(img_t[1] + "\n")
        except Exception:
            print("Error while writing GOOD image path")
            # continue
    # B key was pressed:
    # elif k_code == 98:
    #     try:
    #         bad_imgs.write(img_t[1] + "\n")
    #         os.remove(img_t[1])
    #     except Exception:
    #         print("Erorr while writing BAD image path"
    # ESC was pressed:
    elif k_code == 27:
        # break
        return
    # else:
        # continue


if __name__ == "__main__":
    main()
