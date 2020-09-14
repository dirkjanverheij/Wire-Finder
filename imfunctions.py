import numpy as np
import bisect
import cv2

def myfunction(vd):
    return np.int(vd)

def imadjust(src, tol=1, vin=[0,255], vout=(0,255)):
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img

    assert len(src.shape) == 2 ,'Input image should be 2-dims'

    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.histogram(src,bins=list(range(256)),range=(0,255))[0]

        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, 255): cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs = src-vin[0]
    vs[src<vin[0]]=0
    vd = vs*scale+0.5 + vout[0]
    myfunction2 = np.vectorize(myfunction)
    vd = myfunction2(vd)
    vd[vd>vout[1]] = vout[1]
    dst = vd

    return dst.astype(np.uint8)

def imsharpen(img):
    blur = cv2.GaussianBlur(img,(5,5),0)
    img_sharp = cv2.addWeighted(img,1.5,blur,-0.5,0)

    return img_sharp

def areafilt(img, min_val,max_val=1e8):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros(img.shape[:2], dtype=img.dtype)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_val < area < max_val:
            cv2.drawContours(output, [cnt], 0, (255), -1)
    return output

def imfill(img):
    img_floodfill = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h+2,w+2),np.uint8)
    cv2.floodFill(img_floodfill, mask, (0,0), 255)
    img_floodfill_inv = cv2.bitwise_not(img_floodfill)
    output = img | img_floodfill_inv
    return output

def MAfilt(img,min_val,max_val):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros(img.shape[:2], dtype=img.dtype)
    for cnt in contours:
        if len(cnt) > 4:
            _, (ma, MA), _ = cv2.fitEllipse(cnt)
            if MA > min_val and MA < max_val:
                cv2.drawContours(output, [cnt], 0, (255), -1)
    return output

def mafilt(img,min_val,max_val):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros(img.shape[:2], dtype=img.dtype)
    for cnt in contours:
        if len(cnt) > 4:
            _, (ma, MA), _ = cv2.fitEllipse(cnt)
            if ma > min_val and ma < max_val:
                cv2.drawContours(output, [cnt], 0, (255), -1)
    return output


def perimeterfilt(img, min_val, max_val):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros(img.shape[:2], dtype=img.dtype)
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        if min_val < perimeter < max_val:
            cv2.drawContours(output, [cnt], 0, (255), -1)
    return output


def imrotate(img,angle):
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    output = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return output

def eccentricityfilt(img,min_val,max_val = 1):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros(img.shape[:2], dtype=img.dtype)
    for cnt in contours:
        if len(cnt) > 4:
            _, (ma, MA), _ = cv2.fitEllipse(cnt)
            eccentricity = np.sqrt(1-ma**2/MA**2)
            if eccentricity > min_val and eccentricity < max_val:
                cv2.drawContours(output, [cnt], 0, (255), -1)
    return output

def imcrop(img, x, y, w, h):
    output = img[y:y+h, x:x+w]
    return output

def clear_border(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output = img.copy()
    imgRows = img.shape[0]
    imgCols = img.shape[1]
    radius = 20
    contourList = []

    for cnt in contours:
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]

            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows - 1 - radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols - 1 - radius and colCnt < imgCols)

            if check1 == 1 and check2 == 1:
                cv2.drawContours(output, [cnt], 0, (0,0,0), -1)

    return output


