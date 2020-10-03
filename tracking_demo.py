import numpy as np
import cv2
import imutils
from main import ShapeUtils, CentroidTracker
from collections import OrderedDict
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
utils = ShapeUtils()
ct = CentroidTracker()

TEXT_COLOR = (0, 0, 255)
TEXT_SIZE = 1
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
kernel = np.ones((5, 5), np.uint8)
LENGTH_CONST = 0.01
AREA_CONST = 0.0001
FRAME_HEIGHT = 470
FRAME_WIDTH = 630
#kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
count = 0
c_arr = []
while True:
    _, frame = cap.read()  # grab frames from the video feed
    frame = frame[:400, 100:]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("frame..", frame)
    blur_frame = cv2.bilateralFilter(frame, 11, 200, 200)
    cv2.imshow("blurred_frame..", blur_frame)
    # convert to gray scale
    gray = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    cv2.imshow("thresh", thresh)
    #closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    kernelmg = np.ones((3, 3), np.uint8)
    mg = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel, iterations=1)
    mg_erode = cv2.erode(mg, kernelmg)
    # dist = cv2.distanceTransform(opening, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    # _, dist_t = cv2.threshold(dist, 0.0999999, 1.0, cv2.THRESH_BINARY)
    # kernel1 = np.ones((7, 7), dtype=np.uint8)
    # dia_dist = cv2.dilate(dist_t, kernel1)
    #cv2.imshow("closing", closing)
    cv2.imshow("opening", opening)
    cv2.imshow("mg", mg)
    cv2.imshow("mg_erode", mg_erode)
    contours = cv2.findContours(
        mg_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)  # grab contours
    # print("sa", cv2.contourArea(contours[0]))
    cv2.line(frame, (300, 0), (300, 639), (0, 0, 255), 2)
    print("MAIN", len(contours))
    centroids = utils.ret_centroids(contours)
    print(centroids)
    objects = ct.update(centroids)
    print("objs", len(objects))
    ids = objects.keys()
    for i in ids:
        c = objects[i]
        x_, y_ = c[0], c[1]
        if utils.check_linecross(300, x_) and i not in c_arr:
            count += 1
            c_arr.append(i)
        cv2.putText(frame, "ID : " + str(i), (x_, y_), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 3, cv2.LINE_AA, False)
    print(count)
    cv2.imshow("video", frame)
    cv2.imshow("thresh", thresh)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
