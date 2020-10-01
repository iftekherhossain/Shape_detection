import numpy as np
import cv2
import imutils
from main import ShapeUtils, CentroidTracker
from collections import OrderedDict

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


while True:
    _, frame = cap.read()  # grab frames from the video feed
    cv2.imshow("frame..", frame)
    blur_frame = cv2.bilateralFilter(frame, 11, 200, 200)
    cv2.imshow("blurred_frame..", blur_frame)
    # convert to gray scale
    gray = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2GRAY)
    # use threshold to make the frame binary,black or white
    _, thresh = cv2.threshold(gray, 100, 240, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cv2.imshow("thresh", thresh)
    cv2.imshow("closing", closing)
    # cv2.imshow("opening", opening)
    contours = cv2.findContours(
        closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)  # grab contours
    print("MAIN", len(contours))
    centroids = utils.ret_centroids(contours)
    print(centroids)
    test_objs = ct.update(centroids)
    print("objs", test_objs)
    for contour in contours:  # loop over all contours
        a = cv2.contourArea(contour)  # area of the contour
        if 250000 > a > 100:  # bound the area of the contour
            approx = cv2.approxPolyDP(
                contour, 0.01*cv2.arcLength(contour, True), True)
            # cv2.drawContours(frame, [approx], -1, (0, 0, 255), 3)
            rect = cv2.minAreaRect(contour)  # get minimum area rectangle
            # converts rect to box,co-ord array of 4 elements
            box = cv2.boxPoints(rect)
            box_main = box.astype('int')  # convert the box into integer format
            ordered_box = utils.order_points(box_main)
            tl, tr, br, bl = ordered_box[0], ordered_box[1], ordered_box[2], ordered_box[3]
            tltrX, tltrY = utils.midpoint(tl, tr)
            blbrX, blbrY = utils.midpoint(bl, br)
            tlblX, tlblY = utils.midpoint(tl, bl)
            trbrX, trbrY = utils.midpoint(tr, br)
            M = cv2.moments(contour)
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            #print("kuki", rect)
            wh = rect[1]
            w = wh[0]
            h = wh[1]
            #print("wh", wh)
            if h <= FRAME_HEIGHT and w <= FRAME_WIDTH:  # filter out wheather frame as a contour area
                cv2.drawContours(frame, [box_main], -1, (0, 255, 0), 3)
            else:
                continue
            # print(rect)
            print(len(approx))
            abs_width = round(LENGTH_CONST*w, 3)
            abs_height = round(LENGTH_CONST*h, 3)
            cv2.line(frame, (tltrX, tltrY),
                     (blbrX, blbrY), (255, 0, 0), 2)
            cv2.putText(frame, str(abs_height), (tltrX, tltrY),
                        TEXT_FONT, TEXT_SIZE, TEXT_COLOR)
            cv2.line(frame, (tlblX, tlblY),
                     (trbrX, trbrY), (255, 0, 0), 2)
            cv2.putText(frame, str(abs_width), (tlblX, tlblY),
                        TEXT_FONT, TEXT_SIZE, TEXT_COLOR)
            if len(approx) == 4:  # check for rectangle
                area_contour = str(round(a*AREA_CONST, 2))+"in^2"
                if abs(w-h) <= 50:  # cheak wheather height and width nearly equal
                    cv2.putText(frame, "Square :"+area_contour, (cx, cy),
                                TEXT_FONT, TEXT_SIZE, TEXT_COLOR)
                else:
                    cv2.putText(frame, "Rectangle :"+area_contour, (cx, cy),
                                TEXT_FONT, TEXT_SIZE, TEXT_COLOR)
            elif len(approx) == 3:  # check for triangle
                area_contour = str(round(a*AREA_CONST, 2))+"in^2"
                cv2.putText(frame, "Triangle :" + area_contour, (cx, cy),
                            TEXT_FONT, TEXT_SIZE, TEXT_COLOR)
            elif len(approx) == 5:  # cheack for pentagone
                area_contour = str(round(a*AREA_CONST, 2))+"in^2"
                cv2.putText(frame, "Pentagone:" + area_contour, (cx, cy),
                            TEXT_FONT, TEXT_SIZE, TEXT_COLOR)
            elif len(approx) == 10 and abs(w-h) <= 50:  # cheack for Star
                area_contour = str(round(a*AREA_CONST, 2))+"in^2"
                cv2.putText(frame, "Star:" + area_contour, (cx, cy),
                            TEXT_FONT, TEXT_SIZE, TEXT_COLOR)
            elif len(approx) > 10:  # check for wheater it is circle or elips
                area_contour = str(round(a*AREA_CONST, 2))+"in^2"
                if abs(w-h) <= 50:
                    cv2.putText(frame, "Circle:" + area_contour, (cx, cy),
                                TEXT_FONT, TEXT_SIZE, TEXT_COLOR)
                else:
                    cv2.putText(frame, "Ellipse:" + area_contour, (cx, cy),
                                TEXT_FONT, TEXT_SIZE, TEXT_COLOR)
    cv2.imshow("video", frame)
    cv2.imshow("thresh", thresh)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
