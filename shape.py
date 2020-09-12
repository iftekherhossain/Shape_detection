import numpy as np
import cv2
import imutils
from main import ShapeUtils


cap = cv2.VideoCapture(0)
utils = ShapeUtils()

TEXT_COLOR = (0, 0, 255)
TEXT_SIZE = 1
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX

while True:
    _, frame = cap.read()  # grab frames from the video feed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    # use threshold to make the frame binary,black or white
    _, thresh = cv2.threshold(gray, 100, 240, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    contours = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)  # grab contours
    for contour in contours:  # loop over all contours
        a = cv2.contourArea(contour)  # area of the contour
        if 250000 > a > 100:  # bound the area of the contour
            approx = cv2.approxPolyDP(
                contour, 0.01*cv2.arcLength(contour, True), True)
            #cv2.drawContours(frame, [approx], -1, (0, 0, 255), 3)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box_main = box.astype('int')
            M = cv2.moments(contour)
            ordered_box = utils.order_points(box_main)
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            wh = rect[1]
            w = wh[0]
            h = wh[1]
            print("wh", wh)
            #cv2.drawContours(frame, [box_main], -1, (0,255,0),3)
            print(rect)
            print(len(approx))
            if len(approx) == 4:  # check for rectangle
                if abs(w-h) <= 50:  # cheak wheather height and width nearly equal
                    cv2.putText(frame, "Square", (cx, cy),
                                TEXT_FONT, TEXT_SIZE, TEXT_COLOR)
                else:
                    cv2.putText(frame, "Rectangle", (cx, cy),
                                TEXT_FONT, TEXT_SIZE, TEXT_COLOR)
            elif len(approx) == 3:  # check for triangle
                cv2.putText(frame, "Triangle", (cx, cy),
                            TEXT_FONT, TEXT_SIZE, TEXT_COLOR)

            elif len(approx) == 5:  # cheack for pentagone
                cv2.putText(frame, "Pentagone", (cx, cy),
                            TEXT_FONT, TEXT_SIZE, TEXT_COLOR)
            elif len(approx) == 10:  # cheack for Star
                cv2.putText(frame, "Star", (cx, cy),
                            TEXT_FONT, TEXT_SIZE, TEXT_COLOR)
            elif len(approx) > 10:  # check for wheater it is circle or elips
                if abs(w-h) <= 50:
                    cv2.putText(frame, "Circle", (cx, cy),
                                TEXT_FONT, TEXT_SIZE, TEXT_COLOR)
                else:
                    cv2.putText(frame, "Elips", (cx, cy),
                                TEXT_FONT, TEXT_SIZE, TEXT_COLOR)
    cv2.imshow("video", frame)
    cv2.imshow("thresh", thresh)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
