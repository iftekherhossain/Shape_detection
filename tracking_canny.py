import numpy as np
import cv2
import imutils
from main import ShapeUtils, CentroidTracker
from collections import OrderedDict

cap = cv2.VideoCapture(0)
utils = ShapeUtils()
ct = CentroidTracker(20)

TEXT_COLOR = (0, 0, 255)
TEXT_SIZE = 1
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
kernel = np.ones((5, 5), np.uint8)
LENGTH_CONST = 0.01
AREA_CONST = 0.0001
FRAME_HEIGHT = 470
FRAME_WIDTH = 630
count = 0
c_arr = []
while True:
    _, frame = cap.read()  # grab frames from the video feed
    cv2.imshow("frame..", frame)
    frame = frame[0:400, :]
    #cv2.imshow("cut_frame..", cut_frame)
    blur_frame = cv2.bilateralFilter(frame, 11, 150, 150)
    cv2.imshow("blurred_frame..", blur_frame)
    # median_blur = cv2.medianBlur(frame, 15)
    # gaussian_blur = cv2.GaussianBlur(frame, (15, 15), 0)
    # cv2.imshow("gaussian_blur", gaussian_blur)
    # cv2.imshow("median_blur.", median_blur)

    # blur = cv2.blur(frame, (7, 7))
    # cv2.imshow("blur", blur)
    print("ha", frame.shape)
    cv2.line(frame, (300, 0), (300, 639), (0, 0, 255), 2)
    canny = cv2.Canny(blur_frame, 100, 200)
    # laplacian = cv2.Laplacian(blur_frame, cv2.CV_64F)
    # sobelx = cv2.Sobel(blur_frame, cv2.CV_64F, 0, 1, ksize=5)
    cv2.imshow("Canny", canny)
    # cv2.imshow("sobelx", sobelx)

    # # convert to gray scale
    # gray = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2GRAY)
    # # use threshold to make the frame binary,black or white
    # _, thresh = cv2.threshold(gray, 100, 240, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    # closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    # # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # cv2.imshow("thresh", thresh)
    # cv2.imshow("closing", closing)
    # cv2.imshow("opening", opening)
    contours = cv2.findContours(
        canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)  # grab contours

    print("MAIN", len(contours))
    centroids = utils.ret_centroids(contours)
    # for cen in centroids:
    #     if utils.check_linecross(300, cen[0]):
    #         count += 1
    print("count", count)
    print(centroids)
    objects = ct.update(centroids)
    print("objs", objects)
    ids = objects.keys()
    for i in ids:
        c = objects[i]
        x_, y_ = c[0], c[1]
        if utils.check_linecross(300, x_) and i not in c_arr:
            count += 1
            c_arr.append(i)

        cv2.putText(frame, "ID : " + str(i), (x_, y_), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 3, cv2.LINE_AA, False)

    cv2.imshow("video", frame)
    # cv2.imshow("thresh", thresh)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
