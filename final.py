import numpy as np
import cv2
import imutils
from main import ShapeUtils, CentroidTracker, DataBase
from collections import OrderedDict
import matplotlib.pyplot as plt
from imutils.video import FPS
from imutils.video import WebcamVideoStream

cap = cv2.VideoCapture(0)
utils = ShapeUtils()
ct = CentroidTracker()

TEXT_COLOR = (0, 0, 255)
TEXT_SIZE = 1
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
kernel = np.ones((5, 5), np.uint8)
LENGTH_CONST = 0.01
AREA_CONST = 0.0001
FRAME_HEIGHT = 400
FRAME_WIDTH = 540
# kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
count = 0
c_arr = []
_, first_frame = cap.read()
first_frame = first_frame[:400, 100:]

db = DataBase('shapes.db')
item_id = len(db.read_from_db())

fps = FPS().start()
while True:
    fps.update()
    _, frame = cap.read()  # grab frames from the video feed
    frame = frame[:400, 100:]
    print("frame shape", frame.shape)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    difference = cv2.absdiff(first_frame, frame)
    cv2.imshow("diff", difference)
    blur_frame = cv2.bilateralFilter(difference, 11, 200, 200)
    # cv2.imshow("blurred_frame..", blur_frame)
    gray = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)
    _, thresh = cv2.threshold(gray, 45, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    # cv2.imshow("thresh", thresh)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    # cv2.imshow("opening", opening)
    #-----------------------------------------------------------------------------#
    contours = cv2.findContours(
        opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)  # grab contours
    #-----------------------------------------------------------------------------#
    cv2.line(frame, (300, 0), (300, 639), (0, 0, 255), 2)
    print("MAIN", len(contours))
    centroids = utils.ret_centroids(contours)
    objects = ct.update(centroids)
    print("objs", len(objects))
    ids = objects.keys()
    print(c_arr)
    for i in ids:
        c = objects[i]
        x_, y_ = c[0], c[1]
        if utils.check_linecross(300, x_, 20) and i not in c_arr:
            count += 1
            c_arr.append(i)
        cv2.putText(frame, "ID : " + str(i), (x_, y_), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 3, cv2.LINE_AA, False)
    print("COUNTING", count)
    for contour in contours:  # loop over all contours
        a = cv2.contourArea(contour)  # area of the contour
        if 210000 > a > 130:  # bound the area of the contour
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
            color_frame = frame[tl[1]:br[1], tl[0]:br[0]]
            bgr = np.sum(np.sum(color_frame, axis=0), axis=0)
            sh = (br[0]-tl[0])*(br[1]-tl[1])
            b = str(bgr[0]//sh)
            g = str(bgr[1]//sh)
            r = str(bgr[2]//sh)
            color = b+" "+g+" "+r
            _shape = utils.print_all(frame, approx, cx, cy, a, w, h)
            if utils.check_linecross(300, cx, 10):
                db.data_entry(int(item_id), str(_shape),
                              str(a*AREA_CONST), color)
                print("success")
                item_id += 1
            #print("wh", wh)
            if h <= FRAME_HEIGHT and w <= FRAME_WIDTH:  # filter out wheather frame as a contour area
                cv2.drawContours(frame, [box_main], -1, (0, 255, 0), 3)
            else:
                continue

    cv2.imshow("frame", frame)

    k = cv2.waitKey(1)
    if k == 27:
        break


fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
