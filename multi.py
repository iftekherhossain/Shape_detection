import numpy as np
import cv2
import imutils
from main import ShapeUtils, CentroidTracker, Threading, DataBase
from collections import OrderedDict
import matplotlib.pyplot as plt
from imutils.video import FPS
from threading import Thread


cap = cv2.VideoCapture(0)
utils = ShapeUtils()
ct = CentroidTracker()
cus_thread = Threading()

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

db = DataBase('shapes.db')
item_id = len(db.read_from_db())
print("hello", item_id)


def counting(contours, frame):
    global count
    centroids = utils.ret_centroids(contours)
    objects = ct.update(centroids)
    print("objs", len(objects))
    ids = objects.keys()
    print(c_arr)
    for i in ids:
        c = objects[i]
        if c[0] < 80:  # if any object centroid tends to end of the frame ignore the frame
            continue
        x_, y_ = c[0], c[1]
        if utils.check_linecross(300, x_, 20) and i not in c_arr:
            count += 1
            c_arr.append(i)
        cv2.putText(frame, "ID : " + str(i), (x_, y_), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 3, cv2.LINE_AA, False)
    print("COUNTING", count)


def shape_approx(contours, frame):
    for contour in contours:  # loop over all contours
        a = cv2.contourArea(contour)  # area of the contour
        if 210000 > a > 100:  # bound the area of the contour
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
            _shape = utils.print_all(frame, approx, cx, cy, a, w, h)
            #print("wh", wh)
            if h <= FRAME_HEIGHT and w <= FRAME_WIDTH:  # filter out wheather frame as a contour area
                cv2.drawContours(frame, [box_main], -1, (0, 255, 0), 3)
            else:
                continue


_, first_frame = cap.read()
first_frame = first_frame[:400, 100:]
fps = FPS().start()
while True:
    fps.update()
    _, frame = cap.read()  # grab frames from the video feed
    frame = frame[:400, 100:]
    opening = cus_thread.ret_opening(frame, first_frame)
    #-----------------------------------------------------------------------------#
    contours = cv2.findContours(
        opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)  # grab contours
    #-----------------------------------------------------------------------------#
    cv2.line(frame, (300, 0), (300, 639), (0, 0, 255), 2)
    print("MAIN", len(contours))
    t1 = Thread(target=counting, args=(contours, frame))
    t2 = Thread(target=shape_approx, args=(contours, frame))

    t1.start()
    t2.start()

    t1.join()
    # t2.join()
    cv2.imshow("frame", frame)
    #shape_approx(contours, frame)
    k = cv2.waitKey(1)
    if k == 27:
        break


fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
