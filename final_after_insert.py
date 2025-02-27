"""
This script detects shape and color,counts and add to database real time
"""
import numpy as np
import cv2
import imutils
from main import ShapeUtils, CentroidTracker, DataBase, Background
from collections import OrderedDict
import matplotlib.pyplot as plt
from imutils.video import FPS
from imutils.video import WebcamVideoStream
from threading import Thread

cap = cv2.VideoCapture(0)
utils = ShapeUtils()
ct = CentroidTracker(14)
bg = Background()


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
s_arr = []
_, first_frame = cap.read()
first_frame = first_frame[:400, 100:]

db = DataBase('shapes.db')
item_id = len(db.read_from_db())
temp_database = []
fps = FPS().start()
while True:
    fps.update()
    inv_objects = {}
    _, frame = cap.read()  # grab frames from the video feed
    frame = frame[:400, 100:]
    uframe = frame
    print("frame shape", frame.shape)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    opening = bg.background_subtraction_difframe(first_frame, frame)
    #opening = bg.background_subtraction_mog(frame)
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
        if 200000 > a > 130:  # bound the area of the contour
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
            #-----------For color detection----------#
            color_frame = frame[cy:cy+1, cx:cx+1]
            bgr = np.sum(np.sum(color_frame, axis=0), axis=0)
            print("RGB Color", bgr)
            color = str(bgr[2])+" "+str(bgr[1])+" "+str(bgr[0])
            #----------------------------------------#
            _shape = utils.print_all(frame, approx, cx, cy, a, w, h)
            inv_objects = {tuple(v): k for k, v in objects.items()}
            print("inv_objects", inv_objects)
            cur_object_cent = tuple((cx, cy))
            try:
                item_id = inv_objects[cur_object_cent]
            except:
                pass
            if utils.check_linecross(300, cx, 20) and item_id not in s_arr:
                temp = [int(item_id), str(_shape), str(a*AREA_CONST), color]
                temp_database.append(temp)
                print("success")
                s_arr.append(item_id)
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
print(temp_database)

for data in temp_database:
    db.data_entry(data[0], data[1], data[2], data[3])

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
