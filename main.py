"""
This module is used writting various utility class and methods for shape detection.
"""
import numpy as np
import cv2
from collections import OrderedDict
from scipy.spatial import distance as dist
import sqlite3

TEXT_COLOR = (0, 0, 255)
TEXT_SIZE = 1
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
kernel = np.ones((5, 5), np.uint8)
LENGTH_CONST = 0.01
AREA_CONST = 0.0001
FRAME_HEIGHT = 400
FRAME_WIDTH = 540


class ShapeUtils():
    """This class is used for various shape detection utilites.
    """
    @staticmethod
    def order_points(pts):
        """This function is used for sorting points in a specific way

        Args:
            A list of 4 points. All the points should a array.

        Returns:
            returns a numpy array in sorted order(Top Left>Top Right>Bottom Left> Bottom Right).

        """
        x_sorted = sorted(pts, key=lambda x: x[0])

        # separate leftmost and rightmost points
        left_most = x_sorted[:2]
        right_most = x_sorted[2:]

        # sort leftmost points according to their y coordinates
        sorted_left_most = sorted(left_most, key=lambda x: x[1])

        # sort rightmost points according to their y coordinates
        sorted_right_most = sorted(right_most, key=lambda x: x[1])

        # top_left and bottom left
        top_left = sorted_left_most[0]
        bottom_left = sorted_left_most[1]

        # top_rigth and bottom right
        top_right = sorted_right_most[0]
        bottom_right = sorted_right_most[1]
        return np.array([top_left, top_right, bottom_right, bottom_left])

    @staticmethod
    def midpoint(a, b):
        """Finds the midpoint of any two points.

        Args:
            coordinates of the points

        Returns:
            Returns the coordinate of the midpoint of the given points.

        """
        midX = (a[0]+b[0])//2
        midY = (a[1]+b[1])//2
        return midX, midY

    @staticmethod
    def ret_centroids(contours):
        centroids = []
        for cnt in contours:
            a = cv2.contourArea(cnt)
            if 210000 > a > 150:
                M = cv2.moments(cnt)
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                cents = [cx, cy]
                centroids.append(cents)
        return centroids

    @staticmethod
    def check_linecross(line, point, clearance):
        a_ = abs(int(point-line))
        if a_ <= clearance:
            return 1
        else:
            return 0

    @staticmethod
    def print_all(frame, approx, cx, cy, a, w, h, area_const=AREA_CONST, font=TEXT_FONT, size=TEXT_SIZE, color=TEXT_COLOR):
        if len(approx) == 4:  # check for rectangle
            area_contour = str(round(a*area_const, 2))+"in^2"
            if abs(w-h) <= 50:  # cheak wheather height and width nearly equal
                cv2.putText(frame, "Square :"+area_contour, (cx, cy),
                            font, size, color)
                return "square"
            else:
                cv2.putText(frame, "Rectangle :"+area_contour, (cx, cy),
                            font, size, color)
                return "rectangle"
        elif len(approx) == 3:  # check for triangle
            area_contour = str(round(a*AREA_CONST, 2))+"in^2"
            cv2.putText(frame, "Triangle :" + area_contour, (cx, cy),
                        font, size, color)
            return "triangle"
        elif len(approx) == 5:  # cheack for pentagone
            area_contour = str(round(a*AREA_CONST, 2))+"in^2"
            cv2.putText(frame, "Pentagone:" + area_contour, (cx, cy),
                        font, size, color)
            return "pentagone"
        elif len(approx) == 10 and abs(w-h) <= 50:  # cheack for Star
            area_contour = str(round(a*AREA_CONST, 2))+"in^2"
            cv2.putText(frame, "Star:" + area_contour, (cx, cy),
                        font, size, color)
            return "star"
        elif len(approx) > 10:  # check for wheater it is circle or elips
            area_contour = str(round(a*AREA_CONST, 2))+"in^2"
            if abs(w-h) <= 50:
                cv2.putText(frame, "Circle:" + area_contour, (cx, cy),
                            font, size, color)
                return "circle"
            else:
                cv2.putText(frame, "Ellipse:" + area_contour, (cx, cy),
                            font, size, color)
                return "ellipse"


class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.dissapeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.dissapeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.dissapeared[objectID]

    def update(self, inputCentroids):
        if len(inputCentroids) == 0:
            for objectID in list(self.dissapeared.keys()):
                self.dissapeared[objectID] += 1

                if self.dissapeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.dissapeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:

                for row in unusedRows:

                    objectID = objectIDs[row]
                    self.dissapeared[objectID] += 1

                    if self.dissapeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

                # return the set of trackable objects
        return self.objects


class Threading():
    @staticmethod
    def ret_opening(frame, first_frame):
        #frame = frame[:400, 100:]
        print("frame shape", frame.shape)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        difference = cv2.absdiff(first_frame, frame)
        #cv2.imshow("diff", difference)
        blur_frame = cv2.bilateralFilter(difference, 11, 200, 200)
        #cv2.imshow("blurred_frame..", blur_frame)
        gray = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("gray", gray)
        _, thresh = cv2.threshold(
            gray, 45, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        #cv2.imshow("thresh", thresh)
        opening = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        cv2.imshow("opening", opening)
        return opening


class DataBase():
    def __init__(self, file):
        self.conn = sqlite3.connect(file)
        self.c = self.conn.cursor()
        self.c.execute(
            'CREATE TABLE IF NOT EXISTS my_table(id INTEGER, shape TEXT, size TEXT, color TEXT)')

    def data_entry(self, id, shape, size, color=""):
        self.c.execute("INSERT INTO my_table (id,shape,size,color) VALUES (?, ?, ?, ?)",
                       (id, shape, size, color))
        self.conn.commit()

    def read_from_db(self):
        self.c.execute('SELECT * FROM my_table')
        data = self.c.fetchall()
        return data
