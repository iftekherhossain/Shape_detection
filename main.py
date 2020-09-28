"""
This module is used writting various utility class and methods for shape detection.
"""
import numpy as np
import cv2
from collections import OrderedDict
from scipy.spatial import distance as dist


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
            if 250000 > a > 100:
                M = cv2.moments(cnt)
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                cents = [cx, cy]
                centroids.append(cents)
        return centroids


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
