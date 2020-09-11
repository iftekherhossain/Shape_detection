"""
This module is used writting various utility class and methods for shape detection.
"""
import numpy as np


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
