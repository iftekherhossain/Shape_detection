import cv2
import numpy as np

image = cv2.imread('pic.jpg')
cv2.imshow("image", image)
frame = image[0:2, 0:2]
print(frame)
a = np.sum(np.sum(frame, axis=0), axis=0)
print('af', a)
cv2.imshow("frame", frame)
cv2.waitKey(0)
