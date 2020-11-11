import cv2
import numpy as np
from main import Background
import numpy as np
#bg = Background()
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=25, detectShadows=False,)


while True:
    _, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv2.imshow("frame", frame)
    cv2.imshow("fgmask", fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
