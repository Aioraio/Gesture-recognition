import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
from cvzone.FPS import FPS

cap = cv2.VideoCapture(0)

segmentor = SelfiSegmentation()
fpsReader = FPS()

print(cv2.__version__)
while True:
    success, img = cap.read()
    if success:
        img = cv2.flip(img, 1)  # 水平翻转
        # cv2.imshow("Image", img)
        # imgOut = segmentor.removeBG(img, (255, 255, 255), threshold=0.99)

        # cv2.imshow("Image", imgOut)
        cv2.imshow('img', img)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
