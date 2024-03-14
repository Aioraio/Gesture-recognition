import cv2
from handUnits import HandDetector
import os
import math
import numpy as np
from fileUnits import File
from imgUnits import Imgprocess

camera = cv2.VideoCapture(0)
hand_detector = HandDetector()
file = File()
imgprocess = Imgprocess()


class DatasetGet:
    def __init__(self):
        None

    def __get_roi(self, img, direction):
        if direction == 'None':
            roi = img
        else:
            hand_detector.process(img, draw=False)
            lms = hand_detector.get_lms(img, direction)
            roi = imgprocess.get_roi(img, lms, show=False)
        return roi

    def captureFrame(
        self, captureNum, max_capture_frame, direction='None', process=False
    ):
        capture_num = captureNum
        capture_frame = 0

        while True:
            success, img = camera.read()

            if success:
                img = cv2.flip(img, 1)  # 水平翻转

                roi = np.array(self.__get_roi(img, direction))
                if roi.any():
                    capture_num = capture_num - 1

                    if capture_num <= 0:
                        capture_num = captureNum
                        capture_frame = capture_frame + 1
                        file_name = str(capture_frame) + '.jpg'
                        file_path = file.get_file_path(
                            file_name, putTofolder=True, folder='img2'
                        )
                        # print(file_path)

                        if process == False:
                            cv2.imwrite(file_path, roi)
                        else:
                            cv2.imwrite(file_path, imgprocess.process_img(roi))  # 保存

                if capture_frame >= max_capture_frame:
                    print("录制完成" + str(max_capture_frame) + '张图片')
                    break

                cv2.imshow('Video', img)

            else:
                print("获取失败")
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("被打断")
                break
        camera.release()
        cv2.destroyAllWindows()
