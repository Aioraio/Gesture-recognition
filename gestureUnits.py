import math
import numpy as np
import cv2


def angle(m, n):
    cos = np.dot(m, n) / (
        np.dot(m, m) ** 0.5 * np.dot(n, n) ** 0.5
    )  # cos = m.n/|m|*|n|
    return math.degrees(math.acos(cos))


class GestureDetector:
    def __init__(self, lms=[]):
        self.lms = lms

        # 获得凸包

    def convexHull(self, draw=True):
        if self.lms:
            left_lmsTonp = np.array(self.lms, dtype=np.int32)
            hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17, 10]
            hull = cv2.convexHull(left_lmsTonp[hull_index])
            if draw:
                cv2.polylines(self.lms, [hull], True, (0, 255, 0), 2)
            return hull

    # 得到凸包外的点
    def outFingers(self, tips=[4, 8, 12, 16, 20]):
        out_fingers = []
        hull = self.convexHull(draw=False)
        if self.lms:
            for i in tips:
                pt = (int(self.lms[i][0]), int(self.lms[i][1]))
                dist = cv2.pointPolygonTest(hull, pt, True)
                if dist < 0:
                    out_fingers.append(i)
        return out_fingers

    def get_guester(self, out_fingers):
        str_guester = ''
        if len(out_fingers) == 0 and self.lms:
            str_guester = "0"
        if len(out_fingers) == 1 and out_fingers == [8]:
            v1 = np.array(self.lms[8]) - np.array(self.lms[7])
            v2 = np.array(self.lms[6]) - np.array(self.lms[7])
            if angle(v1, v2) > 150:
                str_guester = "1"
            else:
                str_guester = "9"

        if len(out_fingers) == 2 and out_fingers == [8, 12]:
            str_guester = "2"

        if len(out_fingers) == 3 and out_fingers == [12, 16, 20]:
            str_guester = "3"

        if len(out_fingers) == 4 and out_fingers == [8, 12, 16, 20]:
            str_guester = "4"

        if len(out_fingers) == 5:
            str_guester = "5"

        if len(out_fingers) == 2 and out_fingers == [4, 20]:
            str_guester = "6"

        if len(out_fingers) == 2 and out_fingers == [4, 8]:
            str_guester = "7"

        if len(out_fingers) == 3 and out_fingers == [4, 8, 12]:
            str_guester = "8"

        # 单词
        # if len(out_fingers)==1 and out_fingers[0]==4:
        #     str_guester = "Good"

        return str_guester

    def gestureNum(self):
        out_fingers = self.outFingers()
        return self.get_guester(out_fingers)
