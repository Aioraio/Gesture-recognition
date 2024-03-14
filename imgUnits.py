import cv2
import math


class Imgprocess:
    def __init__(self):
        None
    
    # 获取两点之间距离
    def joint_distance(self, x, y):
        Xd = x[0] - y[0]
        Yd = x[1] - y[1]
        return int(math.sqrt(Xd * Xd + Yd * Yd))

    # 获取感兴趣的区域图像
    def get_roi(self, img, lms, draw=True, show=False):
        if any(lms):
            # 得到相对距离
            # alpha = joint_distance(lms[17], lms[5])  # 5结点到17节点的距离,基本不变
            beta = self.joint_distance(lms[9], lms[0])

            alpha = beta
            # 得到起始点的位置及roi区域的宽和高
            x_0 = (
                lms[0][0] - int(1 * alpha) if lms[0][0] - int(1 * alpha) > 0 else 0
            )  # 如果小于0则是0
            y_0 = lms[0][1] - int(2.3 * beta) if lms[0][1] - int(2.3 * beta) > 0 else 0
            width = int(2.5 * alpha)
            height = int(2.5 * beta)

            # x_0 = 100
            # y_0 = 200
            # width = 160
            # height = 160

            if draw == True:
                cv2.rectangle(
                    img, (x_0, y_0), (x_0 + width, y_0 + height), (0, 255, 255), 2
                )

            roi = img[
                y_0 + 2 : y_0 + height - 2, x_0 + 2 : x_0 + width - 2
            ]  # 避免如果绘制矩形把矩形也加入roi

            if show == True:
                cv2.imshow('Roi', roi)
            return roi

    def process_img(self, roi, size=28):
        img = cv2.bilateralFilter(roi, 9, 75, 75)  # 双边滤波
        img = cv2.resize(roi, (size, size))  # 默认重整为28*28尺寸
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度处理
        return img

    def skinMask(self,roi):
        YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)  # 转换至YCrCb空间
        (y, cr, cb) = cv2.split(YCrCb)  # 拆分出Y,Cr,Cb值
        cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
        _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Ostu处理
        res = cv2.bitwise_and(roi, roi, mask=skin)
        return res