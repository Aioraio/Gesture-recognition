import time
import math
import cv2
import mediapipe as mp
import numpy as np
import pyautogui as pag
from handUnits import HandDetector
from gestureUnits import GestureDetector
import os

import threading
from concurrent.futures import ThreadPoolExecutor

camera = cv2.VideoCapture(0)
# camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

hand_detector = HandDetector()


tips = [4, 8, 12, 16, 20]
roots = [3, 5, 9, 13, 17]

Click = 0
capture_num = 3
capture_frame = 0

# 帧率显示
pTime = 0


def Frame(pTime):
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(
        img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2
    )
    return pTime


# 圆
def joint_circle(img, finger, color=(0, 0, 255)):
    if finger:
        cv2.circle(img, (finger[0], finger[1]), 10, color, cv2.FILLED)


# 线
def joint_line(img, ptStart, ptEnd, color=(0, 255, 0), thickness=2):
    if ptStart and ptEnd:
        cv2.line(img, ptStart, ptEnd, color, thickness)


# 求两点距离
def joint_distance(x, y):
    Xd = x[0] - y[0]
    Yd = x[1] - y[1]
    return int(math.sqrt(Xd * Xd + Yd * Yd))


# 结点与基点的距离计算
def jointList_distance(jointList, base, Name='None '):
    distance = np.zeros(len(jointList), dtype=int)
    if base:
        # print(Name +'distance: ' )
        for i in range(0, len(jointList)):
            distance[i] = joint_distance(jointList[i], base)  # 求距离
            # print(distance[i])
    return distance


# 获取坐标
def get_lms(position, direction):
    lms = []
    if any(position[direction]):
        for i in range(21):
            pos_x = position[direction].get(i)[0]
            pos_y = position[direction].get(i)[1]
            lms.append([int(pos_x), int(pos_y)])
    return lms


# 获取部分坐标
def get_partOflms(pos, lms):
    get_list = []
    for i in pos:
        if lms:
            get_list.append(lms[i])
    return get_list


# 获得凸包
def convexHull(lms, draw=True):
    if lms:
        left_lmsTonp = np.array(lms, dtype=np.int32)
        hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17, 10]
        hull = cv2.convexHull(left_lmsTonp[hull_index])
        if draw:
            cv2.polylines(img, [hull], True, (0, 255, 0), 2)
        return hull


# 得到凸包外的点
def outFingers(lms, tips=[4, 8, 12, 16, 20]):
    out_fingers = []
    hull = convexHull(lms, draw=False)
    if lms:
        for i in tips:
            pt = (int(lms[i][0]), int(lms[i][1]))
            dist = cv2.pointPolygonTest(hull, pt, True)
            if dist < 0:
                out_fingers.append(i)
    return out_fingers


# 识别手势(数字)
def gestureNum(lms, text_pos=(20, 300), color=(255, 0, 255)):
    out_fingers = outFingers(lms)
    gesture_Detector = GestureDetector(out_fingers, lms)
    cv2.putText(
        img,
        gesture_Detector.get_guester(),
        text_pos,
        cv2.FONT_HERSHEY_PLAIN,
        3,
        color,
        2,
    )
    return gesture_Detector.get_guester()


# 一键调用手势识别
def gestureNum2(img, direction, hand_detector, text_pos=(20, 300), color=(255, 0, 255)):
    hand_detector.process(img, draw=False)
    position = hand_detector.find_position(img)
    lms = get_lms(position, direction)
    out_fingers = outFingers(lms)
    gesture_Detector = GestureDetector(out_fingers, lms)
    cv2.putText(
        img,
        gesture_Detector.get_guester(),
        text_pos,
        cv2.FONT_HERSHEY_PLAIN,
        3,
        color,
        2,
    )


def move(img, lms):
    pag.PAUSE = 0.01
    pag.FAILSAFE = False
    if any(lms):
        joint_circle(img, lms[12])
        pag.moveTo(6 * (lms[12][0] - 160), 4.5 * (lms[12][1] - 120))  # 四分之一大小 居中


def strart(img, hand_detector):
    hand_detector.process(img, draw=False)
    position = hand_detector.find_position(img)
    return position


# 获取感兴趣的区域图像
def get_roi(lms, draw=True, show=False):
    # 得到相对距离
    alpha = joint_distance(lms[17], lms[5])  # 5结点到17节点的距离,基本不变
    beta = joint_distance(lms[9], lms[0])

    # 得到起始点的位置及roi区域的宽和高
    x_0 = (
        lms[0][0] - int(1.8 * alpha) if lms[0][0] - int(1.8 * alpha) > 0 else 0
    )  # 如果小于0则是0
    y_0 = lms[0][1] - int(2.3 * beta) if lms[0][1] - int(2.3 * beta) > 0 else 0
    width = int(3.8 * alpha)
    height = int(2.5 * beta)

    if draw == True:
        cv2.rectangle(img, (x_0, y_0), (x_0 + width, y_0 + height), (0, 255, 255), 2)
    if show == True:
        cv2.imshow('Roi', roi)

    roi = img[
        y_0 + 2 : y_0 + height - 2, x_0 + 2 : x_0 + width - 2
    ]  # 避免如果绘制矩形把矩形也加入roi

    return roi


# 对获取到的图像进行处理
def get_data_img(roi, size=28):
    img = cv2.resize(roi, (size, size))  # 默认重整为28*28尺寸
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度处理
    return img


# 得到要保存的文件路径，并判断是否放进文件夹里
def get_file_path(file_name, putTofolder=False, folder=None):
    current_dir = os.getcwd()
    target_dir_path = current_dir

    if putTofolder == True:
        target_dir_name = folder
        target_dir_path = os.path.join(current_dir, target_dir_name)
        os.makedirs(target_dir_path, exist_ok=True)

    file_path = os.path.join(target_dir_path, file_name)
    return file_path


def skinMask(roi):
    YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)  # 转换至YCrCb空间
    (y, cr, cb) = cv2.split(YCrCb)  # 拆分出Y,Cr,Cb值
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Ostu处理
    res = cv2.bitwise_and(roi, roi, mask=skin)
    return res


while True:
    time_1 = time.time()
    success, img = camera.read()

    if success:
        img = cv2.flip(img, 1)  # 水平翻转

        # 创建包含2个线程的线程池
        # pool = ThreadPoolExecutor(max_workers=2)
        # future1 = pool.submit(strart, img,hand_detector)
        # position = future1.result()

        hand_detector.process(img, draw=False)
        position = hand_detector.find_position(img)

        # 获取右手坐标点
        right_lms = get_lms(position, 'Right')

        # 获取左手坐标点
        left_lms = get_lms(position, 'Left')

        if left_lms:
            roi = get_roi(left_lms)

            capture_num = capture_num - 1

            if capture_num <= 0:
                capture_num = 3
                capture_frame = capture_frame + 1
                file_name = 'test_' + str(capture_frame) + '.jpg'
                file_path = get_file_path(file_name, putTofolder=True, folder='img')
                cv2.imwrite(file_path, get_data_img(roi))  # 保存

        if capture_frame >= 30:
            break

        # if left_lms:
        #     roi = get_roi(left_lms)
        #     res = skinMask(roi)
        #     cv2.imshow('res', res)

        # 手掌基点
        # left_base = position['Left'].get(0, None)
        # right_base = position['Right'].get(0, None)

        # 获取指尖
        # left_tip = get_partOflms(tips,left_lms)
        # right_tip = get_partOflms(tips,right_lms)

        # 获取指根
        # left_root = get_partOflms(roots,left_lms)
        # right_root = get_partOflms(roots,right_lms)

        # 绘制凸包
        # left_hull = convexHull(left_lms)
        # right_hull = convexHull(right_lms)

        # 指尖到手掌的距离
        # left_tipTobase = jointList_distance(left_tip
        # right_tipTobase = jointList_distance(right_tip,right_base)
        # 指根到手掌的距离
        # left_rootTobase = jointList_distance(left_root,left_base)
        # right_rootTobase = jointList_distance(right_root,right_base)

        # gestureNum(left_lms)
        # gestureNum(right_lms,pos=(300,300),color=(255,255,0))

        # 一键调用
        # gestureNum2(img,'Left',hand_detector)
        # gestureNum2(img,'Right',hand_detector,pos=(300,300),color=(255,255,0))

        # 绘制矩形
        # cv2.rectangle(img, (160,120), (480,360), (0, 255, 255), 2)

        # 鼠标移动
        # pag.FAILSAFE=False
        # if any(right_lms):
        #     # print(right_lms[8])
        #     joint_circle(img,right_lms[12])
        #     pag.moveTo(6*(right_lms[12][0]-160), 4.5*(right_lms[12][1]-120),0.01) # 四分之一大小 居中
        #     # print("camera: " + str(right_lms[8][1]))
        #     # print("point: " + str(pag.position()[1]))

        # move(img,right_lms) # 单线程

        # move_thread = threading.Thread(target=move, args=(img,right_lms)) # 多线程
        # move_thread.start()

        # # future2 = pool.submit(move, img,right_lms) # 线程池

        # left_gesture = gestureNum(left_lms)
        # right_gesture = gestureNum(right_lms,text_pos=(300,300),color=(255,255,0))

        # # if right_lms:
        # #     print(joint_distance(right_lms[8],right_lms[12]))

        # # 点击与双击
        # if right_gesture == '2':
        #     if joint_distance(right_lms[8],right_lms[12])<20:
        #         pag.click()
        #         Click = Click + 1
        #     if joint_distance(right_lms[8],right_lms[12])>50:
        #         pag.doubleClick()
        #     # print(joint_distance(right_lms[8],right_lms[12]))
        # # 输入左手数字

        # if right_gesture == '3'and left_gesture:
        #     pag.typewrite(left_gesture)

        # # 删除
        # if right_gesture == '4':
        #     # pag.hotkey('ctrl', 'a')
        #     pag.press('backspace')

        # 统计屏幕帧率
        pTime = Frame(pTime)

        cv2.imshow('Video', img)

        time_2 = time.time()

        # Test
        # print('click_num: ' + str(Click))
        # print(str(round(time_2-time_1,3)))

    else:
        print("获取失败")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


camera.release()
cv2.destroyAllWindows()
