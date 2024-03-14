import time
import cv2
import numpy as np
import pyautogui as pag
from handUnits import HandDetector
from gestureUnits import GestureDetector
from fileUnits import File
from imgUnits import Imgprocess

camera = cv2.VideoCapture(0)
hand_detector = HandDetector()
gestureDetector = GestureDetector()
file = File()
imgprocess = Imgprocess()

num = 0
thickness = 2
turn_on = False

Characterdict = {
    0: 'a',
    1: 'b',
    2: 'c',
    3: 'd',
    4: 'e',
    5: 'f',
    6: 'g',
    7: 'h',
    8: 'i',
    9: 'j',
    10: 'k',
    11: 'l',
    12: 'm',
    13: 'n',
    14: 'o',
    15: 'p',
    16: 'q',
    17: 'r',
    18: 's',
    19: 't',
    20: 'u',
    21: 'v',
    22: 'w',
    23: 'x',
    24: 'y',
    25: 'z',
}


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


# 获取部分坐标
def get_partOflms(pos, lms):
    get_list = []
    for i in pos:
        if lms:
            get_list.append(lms[i])
    return get_list


# 识别手势(数字)
def gestureNum(lms, show=True, text_pos=(20, 300), color=(255, 0, 0)):
    gestureDetector.__init__(lms)
    if show == True:
        cv2.putText(
            img,
            gestureDetector.gestureNum(),
            text_pos,
            cv2.FONT_HERSHEY_PLAIN,
            3,
            color,
            2,
        )
    return gestureDetector.gestureNum()


def Move(img, lms):
    h, w, c = img.shape
    width, height = pag.size()

    H = height / int(0.5 * h)
    W = width / int(0.5 * w)
    pag.PAUSE = 0.01
    pag.FAILSAFE = False

    cv2.rectangle(
        img,
        (0, 0),
        (int(0.5 * w), int(0.5 * h)),
        (0, 255, 255),
        2,
    )

    if any(lms):
        cv2.circle(img, (lms[12][0], lms[12][1]), 8, (0, 0, 255), cv2.FILLED)
        x = W * lms[12][0]
        y = H * lms[12][1]
        pag.moveTo(x, y, duration=0)


def Click(lms):
    gesture = gestureNum(lms, text_pos=(300, 300), color=(255, 255, 0))
    if gesture == '2':
        base_distance = imgprocess.joint_distance(lms[9], lms[0])
        click_distance = imgprocess.joint_distance(lms[8], lms[12])

        if base_distance > int(4 * click_distance):
            pag.click(interval=0.1)
        if base_distance < int(1.1 * click_distance):
            pag.click(clicks=2, interval=0.1)

        # print('click_distance: ', click_distance)
        # print(' base_distance: ', base_distance)


while True:
    time_1 = time.time()
    success, img = camera.read()

    if success:
        img = cv2.flip(img, 1)  # 水平翻转

        hand_detector.process(img, draw=False)

        # 获取右手坐标点
        right_lms = hand_detector.get_lms(img, 'Right')

        # 获取左手坐标点
        left_lms = hand_detector.get_lms(img, 'Left')

        num = num + 1
        h, w, c = img.shape

        if right_lms:
            if (
                right_lms[12][0] in range(int(0.9 * w) - 10, int(0.9 * w) + 10)
                and right_lms[12][1] in range(int(0.25 * h) - 10, int(0.25 * h) + 10)
                and num == 1
            ):
                turn_on = not turn_on

        if turn_on == False:
            thickness = 2
        else:
            thickness = -1

        cv2.circle(img, (int(0.9 * w), int(0.25 * h)), 20, (0, 0, 255), thickness)

        if num == 3:
            num = 0

        gestureNum(right_lms)

        # 鼠标移动
        Move(img, right_lms)  # 单线程

        left_gesture = gestureNum(left_lms)
        right_gesture = gestureNum(right_lms, text_pos=(300, 300), color=(255, 255, 0))

        # 点击与双击
        Click(right_lms)

        # # 输入左手数字

        if right_gesture == '3' and left_gesture:
            pag.typewrite(left_gesture)

        # # 删除
        if right_gesture == '4':
            # pag.hotkey('ctrl', 'a')
            pag.press('backspace')

        # 统计屏幕帧率
        pTime = Frame(pTime)

        cv2.imshow('Video', img)

        time_2 = time.time()

        # Test
        # print(str(round(time_2-time_1,3)))

    else:
        print("获取失败")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


camera.release()
cv2.destroyAllWindows()
