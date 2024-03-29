import cv2
import mediapipe as mp

class HandDetector():
    def __init__(self):
        self.hand_detector = mp.solutions.hands.Hands()
        self.drawer = mp.solutions.drawing_utils

    # 画出手部节点    
    def process(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # bgr to rgb
        self.hands_data = self.hand_detector.process(img_rgb)
        if draw:
            if self.hands_data.multi_hand_landmarks:
                for handlms in self.hands_data.multi_hand_landmarks:
                    self.drawer.draw_landmarks(img, handlms, mp.solutions.hands.HAND_CONNECTIONS)


    # 找到位置
    def find_position(self, img):
        h, w, c = img.shape
        position = {'Left':{},'Right':{}}
        if self.hands_data.multi_hand_landmarks:
            i = 0
            for point in self.hands_data.multi_handedness:
                score = point.classification[0].score   # 置信率
                if score >= 0.9:
                    label = point.classification[0].label # 左右手标签
                    hand_lms = self.hands_data.multi_hand_landmarks[i].landmark   # 每个点的位置
                    for id, lm in enumerate(hand_lms):
                        x, y = int(lm.x * w), int(lm.y * h)
                        position[label][id] = (x, y)
                i = i + 1
        return position

    def get_lms(self, img, direction):
        lms = []
        position = self.find_position(img)
        if any(position[direction]):
            for i in range(21):
                pos_x = position[direction].get(i)[0]
                pos_y = position[direction].get(i)[1]
                lms.append([int(pos_x), int(pos_y)])
        return lms

    def get_seq(self,name):
        seq = []
        if name == "tips":
            seq = [4, 8, 12, 16, 20]
        elif name == "roots":
            seq = [3, 5, 9, 13, 17]
        
        return seq
        

    