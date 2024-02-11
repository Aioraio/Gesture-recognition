import math
import numpy as np

def angle(m, n):
    cos = np.dot(m,n) / (np.dot(m,m)**0.5 * np.dot(n,n)**0.5)   # cos = m.n/|m|*|n|        
    return math.degrees(math.acos(cos))

class GestureDetector():
    def __init__(self,out_fingers,lms):
        self.out_fingers = out_fingers
        self.lms = lms
    


    def get_guester(self):
        str_guester = ""
        # 数字
        if len(self.out_fingers)==0 and self.lms:
            str_guester = "0"
        if len(self.out_fingers)==1 and self.out_fingers == [8]:
            v1 = np.array(self.lms[8]) - np.array(self.lms[7])
            v2 = np.array(self.lms[6]) - np.array(self.lms[7])
            if angle(v1, v2) > 150:
                str_guester = "1"
            else: str_guester = "9"

        if len(self.out_fingers)==2 and self.out_fingers == [8,12]:
            str_guester = "2"

        if len(self.out_fingers)==3 and self.out_fingers == [12,16,20]:
            str_guester = "3"

        if len(self.out_fingers)==4 and self.out_fingers == [8,12,16,20]:
            str_guester = "4" 

        if len(self.out_fingers)==5:
            str_guester = "5" 

        if len(self.out_fingers)==2 and self.out_fingers == [4,20]:
            str_guester = "6"
        
        if len(self.out_fingers)==2 and self.out_fingers == [4,8]:
            str_guester = "7"

        if len(self.out_fingers)==3 and self.out_fingers == [4,8,12]:
            str_guester = "8"

        # 单词
        # if len(self.out_fingers)==1 and self.out_fingers[0]==4:
        #     str_guester = "Good"

        return str_guester


