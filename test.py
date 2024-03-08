import cv2
import time
import numpy as np
 
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
 
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

#创建VideoWriter类对象
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

MAX_NUM_IMG = 10
num_img = 1
num_frame = 1

while True:
    ret, frame = cap.read()
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print(fps)
    if ret:
 
        frame = cv2.flip(frame,1)

        # 视频读取
        # out.write(frame)					#保存帧
        # cv2.imshow('frame',frame)


        
        # 灰度化
        # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # cv2.imshow('frame',gray)

        # if num_frame%30 == 15:  # 每30帧取一张图像
        #     if num_img <= MAX_NUM_IMG:
        #         cv2.imwrite('img/'+str(num_img)+'.jpg',gray)
        #         num_img = num_img + 1
        # num_frame = num_frame + 1


        # 肤色检测: YCrCb中 133<=Cr<=173 1<=Cb<=120
        # img = frame
        # ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) # 把图像转换到YUV色域
        # (y, cr, cb) = cv2.split(ycrcb) # 图像分割, 分别获取y, cr, br通道分量图像

        # skin2 = np.zeros(cr.shape, dtype=np.uint8) # 根据源图像的大小创建一个全0的矩阵,用于保存图像数据
        # (x, y) = cr.shape # 获取源图像数据的长和宽

        # # 遍历图像, 判断Cr和Br通道的数值, 如果在指定范围中, 则置把新图像的点设为255,否则设为0
        # for i in  range(0, x): 
        #     for j in  range(0, y):
        #         if (cr[i][j] >  140) and (cr[i][j] <  175) and (cb[i][j] >  100) and (cb[i][j] <  120):
        #             skin2[i][j] =  255
        #         else:
        #             skin2[i][j] =  0

        # cv2.imshow('imname', img)
        # cv2.imshow('imname' +  " Skin2 Cr+Cb", skin2)

        # 背景减除
        
        # fgbg = cv2.createBackgroundSubtractorMOG2()
        # fgmask = fgbg.apply(frame)
        # cv2.imshow("fgbg",fgmask)








    else : 
        print("获取失败")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()