import cv2
import numpy as np


cap = cv2.VideoCapture(0)
MIN_DESCRIPTOR = 32  # surprisingly enough, 2 descriptors are already enough

# YCrCb颜色空间的Cr分量+Otsu法阈值分割算法
def skinMask(roi):
	YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
	(y,cr,cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
	cr1 = cv2.GaussianBlur(cr, (5,5), 0)
	_, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #Ostu处理
	res = cv2.bitwise_and(roi,roi, mask = skin)
	return res


def find_contours(Laplacian):
    #binaryimg = cv2.Canny(res, 50, 200) #二值化，canny检测
    h = cv2.findContours(Laplacian,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) #寻找轮廓
    contour = h[0]
    contour = sorted(contour, key = cv2.contourArea, reverse=True)#对一系列轮廓点坐标按它们围成的区域面积进行排序
    return contour
 
#截短傅里叶描述子
def truncate_descriptor(fourier_result):
    descriptors_in_use = np.fft.fftshift(fourier_result)
    
    #取中间的MIN_DESCRIPTOR项描述子
    center_index = int(len(descriptors_in_use) / 2)
    low, high = center_index - int(MIN_DESCRIPTOR / 2), center_index + int(MIN_DESCRIPTOR / 2)
    descriptors_in_use = descriptors_in_use[low:high]
    
    descriptors_in_use = np.fft.ifftshift(descriptors_in_use)
    return descriptors_in_use
 
##由傅里叶描述子重建轮廓图
def reconstruct(img, descirptor_in_use):
    #descirptor_in_use = truncate_descriptor(fourier_result, degree)
    #descirptor_in_use = np.fft.ifftshift(fourier_result)
    #descirptor_in_use = truncate_descriptor(fourier_result)
    #print(descirptor_in_use)
    contour_reconstruct = np.fft.ifft(descirptor_in_use)
    contour_reconstruct = np.array([contour_reconstruct.real,
                                    contour_reconstruct.imag])
    contour_reconstruct = np.transpose(contour_reconstruct)
    contour_reconstruct = np.expand_dims(contour_reconstruct, axis = 1)
    if contour_reconstruct.min() < 0:
        contour_reconstruct -= contour_reconstruct.min()
    contour_reconstruct *= img.shape[0] / contour_reconstruct.max()
    contour_reconstruct = contour_reconstruct.astype(np.int32, copy = False)
 
    black_np = np.ones(img.shape, np.uint8) #创建黑色幕布
    black = cv2.drawContours(black_np,contour_reconstruct,-1,(255,255,255),1) #绘制白色轮廓
    cv2.imshow("contour_reconstruct", black)
    #cv2.imwrite('recover.png',black)
    return black


##计算傅里叶描述子
def fourierDesciptor(res):
    #Laplacian算子进行八邻域检测
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    dst = cv2.Laplacian(gray, cv2.CV_16S, ksize = 3)
    Laplacian = cv2.convertScaleAbs(dst)
    contour = find_contours(Laplacian)#提取轮廓点坐标
    contour_array = contour[0][:, 0, :]#注意这里只保留区域面积最大的轮廓点坐标
    ret_np = np.ones(dst.shape, np.uint8) #创建黑色幕布
    ret = cv2.drawContours(ret_np,contour[0],-1,(255,255,255),1) #绘制白色轮廓
    contours_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contours_complex.real = contour_array[:,0]#横坐标作为实数部分
    contours_complex.imag = contour_array[:,1]#纵坐标作为虚数部分
    fourier_result = np.fft.fft(contours_complex)#进行傅里叶变换
    #fourier_result = np.fft.fftshift(fourier_result)
    descirptor_in_use = truncate_descriptor(fourier_result)#截短傅里叶描述子
    #reconstruct(ret, descirptor_in_use)
    return ret, descirptor_in_use




while True:
    ret, frame = cap.read()

    if ret:
        frame = cv2.flip(frame,1)

        # 去噪
        blur = cv2.bilateralFilter(frame,9,75,75)
        res = skinMask(blur)

        # cv2.imshow("img",res)

        # 开运算

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
        # cv2.imshow('ret1',ret1)


        binaryimg = cv2.Canny(res, 100, 200) #二值化，canny检测
        h = cv2.findContours(binaryimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #寻找轮廓
        contours = h[0] #提取轮廓
        ret = np.ones(res.shape, np.uint8) #创建黑色幕布
        cv2.drawContours(ret,contours,-1,(255,255,255),1) #绘制白色轮廓
        cv2.imshow("ret", ret)


        # ret, fourier_result = fourierDesciptor(res)
        # cv2.imshow('ret',ret)


    else : 
        print("获取失败")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
