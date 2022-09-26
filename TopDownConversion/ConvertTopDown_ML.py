import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tensorflow.keras as keras
import TopDownfunction_First as TDcvt1
import TopDownfunction_Second_ML as TDcvt2
import time as time
#使用說明: 執行後輸入想要測試的圖片號碼，等待跳出圖片視窗後，用滑鼠點一下球桌(讀取球桌顏色)，之後按下任意鍵讓程式往下執行
width_begin = 10
width_end = 50
c = 0
model = load_model('LongSideModel.h5')
while(c!=27):
    number_of_picture = int(input("Pics:"))
    if(number_of_picture == 0):
        exit()
    #file_name = 'tablephoto/P'+str(number_of_picture)+'.jpeg'
    file_name = 'TablePhotos1/Img'+str(number_of_picture)+'.jpeg'
    #file_name = 'project/dataset/Picture'+str(number_of_picture)+'.jpg'
    OriginPic = cv2.imread(file_name,1)
    #將圖片轉換成1344x1008
    Pic_size=(1344, 1008)
    OriginPic = cv2.resize(OriginPic,Pic_size)

    def OnMouseAction(event,x,y,flags,param):
        global x_index,y_index
        if event == cv2.EVENT_LBUTTONDOWN:
            x_index=x
            y_index=y
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',OnMouseAction)
    cv2.imshow('image',OriginPic)
    cv2.waitKey()
    cv2.destroyAllWindows()


    #HSV transfrom
    OriginPic_HSV = cv2.cvtColor(OriginPic, cv2.COLOR_BGR2HSV)
    #讀取滑鼠點擊位置的HSV值
    Hue = int(OriginPic_HSV[y_index][x_index][0])
    Hue_lower = Hue-10
    Hue_upper = Hue+10
    Saturation = int(OriginPic_HSV[y_index][x_index][1])
    Saturation_lower = Saturation-50
    Saturation_upper = Saturation+50
    Value = int(OriginPic_HSV[y_index][x_index][2])
    Value_lower = Value-50
    Value_upper = Value+50

    ##Test for clustering (Pivot as original HSV)
    '''
    image = OriginPic_HSV*1
    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = image.reshape((-1,3))
    
    # Convert to float type
    pixel_vals = np.float32(pixel_vals)
    #the below line of code defines the criteria for the algorithm to stop running,
    #which will happen is 100 iterations are run or the epsilon (which is the required accuracy)
    #becomes 85%
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    
    # then perform k-means clustering wit h number of clusters defined as 3
    #also random centres are initially choosed for k-means clustering
    k = 3
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    
    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((image.shape))
    cv2.imwrite("SegIm.jpeg", segmented_image)
    '''
    #只留下顏色跟球桌相似的部分
    mask = cv2.inRange(OriginPic_HSV, (Hue_lower, 70, 50), (Hue_upper, 255, 255))
    ROI = cv2.bitwise_and(OriginPic, OriginPic, mask=mask)
    #cv2.imshow('ROI', ROI)
    #cv2.waitKey()

    #binarization
    ROI_gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    triThe ,ROI_Binary = cv2.threshold(ROI_gray, 0, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3))
    ROI_Binary = cv2.erode(ROI_Binary, kernel, iterations=3)
    ROI_Binary = cv2.dilate(ROI_Binary, kernel, iterations=3)

    #抓輪廓
    contours, hierarchy = cv2.findContours(ROI_Binary, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    area = np.zeros(len(contours))
    for i in range(len(contours)):
        area[i]=cv2.contourArea(contours[i])
    index = np.argmax(area)

    #繪製輪廓
    OutputPic = OriginPic*1
    #cv2.drawContours(OutputPic, contours, index , 255, -1)

    #取凸多邊形
    hull = cv2.convexHull(contours[index])
    Table_mask = np.zeros((Pic_size[1], Pic_size[0]), np.uint8)
    Table_mask = cv2.fillConvexPoly(Table_mask, hull, 255)
    Table = cv2.bitwise_and(OriginPic, OriginPic, mask=Table_mask)

    #cornermask
    retval = cv2.minAreaRect(hull)
    x_center = retval[0][0]
    y_center = retval[0][1]
    Matrix2D = cv2.getRotationMatrix2D((x_center, y_center), 0, 1.2) #將mask放大1.2倍當作特徵點可能出現的範圍
    cornermask = cv2.warpAffine(Table_mask, Matrix2D, Pic_size)
    #cv2.imshow('cornermask', cornermask)

    #Convert Top-Down view
    txtheader = 'Picture'+str(number_of_picture)
    image_TopDown, connectline, M = TDcvt1.converttopdown(Hue, Saturation, Value, Table_mask, cornermask, OriginPic, OutputPic, Pic_size, True, True, automode=True)
    #standard_size = (1450, 815) #標準球桌比例
    #image_TopDown = cv2.resize(image_TopDown, standard_size)
    cv2.imwrite('firstRd.jpeg', image_TopDown)


    #########
    #Round 2
    #########
    Pic_size=(1344, 1008)
    #Pic_size = (1450, 815) #標準球桌比例
    OriginPic = cv2.resize(image_TopDown, Pic_size)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image',OnMouseAction)
    cv2.imshow('image',OriginPic)
    cv2.waitKey()
    cv2.destroyAllWindows()

    #start_time = datetime.datetime.now()

    #HSV transfrom
    OriginPic_HSV = cv2.cvtColor(OriginPic, cv2.COLOR_BGR2HSV)
    #讀取滑鼠點擊位置的HSV值
    Hue = int(OriginPic_HSV[y_index][x_index][0])
    Hue_lower = Hue-10
    Hue_upper = Hue+10
    Saturation = int(OriginPic_HSV[y_index][x_index][1])
    Saturation_lower = Saturation-50
    Saturation_upper = Saturation+50
    Value = int(OriginPic_HSV[y_index][x_index][2])
    Value_lower = Value-50
    Value_upper = Value+50

    #只留下顏色跟球桌相似的部分
    mask = cv2.inRange(OriginPic_HSV, (Hue_lower, 70, 50), (Hue_upper, 255, 255))
    ROI = cv2.bitwise_and(OriginPic, OriginPic, mask=mask)
    #cv2.imshow('ROI', ROI)
    #cv2.waitKey()

    #binarization
    ROI_gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    triThe ,ROI_Binary = cv2.threshold(ROI_gray, 0, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3))
    ROI_Binary = cv2.erode(ROI_Binary, kernel, iterations=3)
    ROI_Binary = cv2.dilate(ROI_Binary, kernel, iterations=3)

    #抓輪廓
    contours, hierarchy = cv2.findContours(ROI_Binary, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    area = np.zeros(len(contours))
    for i in range(len(contours)):
        area[i]=cv2.contourArea(contours[i])
    index = np.argmax(area)

    #繪製輪廓
    OutputPic = OriginPic*1
    #cv2.drawContours(OutputPic, contours, index , 255, -1)

    #取凸多邊形
    hull = cv2.convexHull(contours[index])
    Table_mask = np.zeros((Pic_size[1], Pic_size[0]), np.uint8)
    Table_mask = cv2.fillConvexPoly(Table_mask, hull, 255)
    Table = cv2.bitwise_and(OriginPic, OriginPic, mask=Table_mask)

    #cornermask
    retval = cv2.minAreaRect(hull)
    x_center = retval[0][0]
    y_center = retval[0][1]
    Matrix2D = cv2.getRotationMatrix2D((x_center, y_center), 0, 1.2) #將mask放大1.2倍當作特徵點可能出現的範圍
    cornermask = cv2.warpAffine(Table_mask, Matrix2D, Pic_size)
    #cv2.imshow('cornermask', cornermask)

    #Convert Top-Down view
    start = time.time()
    txtheader = 'Picture'+str(number_of_picture)
    image_TopDown1, image_TopDown2, connectline = TDcvt2.converttopdown(Hue, Saturation, Value, Table_mask, cornermask, OriginPic, OutputPic, Pic_size, True, True, automode=True)
    image_TopDown1_gray = cv2.cvtColor(image_TopDown1, cv2.COLOR_BGR2GRAY)
    image_TopDown2_gray = cv2.cvtColor(image_TopDown2, cv2.COLOR_BGR2GRAY)
    pred1_1 = model.predict(image_TopDown1_gray[width_begin:width_end][:][:].reshape(1, 40, 1344, 1))
    pred1_2 = model.predict(image_TopDown1_gray[-width_end:-width_begin][:][:].reshape(1, 40, 1344, 1))
    pred2_1 = model.predict(image_TopDown2_gray[width_begin:width_end][:][:].reshape(1, 40, 1344, 1))
    pred2_2 = model.predict(image_TopDown2_gray[-width_end:-width_begin][:][:].reshape(1, 40, 1344, 1))
    print(time.time()-start)
    
    #cv2.imwrite('Outputs/P'+str(number_of_picture)+'TDInit.jpeg', image_TopDown2)
    if(pred1_1 + pred1_2 > pred2_1+ pred2_2): 
        image_TopDown = image_TopDown1
        print('Ultilize Pred 1')
    elif(pred1_1 + pred1_2 < pred2_1 + pred2_2):
        image_TopDown = image_TopDown2
        print('Ultilize Pred 2')
    else:
       image_TopDown = image_TopDown1 
       print('Ultilize Pred 1')  

    print('pred1_1: '+str(pred1_1))
    print('pred1_2: '+str(pred1_2))
    print('pred2_1: '+str(pred2_1))
    print('pred2_2: '+str(pred2_2))

    cv2.imshow('Longside1-1', image_TopDown1_gray[width_begin:width_end][:][:])
    cv2.imshow('Longside1-2', image_TopDown1_gray[-width_end:-width_begin][:][:])
    cv2.imshow('Longside2-1', image_TopDown2_gray[width_begin:width_end][:][:])
    cv2.imshow('Longside2-2', image_TopDown2_gray[-width_end:-width_begin][:][:])
    #cv2.imwrite('Outputs/P'+str(number_of_picture)+'L1-1.jpeg', image_TopDown1_gray[width_begin:width_end][:][:])
    #cv2.imwrite('Outputs/P'+str(number_of_picture)+'L1-2.jpeg', image_TopDown1_gray[-width_end:-width_begin][:][:])
    #cv2.imwrite('Outputs/P'+str(number_of_picture)+'L2-1.jpeg', image_TopDown2_gray[width_begin:width_end][:][:])
    #cv2.imwrite('Outputs/P'+str(number_of_picture)+'L2-2.jpeg', image_TopDown2_gray[-width_end:-width_begin][:][:])
    standard_size = (1450, 815)
    image_TopDown = cv2.resize(image_TopDown, standard_size)

    cv2.imshow('TopDownS', image_TopDown)
    cv2.imwrite('Outputs/Trans_Img'+str(number_of_picture)+'.jpeg', image_TopDown)
    #cv2.imwrite('Output.jpeg', image_TopDown)
    c = cv2.waitKey()
    cv2.destroyAllWindows()
keras.backend.clear_session()