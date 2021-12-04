import cv2
import numpy as np
import TDF_first as TDcvt1
import TDF_second as TDcvt2
def cvtTD(InputPic, InputPic_size = (1600, 900), OutputPic_size = (1344, 1008), manual = 0):
    InputPic = cv2.resize(InputPic, InputPic_size)
    OriginPic = InputPic

    if manual > 0:
        def OnMouseAction(event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if(OnMouseAction.click < 4):
                    OnMouseAction.corners[OnMouseAction.click][0]=x
                    OnMouseAction.corners[OnMouseAction.click][1]=y
                    cv2.circle(OnMouseAction.pic_for_painting, (x, y), 3, (255, 255, 255), -1)
                    cv2.imshow('image', OnMouseAction.pic_for_painting)
                    OnMouseAction.click += 1
        OnMouseAction.click = 0
        OnMouseAction.corners = np.zeros((4, 2), dtype=np.float32)
        OnMouseAction.pic_for_painting = InputPic*1 #static declare
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', OnMouseAction)
        cv2.imshow('image', OnMouseAction.pic_for_painting)
        cv2.waitKey()
        cv2.destroyAllWindows()
        pts2 = np.array([[0, 0], [OutputPic_size[0], 0], [OutputPic_size[0], OutputPic_size[1]], [0, OutputPic_size[1]]], dtype=np.float32)
        M1 = cv2.getPerspectiveTransform(OnMouseAction.corners, pts2)
        Pic_size1 = OutputPic_size
        M2 = cv2.getPerspectiveTransform(pts2, pts2)
        Pic_size2 = OutputPic_size
    else:
        def OnMouseAction(event,x,y,flags,param):
            global x_index,y_index
            if event == cv2.EVENT_LBUTTONDOWN:
                x_index=x
                y_index=y
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',OnMouseAction)
        cv2.imshow('image', OriginPic)
        cv2.waitKey()
        cv2.destroyAllWindows()
        print("x_index = "+str(x_index)+"\ny_index = "+str(y_index))

        #HSV transfrom
        OriginPic_HSV = cv2.cvtColor(OriginPic, cv2.COLOR_BGR2HSV)
        #read HSV value
        Hue = int(OriginPic_HSV[y_index][x_index][0])
        Hue_lower = Hue-10
        Hue_upper = Hue+10
        
        #table catch roughly
        mask = cv2.inRange(OriginPic_HSV, (Hue_lower, 70, 50), (Hue_upper, 255, 255))
        ROI = cv2.bitwise_and(OriginPic, OriginPic, mask=mask)

        #binarization
        ROI_gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        triThe ,ROI_Binary = cv2.threshold(ROI_gray, 0, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3))
        ROI_Binary = cv2.erode(ROI_Binary, kernel, iterations=3)
        ROI_Binary = cv2.dilate(ROI_Binary, kernel, iterations=3)

        #find contour
        contours, hierarchy = cv2.findContours(ROI_Binary, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        area = np.zeros(len(contours))
        for i in range(len(contours)):
            area[i]=cv2.contourArea(contours[i])
        index = np.argmax(area)

        hull = cv2.convexHull(contours[index])
        Table_mask = np.zeros((InputPic_size[1], InputPic_size[0]), np.uint8)
        Table_mask = cv2.fillConvexPoly(Table_mask, hull, 255)

        #cornermask
        retval = cv2.minAreaRect(hull)
        x_center = retval[0][0]
        y_center = retval[0][1]
        Matrix2D = cv2.getRotationMatrix2D((x_center, y_center), 0, 1.2) #將mask放大1.2倍當作特徵點可能出現的範圍
        cornermask = cv2.warpAffine(Table_mask, Matrix2D, InputPic_size)

        #Convert Top-Down view
        image_TopDown, M1, Pic_size1 = TDcvt1.converttopdown(OriginPic, Table_mask, cornermask, InputPic_size)

        #########
        #Round 2
        #########

        OriginPic = image_TopDown
        #HSV transfrom
        OriginPic_HSV = cv2.cvtColor(OriginPic, cv2.COLOR_BGR2HSV)

        #table catch roughly
        mask = cv2.inRange(OriginPic_HSV, (Hue_lower, 70, 50), (Hue_upper, 255, 255))
        ROI = cv2.bitwise_and(OriginPic, OriginPic, mask=mask)

        #binarization
        ROI_gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        triThe ,ROI_Binary = cv2.threshold(ROI_gray, 0, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3))
        ROI_Binary = cv2.erode(ROI_Binary, kernel, iterations=3)
        ROI_Binary = cv2.dilate(ROI_Binary, kernel, iterations=3)

        #find contour
        contours, hierarchy = cv2.findContours(ROI_Binary, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        area = np.zeros(len(contours))
        for i in range(len(contours)):
            area[i]=cv2.contourArea(contours[i])
        index = np.argmax(area)

        #get convexhull
        hull = cv2.convexHull(contours[index])
        Table_mask = np.zeros((OriginPic.shape[0], OriginPic.shape[1]), np.uint8)
        Table_mask = cv2.fillConvexPoly(Table_mask, hull, 255)

        #cornermask
        retval = cv2.minAreaRect(hull)
        x_center = retval[0][0]
        y_center = retval[0][1]
        Matrix2D = cv2.getRotationMatrix2D((x_center, y_center), 0, 1.2) #將mask放大1.2倍當作特徵點可能出現的範圍
        cornermask = cv2.warpAffine(Table_mask, Matrix2D, (Table_mask.shape[1], Table_mask.shape[0]))
        
        #Convert Top-Down view
        image_TopDown, M2, Pic_size2 = TDcvt2.converttopdown(OriginPic, Table_mask, cornermask, OutputPic_size)

    return M1, Pic_size1, M2, Pic_size2, image_TopDown