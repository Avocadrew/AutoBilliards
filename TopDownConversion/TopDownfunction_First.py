import numpy as np
import cv2
#ROImask為球桌的mask
#cornermask指定特徵點可能所在的範圍
#InputPic為原圖
#OutputPic用來做標記、畫線的圖
#plotcorner決定是否要將特徵點畫出來
#plotline決定是否要將特徵點用線連起來
#savetxt將線上的RGB值存成txt檔
#automode決定是否要手動點選特徵點
def converttopdown(Hue, Saturation, Value, ROImask, cornermask, InputPic, OutputPic, Pic_size=(1920, 1080), plotcorner=False, plotline=False, automode=True):
    origin_pic = InputPic*1
    kernel = np.ones((3,3), np.uint8)
    edge = cv2.morphologyEx(ROImask, cv2.MORPH_GRADIENT, kernel, iterations=1)
    #image_of_interst = cv2.bitwise_and(InputPic, InputPic, mask=ROImask)
    #cv2.imshow('edge', edge)
    #cv2.waitKey()
    if(automode):
        lines = cv2.HoughLinesP(edge, 1, np.pi/180, 1, minLineLength=100, maxLineGap=8) #找尋edge上可能的直線
        line_image_before = ROImask*0
        for line in lines:
            x1,y1,x2,y2 = line[0]
            z1 = x1 - x2
            z2 = y1 - y2
            cv2.line(line_image_before, (x1-20*z1,y1-20*z2), (x2+20*z1,y2+20*z2), 255, 3)
        cv2.imshow('line_image_before.jpeg', line_image_before) #斜率合併之前所得到的線
        cv2.imwrite('line_image_before.jpeg', line_image_before) #斜率合併之前所得到的線
        cv2.waitKey()
        #合併斜率相似的線
        lines = np.array(lines)
        x = len(lines)
        k = -1
        while(1):
            if(k==x):
                break
            else:
                while(1):
                    k = k + 1
                    if(k<x):
                        x1,y1,x2,y2 = lines[k][0]
                        try:
                            slopeL1 = (y1-y2)/(x1-x2)
                        except:
                            slopeL1 = (y1-y2)/(x1-x2+0.01)
                        l = k
                        b = y1 - slopeL1 * x1
                        while(1):
                            l = l + 1
                            if(l<x):
                                xi1,yi1,xi2,yi2 = lines[l][0]
                                try:
                                    slopeL2 = (yi1-yi2)/(xi1-xi2)
                                except:
                                    slopeL2 = (yi1-yi2)/(xi1-xi2+0.01)
                                bi = yi2 - slopeL2 * xi2
                                try:
                                    xintersect = (bi-b)/(slopeL1-slopeL2)
                                except:
                                    xintersect = (bi-b)/(slopeL1-slopeL2+0.01)
                                #yintersect = slopeL2 * xintersect + bi
                                if(xintersect < Pic_size[1] and xintersect > 0 and abs(slopeL1-slopeL2)<=0.2):
                                    #print("Del",slopeL1, slopeL2, xintersect)
                                    lines = np.delete(lines,l,0)
                                    l = l - 1
                                    x = x - 1
                            else:
                                break
                    else:
                        break
        lines = lines.tolist()
        line_image_after = ROImask*0
        for line in lines:
            x1,y1,x2,y2 = line[0]
            z1 = x1 - x2
            z2 = y1 - y2
            cv2.line(line_image_after, (x1-20*z1,y1-20*z2), (x2+20*z1,y2+20*z2), 255, 3)
        #line_image_after = cv2.dilate(line_image_after, kernel, iterations=3)
        #line_image_after = cv2.erode(line_image_after, kernel, iterations=3)
        cv2.imshow('line_image_after', line_image_after) #斜率合併後所得到的線
        cv2.imwrite('line_image_after.jpeg', line_image_after)
        cv2.waitKey()
        cv2.destroyAllWindows()
        corners = cv2.goodFeaturesToTrack(line_image_after, 4, 0.1, 200, mask = cornermask)
        x0, y0 = corners[0].ravel()
        x1, y1 = corners[1].ravel()
        x2, y2 = corners[2].ravel()
        x3, y3 = corners[3].ravel()
        x4, y4 = 960, 540
        connectline = line_image_after*0
        for corner in corners:
            x, y = corner.ravel()
            cv2.line(connectline, (int(x), int(y)), (int(x1), int(y1)), 255, 3)
            cv2.line(connectline, (int(x), int(y)), (int(x2), int(y2)), 255, 3)
            cv2.line(connectline, (int(x), int(y)), (int(x3), int(y3)), 255, 3)
            cv2.line(connectline, (int(x), int(y)), (int(x0), int(y0)), 255, 3)
        cv2.circle(connectline, (int(x0), int(y0)), radius=2, color=(0, 255, 0), thickness=3)
        cv2.circle(connectline, (int(x1), int(y1)), radius=2, color=(0, 255, 0), thickness=3)
        cv2.circle(connectline, (int(x2), int(y2)), radius=2, color=(0, 255, 0), thickness=3)
        cv2.circle(connectline, (int(x3), int(y3)), radius=2, color=(0, 255, 0), thickness=3)
        #cv2.imshow('connectline', connectline) #斜率合併後所得到的線
        #cv2.waitKey()
    
           
    corners = cv2.goodFeaturesToTrack(connectline, 5, 0.1, 50) #將對角交點找出來
    #print(corners)
    #判斷中心點
    for corner in corners:
        x, y = corner.ravel()
        if ((x-x0)**2 + (y-y0)**2) > 500:
            if(((x-x1)**2 + (y-y1)**2) > 500):
                if(((x-x2)**2 + (y-y2)**2) > 500):
                    if(((x-x3)**2 + (y-y3)**2) > 500):
                        x4, y4 = x, y
                        break
    #print(x4, y4)
    pts1 = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32)
    pts1[0] = [x0, y0]
    #判斷對角關係
    try:
        equation1 = abs(y4 - (x4*(float(y0-y1)/float(x0-x1)) + (y0-x0*(float(y0-y1)/float(x0-x1)))))
    except:
        equation1 = abs(y4 - (x4*(float(y0-y1)/float(x0-x1+0.1)) + (y0-x0*(float(y0-y1)/float(x0-x1+0.1)))))
    try:
        equation2 = abs(y4 - (x4*(float(y0-y2)/float(x0-x2)) + (y0-x0*(float(y0-y2)/float(x0-x2)))))
    except:
        equation2 = abs(y4 - (x4*(float(y0-y1)/float(x0-x1+0.1)) + (y0-x0*(float(y0-y1)/float(x0-x1+0.1)))))
    try:
        equation3 = abs(y4 - (x4*(float(y0-y3)/float(x0-x3)) + (y0-x0*(float(y0-y3)/float(x0-x3)))))
    except:
        equation3 = abs(y4 - (x4*(float(y0-y3)/float(x0-x3+0.1)) + (y0-x0*(float(y0-y3)/float(x0-x3+0.1)))))
    diagonal = np.min((equation1, equation2, equation3))
    if(diagonal == equation1):
        #print("equation1 work")
        pts1[1] = [x2, y2]
        pts1[2] = [x1, y1]
        pts1[3] = [x3, y3]
    elif(diagonal == equation2):
        #print("equation2 work")
        pts1[1] = [x1, y1]
        pts1[2] = [x2, y2]
        pts1[3] = [x3, y3]
    elif(diagonal == equation3):
        #print("equation3 work")
        pts1[1] = [x1, y1]
        pts1[2] = [x3, y3]
        pts1[3] = [x2, y2]
    #print(pts1)

    cx = int(pts1[0][0]+pts1[1][0]+pts1[2][0]+pts1[3][0])/4
    cy = int(pts1[0][1]+pts1[1][1]+pts1[2][1]+pts1[3][1])/4
    for i in range(0,4):
        if(pts1[i][0]<cx):
            pts1[i][0]=pts1[i][0]-int(abs(pts1[i][0]-cx)*0.25)
        else:
            pts1[i][0]=pts1[i][0]+int(abs(pts1[i][0]-cx)*0.25)
        if(pts1[i][1]<cy):
            pts1[i][1]=pts1[i][1]-int(abs(pts1[i][1]-cy)*0.25)
        else:
            pts1[i][1]=pts1[i][1]+int(abs(pts1[i][1]-cy)*0.25)
            
        if(pts1[i][0] > Pic_size[0]):
            pts1[i][0] = Pic_size[0]
        elif(pts1[i][0] < 0):
            pts1[i][0] = 0
        if(pts1[i][1] > Pic_size[1]):
            pts1[i][1] = Pic_size[1]
        elif(pts1[i][1] < 0):
            pts1[i][1] = 0
    pts2 = np.array([[0, 0], [Pic_size[0], 0], [Pic_size[0], Pic_size[1]], [0, Pic_size[1]]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    print(M)
    image_TopDown = cv2.warpPerspective(InputPic, M, Pic_size)
    # Fill image with red color(set each pixel to red)
    table_color_canvas = np.zeros((Pic_size[1], Pic_size[0], 3), np.uint8)
    table_color_canvas[:] = (Hue, Saturation, Value)
    table_color_canvas = cv2.cvtColor(table_color_canvas, cv2.COLOR_HSV2BGR)
    
    if(plotcorner):    
        cv2.circle(OutputPic, (int(x0), int(y0)), 5, (0, 255, 0), -1)
        cv2.circle(OutputPic, (int(x1), int(y1)), 5, (0, 255, 0), -1)
        cv2.circle(OutputPic, (int(x2), int(y2)), 5, (0, 255, 0), -1)
        cv2.circle(OutputPic, (int(x3), int(y3)), 5, (0, 255, 0), -1)
        cv2.circle(OutputPic, (int(x4), int(y4)), 5, (0, 0, 255), -1)

    if(plotline):
        cv2.line(OutputPic, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 1)
        cv2.line(OutputPic, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        cv2.line(OutputPic, (int(x2), int(y2)), (int(x3), int(y3)), (0, 0, 255), 1)
        cv2.line(OutputPic, (int(x0), int(y0)), (int(x3), int(y3)), (0, 0, 255), 1)
        cv2.line(OutputPic, (int(x0), int(y0)), (int(x2), int(y2)), (0, 0, 255), 1)
        cv2.line(OutputPic, (int(x1), int(y1)), (int(x3), int(y3)), (0, 0, 255), 1)

    return image_TopDown, connectline, M
