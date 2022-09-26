import numpy as np
import cv2
#ROImask is mask of table
#cornermask is the range of feature.
def converttopdown(InputPic, ROImask, cornermask, Pic_size=(1344, 1008)):
    origin_pic = InputPic*1
    kernel = np.ones((3,3), np.uint8)
    edge = cv2.morphologyEx(ROImask, cv2.MORPH_GRADIENT, kernel, iterations=1)
    #cv2.imshow("edge", edge)
    #cv2.waitKey()
    lines = cv2.HoughLinesP(edge, 1, np.pi/180, 1, minLineLength=100, maxLineGap=8) #找尋edge上可能的直線
    line_image_before = ROImask*0
    for line in lines:
        x1,y1,x2,y2 = line[0]
        z1 = x1 - x2
        z2 = y1 - y2
        cv2.line(line_image_before, (x1-20*z1,y1-20*z2), (x2+20*z1,y2+20*z2), 255, 3)
    #cv2.imshow("line", line_image_before)
    #cv2.waitKey()
    #merge the lines which are similar.
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
           
    corners = cv2.goodFeaturesToTrack(connectline, 5, 0.1, 50) #將對角交點找出來
    #print(corners)
    #find the diagonal intersection.
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
    #discriminate diagnal relation.
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
    x0, y0 = pts1[0]
    x1, y1 = pts1[1]
    x2, y2 = pts1[2]
    x3, y3 = pts1[3]
    
    #find the pocket.
    totx_red = 0
    toty_red = 0
    cut = 5
    for i in range(1, cut):
        x = int(x0 + i*(x1-x0)/cut)
        y = int(y0 + i*(y1-y0)/cut)
        xx = int(x3 + i*(x2-x3)/cut)
        yy = int(y3 + i*(y2-y3)/cut)
        totx_red=totx_red+(x-xx)
        toty_red=toty_red+(y-yy)
                           
    totx_blue = 0
    toty_blue = 0
    for i in range(1, cut):
        x = int(x0 + i*(x3-x0)/cut)
        y = int(y0 + i*(y3-y0)/cut)
        xx = int(x1 + i*(x2-x1)/cut)
        yy = int(y1 + i*(y2-y1)/cut)
        totx_blue = totx_blue + (x-xx)
        toty_blue = toty_blue + (y-yy)
        
    try:
        slopeB = float(toty_red)/totx_red
    except:
        slopeB = float(toty_red)/(totx_red+0.1)
    try:
        slopeA = float(toty_blue)/totx_blue
    except:
        slopeA = float(toty_blue)/(totx_blue+0.1)

    try:
        slope_0to3 = float(y3-y0)/(x3-x0)
    except:
        slope_0to3 = float(y3-y0)/(x3-x0+0.1)
    try:
        slope_1to2 = float(y2-y1)/(x2-x1)
    except:
        slope_1to2 = float(y2-y1)/(x2-x1+0.1)
    if abs(slope_0to3) > 50:
        slope_0to3 = 50
    if abs(slope_1to2) > 50:
        slope_1to2 = 50
    maybe_hole12_minimun = 255
    for i in range(-2, 3):
        intersection_mask = ROImask*0
        A = slopeA*(1+0.25*i)
        cv2.line(intersection_mask, (0, int(y0 + slope_0to3*(-x0))), (Pic_size[0], int(y0 + slope_0to3*(Pic_size[0]-x0))), 255, 3)
        cv2.line(intersection_mask, (0, int(y1 + slope_1to2*(-x1))), (Pic_size[0], int(y1 + slope_1to2*(Pic_size[0]-x1))), 255, 3)
        cv2.line(intersection_mask, (0, int(y4 + A*(-x4))), (Pic_size[0], int(y4 + A*(Pic_size[0]-x4))), 255, 3)

        intersections = cv2.goodFeaturesToTrack(intersection_mask, 2, 0.1, 50, mask=cornermask) 
        x, y = intersections[0].ravel()
        xx, yy = intersections[1].ravel()
        
        maybe_hole_mask1 = ROImask*0
        maybe_hole_mask2 = ROImask*0
        cv2.circle(maybe_hole_mask1, (int(x), int(y)), 5, 255, -1)
        cv2.circle(maybe_hole_mask2, (int(xx), int(yy)), 5, 255, -1)
        maybe_hole1 = cv2.bitwise_and(origin_pic, origin_pic, mask=maybe_hole_mask1)
        maybe_hole2 = cv2.bitwise_and(origin_pic, origin_pic, mask=maybe_hole_mask2)
        maybe_hole1_mean = np.mean(maybe_hole1[maybe_hole_mask1==255].reshape(-1))
        maybe_hole2_mean = np.mean(maybe_hole2[maybe_hole_mask2==255].reshape(-1))
        maybe_hole12_minimun = np.min((maybe_hole1_mean, maybe_hole2_mean, maybe_hole12_minimun))

    try:
        slope_0to1 = float(y1-y0)/(x1-x0)
    except:
        slope_0to1 = float(y1-y0)/(x1-x0+0.1)
    try:
        slope_3to2 = float(y2-y3)/(x2-x3)
    except:
        slope_3to2 = float(y2-y3)/(x2-x3+0.1)
    if abs(slope_0to1) > 50:
        slope_0to1 = 50
    if abs(slope_3to2) > 50:
        slope_3to2 = 50
    maybe_hole34_minimun = 255
    for i in range(-2, 3):
        intersection_mask = ROImask*0
        B = slopeB*(1+0.25*i)
        cv2.line(intersection_mask, (0, int(y0 + slope_0to1*(-x0))), (Pic_size[0], int(y0 + slope_0to1*(Pic_size[0]-x0))), 255, 3)
        cv2.line(intersection_mask, (0, int(y3 + slope_3to2*(-x3))), (Pic_size[0], int(y3 + slope_3to2*(Pic_size[0]-x3))), 255, 3)
        cv2.line(intersection_mask, (0, int(y4 + B*(-x4))), (Pic_size[0], int(y4 + B*(Pic_size[0]-x4))), 255, 3)

        intersections = cv2.goodFeaturesToTrack(intersection_mask, 2, 0.1, 50, mask=cornermask)
        x, y = intersections[0].ravel()
        xx, yy = intersections[1].ravel()

        maybe_hole_mask3 = ROImask*0
        maybe_hole_mask4 = ROImask*0
        cv2.circle(maybe_hole_mask3, (int(x), int(y)), 5, 255, -1)
        cv2.circle(maybe_hole_mask4, (int(xx), int(yy)), 5, 255, -1)
        maybe_hole3 = cv2.bitwise_and(origin_pic, origin_pic, mask=maybe_hole_mask3)
        maybe_hole4 = cv2.bitwise_and(origin_pic, origin_pic, mask=maybe_hole_mask4)
        maybe_hole3_mean = np.mean(maybe_hole3[maybe_hole_mask3==255].reshape(-1))
        maybe_hole4_mean = np.mean(maybe_hole4[maybe_hole_mask4==255].reshape(-1))
        maybe_hole34_minimun = np.min((maybe_hole3_mean, maybe_hole4_mean, maybe_hole34_minimun))

    #decide rotation direction
    if(maybe_hole12_minimun >= maybe_hole34_minimun):
        #print('hole on the greenline2')
        if(x0 > x1):
            if(y0 > y3):
                pts2 = np.array([[Pic_size[0], Pic_size[1]], [0, Pic_size[1]], [0, 0], [Pic_size[0], 0]], dtype=np.float32)
            elif(y0 < y3):
                pts2 = np.array([[Pic_size[0], 0], [0, 0], [0, Pic_size[1]], [Pic_size[0], Pic_size[1]]], dtype=np.float32)
            else:
                if(y0 > y1):
                    if(x0 > x3):
                        pts2 = np.array([[Pic_size[0], 0], [0, 0], [0, Pic_size[1]], [Pic_size[0], Pic_size[1]]], dtype=np.float32)
                    if(x0 < x3):
                        pts2 = np.array([[0, 0], [Pic_size[0], 0], [Pic_size[0], Pic_size[1]], [0, Pic_size[1]]], dtype=np.float32)
                else:
                    if(x0 > x3):
                        pts2 = np.array([[0, 0], [Pic_size[0], 0], [Pic_size[0], Pic_size[1]], [0, Pic_size[1]]], dtype=np.float32)
                    if(x0 < x3):
                        pts2 = np.array([[Pic_size[0], 0], [0, 0], [0, Pic_size[1]], [Pic_size[0], Pic_size[1]]], dtype=np.float32)
        elif(x0 < x1):
            if(y0 > y3):
                pts2 = np.array([[0, Pic_size[1]], [Pic_size[0], Pic_size[1]], [Pic_size[0], 0], [0, 0]], dtype=np.float32)
            elif(y0 < y3):
                pts2 = np.array([[0, 0], [Pic_size[0], 0], [Pic_size[0], Pic_size[1]], [0, Pic_size[1]]], dtype=np.float32)
            else:
                if(y0 > y1):
                    if(x0 > x3):
                        pts2 = np.array([[Pic_size[0], 0], [0, 0], [0, Pic_size[1]], [Pic_size[0], Pic_size[1]]], dtype=np.float32)
                    if(x0 < x3):
                        pts2 = np.array([[0, 0], [Pic_size[0], 0], [Pic_size[0], Pic_size[1]], [0, Pic_size[1]]], dtype=np.float32)
                else:
                    if(x0 > x3):
                        pts2 = np.array([[0, 0], [Pic_size[0], 0], [Pic_size[0], Pic_size[1]], [0, Pic_size[1]]], dtype=np.float32)
                    if(x0 < x3):
                        pts2 = np.array([[Pic_size[0], 0], [0, 0], [0, Pic_size[1]], [Pic_size[0], Pic_size[1]]], dtype=np.float32)
        else:
            if(y0 > y1):
                if(x0 > x3):
                    pts2 = np.array([[Pic_size[0], 0], [0, 0], [0, Pic_size[1]], [Pic_size[0], Pic_size[1]]], dtype=np.float32)
                if(x0 < x3):
                    pts2 = np.array([[0, 0], [Pic_size[0], 0], [Pic_size[0], Pic_size[1]], [0, Pic_size[1]]], dtype=np.float32)
            else:
                if(x0 > x3):
                    pts2 = np.array([[0, 0], [Pic_size[0], 0], [Pic_size[0], Pic_size[1]], [0, Pic_size[1]]], dtype=np.float32)
                if(x0 < x3):
                    pts2 = np.array([[Pic_size[0], 0], [0, 0], [0, Pic_size[1]], [Pic_size[0], Pic_size[1]]], dtype=np.float32)
    else:
        #print('hole on the greenline1')
        if(x0 > x3):
            if(y0 > y1):
                pts2 = np.array([[Pic_size[0], Pic_size[1]], [Pic_size[0], 0], [0, 0], [0, Pic_size[1]]], dtype=np.float32)
            elif(y0 < y1):
                pts2 = np.array([[Pic_size[0], 0], [Pic_size[0], Pic_size[1]], [0, Pic_size[1]], [0, 0]], dtype=np.float32)
            else:
                if(y0 > y3):
                    if(x0 > x1):
                        pts2 = np.array([[Pic_size[0], 0], [Pic_size[0], Pic_size[1]], [0, Pic_size[1]], [0, 0]], dtype=np.float32)
                    if(x0 < x1):
                        pts2 = np.array([[0, 0], [0, Pic_size[1]], [Pic_size[0], Pic_size[1]], [Pic_size[0], 0]], dtype=np.float32)
                else:
                    if(x0 > x1):
                        pts2 = np.array([[0, 0], [0, Pic_size[1]], [Pic_size[0], Pic_size[1]], [Pic_size[0], 0]], dtype=np.float32)
                    if(x0 < x1):
                        pts2 = np.array([[Pic_size[0], 0], [Pic_size[0], Pic_size[1]], [0, Pic_size[1]], [0, 0]], dtype=np.float32)
        elif(x0 < x3):
            if(y0 > y1):
                pts2 = np.array([[0, Pic_size[1]], [0, 0], [Pic_size[0], 0], [Pic_size[0], Pic_size[1]]], dtype=np.float32)
            elif(y0 < y1):
                pts2 = np.array([[0, 0], [0, Pic_size[1]], [Pic_size[0], Pic_size[1]], [Pic_size[0], 0]], dtype=np.float32)
            else:
                if(y0 > y3):
                    if(x0 > x1):
                        pts2 = np.array([[Pic_size[0], 0], [Pic_size[0], Pic_size[1]], [0, Pic_size[1]], [0, 0]], dtype=np.float32)
                    if(x0 < x1):
                        pts2 = np.array([[0, 0], [0, Pic_size[1]], [Pic_size[0], Pic_size[1]], [Pic_size[0], 0]], dtype=np.float32)
                else:
                    if(x0 > x1):
                        pts2 = np.array([[0, 0], [0, Pic_size[1]], [Pic_size[0], Pic_size[1]], [Pic_size[0], 0]], dtype=np.float32)
                    if(x0 < x1):
                        pts2 = np.array([[Pic_size[0], 0], [Pic_size[0], Pic_size[1]], [0, Pic_size[1]], [0, 0]], dtype=np.float32)
        else:
            if(y0 > y3):
                if(x0 > x1):
                    pts2 = np.array([[Pic_size[0], 0], [Pic_size[0], Pic_size[1]], [0, Pic_size[1]], [0, 0]], dtype=np.float32)
                if(x0 < x1):
                    pts2 = np.array([[0, 0], [0, Pic_size[1]], [Pic_size[0], Pic_size[1]], [Pic_size[0], 0]], dtype=np.float32)
            else:
                if(x0 > x1):
                    pts2 = np.array([[0, 0], [0, Pic_size[1]], [Pic_size[0], Pic_size[1]], [Pic_size[0], 0]], dtype=np.float32)
                if(x0 < x1):
                    pts2 = np.array([[Pic_size[0], 0], [Pic_size[0], Pic_size[1]], [0, Pic_size[1]], [0, 0]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts1, pts2)
    image_TopDown = cv2.warpPerspective(InputPic, M, Pic_size)

    return image_TopDown, M, Pic_size