import cv2
import numpy as np
import ConvertTD as cvtTD
import YoloAPI
video = cv2.VideoCapture("videos_selfcollect/IMG_4330.MOV")
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out_video = cv2.VideoWriter('virtual_videos/output.mp4', fourcc, 30, (1344, 1008))
ret ,frame = video.read()
InputPic_size = (1600, 900)
OutputPic_size = (1344, 1008)
while(1):
    M1, Pic_size1, M2, Pic_size2, image_TopDown = cvtTD.cvtTD(frame, InputPic_size, OutputPic_size)
    cv2.imshow('CVT_Result', image_TopDown)
    cv2.waitKey()
    cv2.destroyAllWindows()
    while(1):
        c = input("Enter 'g' if convertion is good\nEnter 'b' redo the convertion\nEnter \"manual\" to mark the corner byself\n")
        if c == 'g' or c == 'b' or c == "manual" or c == "exit":
            break
    if c == 'g' or c == "exit":
        break
    while c == 'manual':
        M1, Pic_size1, M2, Pic_size2, image_TopDown = cvtTD.cvtTD(frame, InputPic_size, OutputPic_size, manual = 1)
        cv2.imshow('CVT_Result', image_TopDown)
        cv2.waitKey()
        cv2.destroyAllWindows()
        while(1):
            c = input("Enter 'g' if convertion is good\nEnter 'b' redo the convertion\nEnter \"auto\" to make the corner automatically\n")
            if c == 'g' or c == 'b' or c == "auto" or c == "exit":
                break
        if c == 'g' or c == "exit" or "auto":
            break
    if c == 'g' or c == "exit":
        break
if(c == 'g'):
    yolo = YoloAPI.opencvYOLO(modeltype="yolov3-tiny", objnames="project\Simulate\Simulate_dataset\Yolo_0925\obj.names",
    weights="project\Simulate\Simulate_dataset\Yolo_0925\weights\yolov3_last.weights", cfg="project\Simulate\Simulate_dataset\Yolo_0925\yolov3_valid.cfg")
    yolo.setScore(0.6)
    yolo.setNMS(0.6)

    #create background
    Background = np.zeros((1008, 1344, 3), dtype=np.uint8)
    Background[:, :, 0] = 147
    Background[:, :, 1] = 219
    Background[:, :, 2] = 139
    cv2.rectangle(Background, (25, 25), (1318, 982), 0, 2)
    cv2.circle(Background, (20, 20), 25, (127, 127, 127), -1)
    cv2.circle(Background, (20, 20), 25, (0, 0, 0), 2)
    cv2.circle(Background, (1324, 20), 25, (127, 127, 127), -1)
    cv2.circle(Background, (1324, 20), 25, (0, 0, 0), 2)
    cv2.circle(Background, (1324, 988), 25, (127, 127, 127), -1)
    cv2.circle(Background, (1324, 988), 25, (0, 0, 0), 2)
    cv2.circle(Background, (20, 988), 25, (127, 127, 127), -1)
    cv2.circle(Background, (20, 988), 25, (0, 0, 0), 2)
    cv2.circle(Background, (672, 10), 25, (127, 127, 127), -1)
    cv2.circle(Background, (672, 10), 25, (0, 0, 0), 2)
    cv2.circle(Background, (672, 998), 25, (127, 127, 127), -1)
    cv2.circle(Background, (672, 998), 25, (0, 0, 0), 2)

    count = 0
    total_frame = 900
    while(count != total_frame):
        ret, frame = video.read()
        frame = cv2.resize(frame, InputPic_size)
        out = cv2.warpPerspective(frame, M1, Pic_size1)
        out = cv2.warpPerspective(out, M2, Pic_size2)
        VirTable = Background*1
        yolo.getObject(out, drawBox=False)
        pred_coord_label = yolo.object_coord_and_label()
        for item in pred_coord_label:
            x = item[0]
            y = item[1]
            label = item[2]
            ballcolor = (0, 0, 0)
            if label == "brown":
                ballcolor = (113, 141, 182)
            elif label == "white":
                ballcolor = (255, 255, 255)
            elif label == "red":
                ballcolor = (25, 25, 225)
            elif label == "black":
                ballcolor = (150, 150, 150)
            elif label == "orange":
                ballcolor = (10, 110, 240)
            elif label == "yellow stripe":
                ballcolor = (50, 220, 220)
            elif label == "green":
                ballcolor = (35, 145, 5)
            elif label == "pink":
                ballcolor = (220, 110, 220)
            elif label == "blue":
                ballcolor = (240, 50, 50)
            elif label == "yellow":
                ballcolor = (50, 220, 220)
            
            if label != "yellow stripe":
                cv2.circle(VirTable, (x, y), 20, ballcolor, -1)
                cv2.circle(VirTable, (x, y), 20, (0, 0, 0), 2)
                cv2.circle(VirTable, (x, y), 5, (0, 0, 0), 2)
            else:
                cv2.circle(VirTable, (x, y), 20, ballcolor, -1)
                cv2.circle(VirTable, (x, y), 12, (255, 255, 255), -1)
                cv2.circle(VirTable, (x, y), 20, (0, 0, 0), 2)
                cv2.circle(VirTable, (x, y), 5, (0, 0, 0), 2)
        out_video.write(VirTable)
        count+=1
        print(count)
video.release()
out_video.release()