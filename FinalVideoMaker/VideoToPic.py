import numpy as np
from cv2 import cv2
# 檔名後面的數字開始點 ex:number=20 儲存的圖片檔名為 xxxxx20.jpg 依序往後
number = 201
# 要save的圖片數量
Save_Number_of_Picture = 600
# 每幾幀存一張圖 
frame_leap = 4
# 從第幾秒開始擷取 單位為秒
start_time = 32
video = cv2.VideoCapture('virtual_videos/IMG_4330_TDV.mp4')
video_time = video.get(cv2.CAP_PROP_POS_MSEC)
video_time = video_time + start_time*1000
retval = video.set(cv2.CAP_PROP_POS_MSEC, video_time)
start_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
for i in range(Save_Number_of_Picture):
    retval = video.set(cv2.CAP_PROP_POS_FRAMES, start_frame+i*frame_leap)
    ret, frame = video.read()
    file_name = 'VideoPics/IMG_4330_TDV_P'+str(number)+'.jpg'
    cv2.imwrite(file_name, frame)
    number += 1
video.release()
cv2.destroyAllWindows()