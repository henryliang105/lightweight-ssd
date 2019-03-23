import numpy as np
import cv2
import sys

cap = cv2.VideoCapture('/home/ubuntu/Videos/haitech/test/day/test6.avi')
write_path = '/home/ubuntu/Videos/haitech/test/day_test_data/test6/'

ret, frame = cap.read()

#cap.set(cv2.CAP_PROP_POS_FRAMES, 800)
ori_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
ori_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

img_width = 500
img_height = 375

print (ori_width, ori_height)

while(cap.isOpened()):
    ret, frame = cap.read()
    framePos = cap.get(cv2.CAP_PROP_POS_FRAMES)

    back_view = frame[5 : int(ori_height / 2 - 5), int(ori_width / 2 + 12) : int(ori_width) - 5]
    left_view = frame[int(ori_height / 2 + 5) : int(ori_height), 5 : int(ori_width / 2 - 12)]
    right_view = frame[int(ori_height / 2 + 5) : int(ori_height), int(ori_width / 2 + 12) : int(ori_width) - 5]
    
    #frame = cv2.resize(frame, (960, 540))
    cv2.imshow("result", frame)
    cv2.imshow("back view", back_view)
    cv2.imshow("left view", left_view)
    cv2.imshow("right view", right_view)


    if framePos % 150 == 0:
        back_view = cv2.resize(back_view, (img_width, img_height)) 
        left_view = cv2.resize(left_view, (img_width, img_height)) 
        right_view = cv2.resize(right_view, (img_width, img_height)) 
        cv2.imwrite(write_path + 'b' + str(framePos) + '.jpg', back_view)
        cv2.imwrite(write_path + 'l' + str(framePos) + '.jpg', left_view)
        cv2.imwrite(write_path + 'r' + str(framePos) + '.jpg', right_view)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()