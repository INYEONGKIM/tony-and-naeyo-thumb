# -*- coding: utf-8 -*-

import cv2
import signal
import os
import numpy as np
from keras.models import load_model

# MAC : SIGPROF 27
# ubuntu : SIGPROF 27
def signalCameraOnHandler(signum, frame):
    print signum, "CAM"
    global img_flag
    global cap

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    img_flag = "CAM"


# MAC : SIGINFO 29
# ubuntu : SIGRTMAX	64
def signalMapHandler(signum, frame):
    print signum, "MAP"
    global img_flag
    global cap

    cap = cv2.VideoCapture('naeyo_map.gif')
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    img_flag = "MAP"


# MAC : SIGUSR1 30
# Ubuntu : SIGUSR1 10
def signalSmileFaceHandler(signum, frame):
    print signum, "SMILE"
    global img_flag
    global cap

    cap = cv2.VideoCapture('naeyo_smile_blur.gif')
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    img_flag = "SMILE"


# MAC : SIGUSR2 31
# ubuntu : SIGUSR2 12
def signalDefaultFaceHandler(signum, frame):
    print signum, "NORMAL"
    global img_flag
    global cap

    cap = cv2.VideoCapture('naeyo_normal_blur.gif')
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    img_flag = "NORMAL"

cap = cv2.VideoCapture('naeyo_smile_blur.gif')
img_flag = "SMILE"

def Main():
    global cap, img_flag

    kernel = np.ones((3, 3), np.uint8)

    cap = cv2.VideoCapture(0)
    img_flag = "CAM"

    cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)

    model = load_model('thr_final_model2.h5')

    print "pid : ", os.getpid()

    reset_flag = False

    thumbs_up_guess_stack = 0
    smile_face_stack = 0
    camera_page_stack = 0
    map_page_stack = 0

    while True:
        try:
            ret, frame = cap.read()

            # for end of gif
            if not ret:
                if img_flag == "NORMAL":
                    cap = cv2.VideoCapture('naeyo_normal_blur.gif')
                    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

                elif img_flag == "SMILE":
                    cap = cv2.VideoCapture('naeyo_smile_blur.gif')
                    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

                elif img_flag == "MAP":
                    cap = cv2.VideoCapture('naeyo_map.gif')
                    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

                ret, frame = cap.read()

            if img_flag == "CAM":
                camera_page_stack += 1

                frame = cv2.flip(frame, 1)
                roi = frame[100:300, 100:300]
                roi_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
                roi_s = roi[100:140, 80:120]
                roi_s_ycrcb = cv2.cvtColor(roi_s, cv2.COLOR_BGR2YCrCb)
                cv2.rectangle(frame, (125, 100), (275, 300), (0, 0, 255), 0)

                y = roi_s_ycrcb[20][20][0]
                cr = roi_s_ycrcb[20][20][1]
                cb = roi_s_ycrcb[20][20][2]

                if y > 180:
                    lower_skin = np.array([y - 20, cr - 10, cb - 10], dtype=np.uint8)
                    upper_skin = np.array([y + 20, cr + 10, cb + 10], dtype=np.uint8)
                    if y > 230:
                        lower_skin = np.array([y - 20, cr - 10, cb - 10], dtype=np.uint8)
                        upper_skin = np.array([250, cr + 10, cb + 10], dtype=np.uint8)

                    mask = cv2.inRange(roi_ycrcb, lower_skin, upper_skin)
                    mask = cv2.erode(mask, kernel, iterations=1)
                    mask = cv2.dilate(mask, kernel, iterations=1)
                    # blur the image
                    mask = cv2.GaussianBlur(mask, (5, 5), 0)

                    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    areas = [cv2.contourArea(c) for c in contours]
                    max_index = np.argmax(areas)
                    cnt = contours[max_index]
                    cv2.drawContours(mask, cnt, -1, (255, 0, 0), 5)

                    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    img = cv2.resize(mask, None, fx=0.25, fy=0.25)
                    # img = cv2.resize(mask, None, fx=50 / mask.shape[1], fy=50 / mask.shape[0])

                    img = img / 255

                    test = np.array(img)
                    test = np.expand_dims(test, axis=0)

                    predict = model.predict_classes(test)
                    if predict == 1:
                        thumbs_up_guess_stack += 1

                    # cv2.imshow('roi', mask)
                else:
                    # print 'no skin color'
                    cv2.putText(frame, 'NO SKIN COLOR', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

            elif img_flag == "SMILE":
                smile_face_stack += 1
                cap = cv2.VideoCapture('naeyo_smile_blur.gif')

            elif img_flag == "NORMAL":
                cap = cv2.VideoCapture('naeyo_normal_blur.gif')

            elif img_flag == "MAP":
                map_page_stack += 1
                cap = cv2.VideoCapture('naeyo_map.gif')

            # for signal call
            if thumbs_up_guess_stack >= 6:
                # paging to smile
                os.system("kill -10 " + str(os.getpid()))
                thumbs_up_guess_stack = 0
                reset_flag = True

                cv2.putText(frame, 'Thumbs up!', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

            if smile_face_stack >= 100 or camera_page_stack >= 150:
                # paging to map
                os.system("kill -64 " + str(os.getpid()))
                reset_flag = True

            # now threshold == 300
            if map_page_stack >= 200:
                # paging to camera
                os.system("kill -27 " + str(os.getpid()))
                reset_flag = True

            if reset_flag:
                thumbs_up_guess_stack = 0
                smile_face_stack = 0
                camera_page_stack = 0
                map_page_stack = 0
                reset_flag = False

            cv2.imshow('frame', frame)
            cv2.imshow('roi', mask)

            k = cv2.waitKey(50) & 0xFF
            if k == 27:
                break

        except:
            pass

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Set Signal
    signal.signal(signal.SIGPROF, signalCameraOnHandler)
    signal.signal(signal.SIGRTMAX, signalMapHandler)
    signal.signal(signal.SIGUSR1, signalSmileFaceHandler)
    signal.signal(signal.SIGUSR2, signalDefaultFaceHandler)

    Main()
