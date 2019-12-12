#keras 2.3.0
#tensorflow 1.14.0
#tensorflow-estimator 1.14.0
####make tensorflow, estimator version same

import cv2
import numpy as np
from keras.models import load_model

def Main():
    categories = ['nothing_thr', 'thumbsup_thr']
    kernel = np.ones((3, 3), np.uint8)
    cap = cv2.VideoCapture(0)
    model = load_model('thr_final_model2.h5')
    count_list = []
    while 1:
        try:
            ret, frame = cap.read()
            print frame.shape
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

                _,contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                areas = [cv2.contourArea(c) for c in contours]
                max_index = np.argmax(areas)
                cnt = contours[max_index]
                cv2.drawContours(mask, cnt, -1, (255, 0, 0), 5)

                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                img = cv2.resize(mask, None, fx=0.25, fy=0.25)
#                img = cv2.resize(mask, None, fx=50 / mask.shape[1], fy=50 / mask.shape[0])

                img = img / 255

                test = np.array(img)
                test = np.expand_dims(test, axis=0)

                predict = model.predict_classes(test)

                count_list.append(predict[0])
                print count_list
                if len(count_list) == 4:
                    if count_list.count(1) == 4:
                        cv2.putText(frame, 'Thumbs up!', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3,
                                    cv2.LINE_AA)
                    if count_list.count(0) == 4:
                        cv2.putText(frame, 'Nothing', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
                    count_list = []

                cv2.imshow('roi', mask)
                cv2.imshow('frame', frame)
                k = cv2.waitKey(5) & 0xFF
                if k == 27:
                    break

            else:
                cv2.imshow('frame', frame)
                cv2.putText(frame, 'NO SKIN COLOR', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
                print 'no skin color'
                cv2.waitKey(1)
        except:
            pass

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    Main()
