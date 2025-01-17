#!/usr/bin/env python

import rospy
import cv2
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from keras.models import load_model
import numpy as np
import tensorflow as tf
import now_test as myTest

import Queue

callback_queue = Queue.Queue()


kernel = np.ones((3, 3), np.uint8)
model = load_model('thr_final_model2.h5')
model._make_predict_function()

graph = tf.get_default_graph()

def prediction_callback(img):

    test = np.array(img)
    test = np.expand_dims(test, axis=0)

    return model.predict_classes(test)
    

def callback(img_msg):

    # try:
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")

    frame = cv2.resize(cv_image,(640,480))
    frame = cv2.flip(frame,1)
    roi = frame[100:300, 100:300]
    roi_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    roi_s = roi[100:140, 80:120]
    roi_s_ycrcb = cv2.cvtColor(roi_s, cv2.COLOR_BGR2YCrCb)
    cv2.rectangle(frame, (125, 100), (275, 300), (0, 0, 255), 0)

    y = roi_s_ycrcb[20][20][0]
    cr = roi_s_ycrcb[20][20][1]
    cb = roi_s_ycrcb[20][20][2]
    
    global graph
    global model

    if y > 180:
        if y>229:
            y=230
        lower_skin = np.array([y - 20, cr - 10, cb - 10], dtype=np.uint8)
        upper_skin = np.array([y + 20, cr + 10, cb + 10], dtype=np.uint8)
    
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

        img = img / 255

        # test = np.array(img)
        # test = np.expand_dims(test, axis=0)

        # test
        # predict = myTest.predictor(test)

        # with graph.as_default():
            # predict = model.predict_classes(test)
        with graph.as_default():
            callback_queue.put(lambda: prediction_callback(img))

        # predict = model.predict_classes(test)

        cv2.imshow('roi', mask)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    else:
        cv2.imshow('frame', frame)
        cv2.putText(frame, 'NO SKIN COLOR', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
        print 'no skin color'
        cv2.waitKey(1)
    # except Queue.Empty:
    #     pass

def receiver():
    pass
    # rospy.init_node('receiver', anonymous=True)
    # rospy.Subscriber('image_topic',Image,callback)
    # rospy.spin()

if __name__ == '__main__':
    rospy.init_node('receiver', anonymous=True)
    rospy.Subscriber('image_topic',Image, callback)

    while not rospy.is_shutdown():
        try:
            print "hit", callback_queue.get(True, 2)()[0]
        except Queue.Empty:
            print "here"
            pass
    rospy.spin()
    
# receiver()

    