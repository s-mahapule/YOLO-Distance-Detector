# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 08:49:49 2022

@author: ggopi
"""

import cv2 as cv 
import numpy as np
import pyttsx3
import engineio

engine =pyttsx3.init()

# Distance constants 
KNOWN_DISTANCE = 45 #INCHES
PERSON_WIDTH = 16 #INCHES
MOBILE_WIDTH = 3.0 #INCHES
CLOCK_WIDTH = 11

# Object detector constant 
CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
# defining fonts 
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file 
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(608, 608), scale=1/255, swapRB=True)

# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list =[]
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id 
        color= COLORS[int(classid) % len(COLORS)]
    
        label = "%s : %f" % (class_names[classid], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
    
        # getting the data 
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid ==0: # person class id 
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid ==67:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid ==74: #clock
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        # if you want inclulde more classes then you have to simply add more [elif] statements here
        # returning list containing the object data. 
    return data_list

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance

# reading the reference image from dir 
ref_person = cv.imread('ReferenceImages/image14.png')
ref_mobile = cv.imread('ReferenceImages/image4.png')
ref_clock = cv.imread('ReferenceImages/image_car2.jpg')

mobile_data = object_detector(ref_mobile)
print(mobile_data)
mobile_width_in_rf = mobile_data[1][1]

person_data = object_detector(ref_person)
print(person_data)
person_width_in_rf = person_data[0][1]

clock_data = object_detector(ref_clock)
print(clock_data)
clock_width_in_rf = 264
#clock_data[0]

print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")

# finding focal length 
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)

focal_clock = focal_length_finder(KNOWN_DISTANCE, CLOCK_WIDTH, clock_width_in_rf)

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    dic = {}
    string = ""
    data = object_detector(frame) 
    for d in data:
        if d[0] =='person':
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            x, y = d[2]
            dic['person'] = str(round(distance,2))
        elif d[0] =='cell phone':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
            dic['cell phone'] = str(round(distance,2))
        elif d[0] =='clock':
            distance = distance_finder(focal_clock, CLOCK_WIDTH, d[1])
            x, y = d[2]
            dic['clock'] = str(round(distance,2))
            
        cv.rectangle(frame, (x, y-3), (x+150, y+23),BLACK,-1 )
        cv.putText(frame, f'Dis: {round(distance,2)} inch', (x+5,y+13), FONTS, 0.48, GREEN, 2)
        for i in dic:
            string += str(i)+str(dic[i])+"inches" 
            
        engine.say(string)
        engine.runAndWait()
        
    cv.imshow('frame',frame)
    
    key = cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()
cap.release()

