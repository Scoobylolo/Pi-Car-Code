import cv2
import numpy as np
import Actuation
import time
from functools import partial
import threading


width = 640
length = 400
threshold = 0.5 # Threshold to detect object
OBJECT_TOLERANCE=int((width+length)/2*0.05)
input_x,input_y=150,150
step_counter=0
consecutive_counters=[]
keep_turning=True

last_rotation=[True,True]
turn_strength=[100,100]

sleep_time=0.25

desired_objects=["cup","mouse","vase","bowl","eye glasses"]
desired_objects=["eye glasses","hat"]


classNames = []
classFile = "coco.names"
with open(classFile,"r") as f:
    classNames = f.read().rstrip("\n").split("\n")

print(classNames)

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(input_x,input_y)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def find_middle_of_box(box):
    #returns middle of a box
    x=int(abs(box[0]-box[2]))
    y=int(abs(box[1]-box[3]))
    return x,y

def dummy():
    pass #this function does not do anything

def center_camera(box,car_instance):
    global last_rotation
    x_little,y_little=find_middle_of_box(box)
    x_big,y_big=find_middle_of_box([0,0,width,length])
    to_return={"x":dummy,"y":dummy}
    
    if x_little-OBJECT_TOLERANCE<x_big:
        last_rotation=[True,True]
        to_return["x"]=partial(car_instance.turn,*turn_strength, *last_rotation) #to the right
    elif x_little+OBJECT_TOLERANCE>x_big:
        last_rotation=[False,False]
        to_return["x"]=partial(car_instance.turn,*turn_strength, *last_rotation) #to the left
        
    if y_little-OBJECT_TOLERANCE<y_big:
        last_rotation=[True,False]
        to_return["y"]=partial(car_instance.turn,*turn_strength, *last_rotation) #forward
    elif y_little+OBJECT_TOLERANCE>y_big:
        last_rotation=[False,True]
        to_return["y"]=partial(car_instance.turn,*turn_strength, *last_rotation) #reverse
        
    return to_return

def union(list1,dict1):
    union_dict={}
    for key,value in dict1.items():
        if key in list1:
            union_dict[key]=value
    return union_dict


def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo={}
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo[className]=box
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    return img,objectInfo

def object_found():
    car_instance.turn(100,100,True,False)
    time.sleep(0.5)
    car_instance.turn(0,0,False,False)

def motor_thread():
    while keep_turning:
        if not keep_turning:
            break
        car_instance.turn(*turn_strength,*last_rotation)
        if not keep_turning:
            break
        time.sleep(0.25)
        if not keep_turning:
            break
        car_instance.turn(0,0,False,False)
        time.sleep(1.5)

def main(car_instance):
    global step_counter,keep_turning
    cap = cv2.VideoCapture(0)
    cap.set(3,width)
    cap.set(4,length)
    #cap.set(10,70)
    
    engines=threading.Thread(target=motor_thread)
    engines.start()

    count = 0
    while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img,threshold,0.3)
        #cv2.imshow("Image output",img)
        
        
        
        
        print("image taken and analyzed")
        print(objectInfo)
        objects_found=union(desired_objects,objectInfo)
        print(objects_found)
        print("\n")
        # Checking if there was at least one object detected
        as_list=list(objects_found.keys())
        if as_list:
            #lets just go to the first one and take it down!
            to_focus = as_list[0]
            object_box_array = objects_found[as_list[0]]
            
            print(object_box_array)

            #rotation=center_camera(object_box_array,car_instance)

            #if rotation["x"].__name__!="dummy":
            #rotation["x"]()
            #time.sleep(1)
            #car_instance.turn(0,0,False,False)
            #car_instance.stop()
            #if rotation["y"].__name__!="dummy":
            #rotation["y"]()
            #time.sleep(1)
            #car_instance.turn(0,0,False,False)
            #car_instance.stop()

            step_counter=0
            consecutive_counters.append(1)
            object_found()
            keep_turning=False
            break



        # if the `q` key was pressed, break from the loop
        #key = cv2.waitKey(1) & 0xFF
        #if key == ord("q"):
            #break

        step_counter+=1
        consecutive_counters.append(0)






if __name__ == "__main__":
    car_instance=Actuation.ControlCar()
    main(car_instance)

