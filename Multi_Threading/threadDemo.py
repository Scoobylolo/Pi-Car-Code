import argparse
import cv2
from frames_per_sec import CountsPerSec
from VideoGet import VideoGet
from videoShow import VideoShow


classNames = []
classFile = "../Object_Detection/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "../Object_Detection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "../Object_Detection/frozen_inference_graph.pb"

pixel_inputs=(150,90)

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(*pixel_inputs)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

color=(255,0,255)

def putIterationsPerSec(frame, iterations_per_sec,thres=0.5,nms=0.2,objects=[],draw=True):
    """
    Add iterations per second text to lower-left corner of a frame.
    """

    classIds, confs, bbox = net.detect(frame,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects: 
                objectInfo.append([box,className])
                if draw:
                    cv2.rectangle(frame,box,color=color,thickness=2)
                    cv2.putText(frame,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,color,2)
                    cv2.putText(frame,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,color,2)





    cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
    return frame

def noThreading(source=0):
    """Grab and show video frames without multithreading."""

    cap = cv2.VideoCapture(source)
    cps = CountsPerSec().start()

    while True:
        (grabbed, frame) = cap.read()
        if not grabbed or cv2.waitKey(1) == ord("q"):
            break

        frame = putIterationsPerSec(frame, cps.countsPerSec())
        cv2.imshow("Video", frame)
        cps.increment()

def threadVideoGet(source=0):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Main thread shows video frames.
    """

    video_getter = VideoGet(source).start()
    cps = CountsPerSec().start()
    while True:
        if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
            video_getter.stop()
            break

        frame = video_getter.frame
        frame = putIterationsPerSec(frame, cps.countsPerSec())
        cv2.imshow("Video", frame)
        cps.increment()

if __name__=="__main__":
    threadVideoGet()
    #noThreading()
    
