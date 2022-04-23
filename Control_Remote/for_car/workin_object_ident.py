import cv2
import numpy as np
import Actuation

#thres = 0.45 # Threshold to detect object

classNames = []
classFile = "/home/pi/Desktop/Project/Object_Detection/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/pi/Desktop/Project/Object_Detection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pi/Desktop/Project/Object_Detection/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(125,125)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    return img,objectInfo

width = 640
length = 400

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3,width)
    cap.set(4,length)
    #cap.set(10,70)
    count = 0
    while True:
        
        
        success, img = cap.read()
        result, objectInfo = getObjects(img,0.50,0.3, objects=[])
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Output",img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break




def linelocation (pixel_array, line_color, w, l):
    
    found_line = False
    row_end = False
    tl = [0,0]
    tr = [0,0]
    bl = [0,0]
    br = [0,0]
    
    for i in range (l-10, l-1):
        if row_end:
            row_end = False
            found_line = False
            break
        for j in range (0, w-1):
            
            if (pixel_array[i][j] in line_color) and (not found_line):
                tl = [j, i]
                found_line = True
            if (pixel_array[i][j] in line_color) and found_line and (not row_end):
                if (j > tr[0]):
                    tr[0] = j
                if j == (w-1):
                    row_end = True
            if row_end:
                break
    
    for j in range (0, w-1):# '0' may change between 0-10
        if (pixel_array[l-1][j] in line_color):
            bl = [j, l-1]
            found_line = True
        if (pixel_array[l-1][j] in line_color) and found_line:
            if (j > br[0]):
                br[0] = j
    return [tl[0], tl[1], tr[0], tr[1]],[bl[0], bl[1], br[0], br[1]]

if __name__ == "__main__":

    main()
        
        # Checking if there is a marshmello on the way
        
            
    #w_line_color_range = np.arange(0, 20, 1)
    #test = linelocation(imgGray, w_line_color_range, width, length)
    #print(test)
    #print(imgGray[450][350])
