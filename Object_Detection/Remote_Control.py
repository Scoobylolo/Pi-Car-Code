import cv2
import Actuation as act

width = 640
length = 400

def controller ():
    
    if (cv2.waitKey(33) == 119):
       act.drive()
       print("DRIVE")
    elif (cv2.waitKey(33) == 97):
       act.left()
    elif (cv2.waitKey(33) == 115):
       act.reverse()
    elif (cv2.waitKey(33) == 100):
       act.right()
    elif (cv2.waitKey(33) == 32):
       act.stop()
    else:
       act.stop()

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(3,width)
    cap.set(4,length)

    
while True:
    success, img = cap.read()
    cap.set(3,width)
    cap.set(4,length)
    #cv2.imshow("Output",img)
    controller()
    print(cv2.waitKey(33))
    cv2.waitKey(1)
    