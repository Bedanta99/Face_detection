# Face-Eye-detection-using-opencv-in-Python
import cv2
import numpy as np
face =cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
eye =cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")
cap = cv2.VideoCapture(0)
if not cap.isOpened:
        raise IOError("cannot open")
while True:
        _,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces =face.detectMultiScale(gray,1.3,5) 
        for (x,y,w,h) in faces:
            cv2.circle(frame,(int(x+0.5*w),int(y+0.5*h)),int(0.3*(w+h)),(255,0,0),2)
            roi_gray= gray[y:y+h,x:x+w]
            roi_color=frame[y:y+h,x:x+w]
            eyes= eye.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.circle(roi_color,(int(ex+0.5*ew),int(ey+0.5*eh)),int(0.3*(ew+eh)),(255,0,0),2)
        cv2.imshow("face dtection",frame)
        c=cv2.waitKey(1)
        if c==27:
            break
cap.release()
cv2.destroyAllWindows()
