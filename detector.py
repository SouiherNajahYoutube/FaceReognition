import cv2
import numpy as np
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cap=cv2.VideoCapture(0);
recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load('trainner/trainner.yml')
id=0
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 2, 3, 1, 5, 7)
while(True):
    ret,img=cap.read();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    faces = detector.detectMultiScale(gray, 1.3, 5);
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if id==1:
            id="Najah"
        elif (id<>1):
            id=raw_input('enter your id')
        cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255);
    cv2.imshow('frame',img);
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
cap.release();
cv2.destroyAllWindows();
   
