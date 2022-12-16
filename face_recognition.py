import cv2
import numpy as np
import pygame

##set windows size
height = 480
width = 640

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#eyeCascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

#set camera
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3,width)
cam.set(4,height)

#music play
pygame.init()
pygame.mixer.init()
sound = pygame.mixer.Sound("test.mp3")
volume = 50
sound.set_volume(volume/100)
sound.play()

#stream
while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=7,
        minSize=(100,100)
    )
    sum = 0
    eye_sum = 0
    num = 0
    for (x,y,w,h) in faces:
        #print("face: {} {} {} {}".format(x,y,w,h))
        #cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]
        
        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.5,
            minNeighbors=7,
            minSize=(5, 5),
        )
        for (ex, ey, ew, eh) in eyes:
            #print("eyes: {} {} {} {}".format(ex,ey,ew,eh))
            if(ey < h//3):
                num = num + 1
                sum = sum + ey + y
                eye_sum = eye_sum + eh
                #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                cv2.circle(roi_color, (ex + ew//2, ey + eh//2), 5, (0, 0, 255), -1) 
    if(num != 0):
        eye_height = sum/num
        #cv2.putText(img, "eye height: {}".format(eye_height), (10, 80), cv2.FONT_HERSHEY_SIMPLEX,  1, (0, 0, 255), 1, cv2.LINE_AA)
        volume = 50 - (eye_height - height/2+(2*eye_sum)/(3*num))*400/(height)
    if(volume>100):
        volume = 100
    elif(volume<0):
        volume = 0
    sound.set_volume(volume/100)
    cv2.putText(img, "volume: {:.1f}".format(volume), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,  1, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.ellipse(img, (width//2,height//2),(120,180),0,0,360,(0,255,0),2)
    cv2.imshow('test',img)
    k = cv2.waitKey(30) & 0xff
    if not (cv2.getWindowProperty('test',cv2.WND_PROP_VISIBLE)):
        break
    if k ==27:
        break
pygame.mixer.pause()
cam.release()
cv2.destroyAllWindows()