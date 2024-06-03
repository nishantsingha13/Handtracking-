import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)#1camera

mpHands=mp.solutions.hands
hands=mpHands.Hands()#2create object
mpDraw=mp.solutions.drawing_utils#6making the 21 landmarks points caluclation

#8fps
pTime=0#previoustime
cTime=0#currenttime



while True:
    success,img=cap.read()#1
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#3because hand object only take rgb image
    result=hands.process(imgRGB)#4process will process the frame but not display anything
    #print(result.multi_hand_landmarks)#5tells and gives values of hand while run

    if result.multi_hand_landmarks:#7
        for handLandmarks in result.multi_hand_landmarks:#7for all the hands
            #10getting info within the hands(like id no,coordinates)landmark information will give x and y coordinate
            for id,lm in enumerate(handLandmarks.landmark):#for each id we will have landmark(x and y cordinate)
               # print(id,lm)
                #11we have x and y coordiante in decimal and we want it in pixel(height,width)
                h,w,c=img.shape#c=channel(no of colors in image)
                cx,cy=int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)#id beacuse we will know about which landmarks it tells about

                if id==0:
                  cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)#fill on fill the circle


            mpDraw.draw_landmarks(img,handLandmarks,mpHands.HAND_CONNECTIONS)#7hand connection will connect the hand


    #8
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    #9 showing framerate
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)


    cv2.imshow("Image",img)#1capturing the video,opening of camera
    cv2.waitKey(1)##1adjusted the parameter to 1 for continuous display


