import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
lst=0
cy=0
while True:
    success, img_raw = cap.read()
    img = cv2.flip(img_raw,1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 8:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                   #print(cx,cy)
                    #print((cy-lst)/(cTime - pTime))
                    if 15<cx<135 and 295<cy<305: #and ((cy-lst)/(cTime - pTime))<-500:
                        print("A struck")
                        cv2.putText(img, str('A'), (70, 295), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 0), 3)
                    if 180<cx<295 and 395<cy<405: #and ((cy-lst)/(cTime - pTime))<-500:    
                        print("B struck")
                        cv2.putText(img, str('B'), (225, 395), cv2.FONT_HERSHEY_PLAIN, 3,(0, 0, 255), 3)
                    
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
   
    #print(cTime - pTime)
    pTime = cTime
    lst=cy
    img=cv2.line(img, (10,300), (140,300), (255,0,0), 4)
    #cv2.putText(img, str('A'), (70, 295), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 0), 3)
    img=cv2.line(img, (171,400), (300,400), (0,0,255), 6)
    #cv2.putText(img, str('B'), (225, 395), cv2.FONT_HERSHEY_PLAIN, 3,(0, 0, 255), 3)
    img=cv2.line(img, (331,400), (460,400), (0,255,0), 6)
    #cv2.putText(img, str('C'), (390, 395), cv2.FONT_HERSHEY_PLAIN, 3,(0, 255, 0), 3)
    img=cv2.line(img, (491,300), (620,300), (255,255,0), 4)
    # cv2.putText(img, str('D'), (540, 295), cv2.FONT_HERSHEY_PLAIN, 3,(255, 255, 0), 3)
   
    cv2.imshow("Image", img)
    cv2.waitKey(1)
