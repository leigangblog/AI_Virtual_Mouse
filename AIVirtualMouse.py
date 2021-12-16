import cv2
import numpy as np
from HandTrackingModule import HandDetector
import time
import autopy

######################
wCam, hCam = 640, 480
frameR = 100     #Frame Reduction
smoothening = 7  #random value
######################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(3, wCam)
cap.set(4, hCam)

detector = HandDetector(detectionCon=0.8, maxHands=2)
wScr, hScr = autopy.screen.size()

# print(wScr, hScr)

while True:
    # Step1: Find the landmarks
    success, img = cap.read()
    hands,img = detector.findHands(img)


    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right


        # Step2: Get the tip of the index and middle finger
        if len(lmList1) != 0:
            x1, y1 = lmList1[8]
            x2, y2 = lmList1[12]
            # Step3: Check which fingers are up
            fingers1 = detector.fingersUp(hand1)
            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                          (255, 0, 255), 2)

            # Step4: Only Index Finger: Moving Mode
            if fingers1[1] == 1 and fingers1[2] == 0:

                # Step5: Convert the coordinates
                x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))
                # Step6: Smooth Values
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                # Step7: Move Mouse
                autopy.mouse.move(wScr - clocX, clocY)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

            # Step8: Both Index and middle are up: Clicking Mode
            if fingers1[1] == 1 and fingers1[2] == 1:
                # Step9: Find distance between fingers
                length, lineInfo,img = detector.findDistance(lmList1[8], lmList1[12], img)
                # Step10: Click mouse if distance short
                if length < 40:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    autopy.mouse.click()
    # Step11: Frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (28, 58), cv2.FONT_HERSHEY_PLAIN, 3, (255, 8, 8), 3)
    # Step12: Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)