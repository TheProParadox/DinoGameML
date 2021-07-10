import HandTrackingModule as htm
import keyboard as kb
import cv2 as cv
import time

capture = cv.VideoCapture(0)
detector = htm.handDetector(detectionConfidence=0.9)

pTime = 0
cTime = 0

while True:
    _ , image = capture.read()
    image = cv.flip(image, 1)

    img = detector.findHands(image)
    userHand = detector.findPosition(image)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(image,"FPS: " + str(int(fps)), (100,50), cv.FONT_HERSHEY_SIMPLEX, 2, 0, 2)
    if len(userHand) != 0:
        if userHand[8][2] < userHand[6][2] and userHand[12][2] > userHand[10][2]:
            kb.press('space')      
        else:
            kb.release('space')    

        if userHand[8][2] < userHand[6][2] and userHand[12][2] < userHand[10][2]:
            kb.press('down')
        else:
            kb.release('down')          
    
    cv.imshow("User Cam", image)
    cv.waitKey(1)