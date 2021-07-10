import cv2 as cv
import mediapipe as mp


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionConfidence=detectionConfidence
        self.trackConfidence=trackConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionConfidence, self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    
    def findHands(self, image, draw=True):
        imgRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLnd in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLnd, self.mpHands.HAND_CONNECTIONS)

        return image



    
    def findPosition(self, image, handNo=0, draw=True):

        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                height, width, channels = image.shape
                cx, cy = int(lm.x * width), int(lm.y * height)

                if draw:
                    cv.circle(image, (cx,cy), 15, (255,0,255), -1)

                lmList.append([id, cx, cy, lm.z])

        return lmList

    
def main():
    capture = cv.VideoCapture(0)

    detector = handDetector()
    while True:
        success, image = capture.read()
        image = cv.flip(image, 1)

        img = detector.findHands(image)
        list1 = detector.findPosition(image)
        if len(list1) != 0:
            print(list1[4])

        cv.imshow("Image", image)
        cv.waitKey(1)


if __name__ == "__main__":
    main()