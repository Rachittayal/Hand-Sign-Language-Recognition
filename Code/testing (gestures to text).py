import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os
from gtts import gTTS
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
#If GPU is not suppored above statement is used

cap = cv2.VideoCapture(0) #Our system camera index code is 0
detector = HandDetector(maxHands=1)
classifier = Classifier("C:\\Users\\rachi\\OneDrive\\Desktop\\Hand Sign Language Recognition- (Hack Bytes)\\keras_model.h5", "C:\\Users\\rachi\\OneDrive\\Desktop\\Hand Sign Language Recognition- (Hack Bytes)\\labels.txt")

offset = 20
imgSize = 450

counter = 0

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgBlack = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if not imgCrop.size:
            continue  # Skip this frame if imgCrop is empty

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgBlack[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgBlack, draw=False)
            print(prediction, index)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgBlack[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgBlack, draw=False)

        cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 0, 0,), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 0), 4)

        # Convert prediction letter to audio
        tts = gTTS(text=labels[index], lang='en')
        tts.save('say.mp3')
        time.sleep(1.5)
        os.system("start say.mp3")
    cv2.imshow("Image", imgOutput)

    if cv2.waitKey(1) == ord('e'): #to stop the code press "e"
        break

    
cap.release()
cv2.destroyAllWindows()
