import numpy as np
import cv2
import imutils
from imutils.object_detection import non_max_suppression
from collections import deque

cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()
    if ret:
        image = imutils.resize(frame, width=1100)

        # convert the image from RGB to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # define range of orange color in HSV
        lower_green = np.array([44, 54, 63])
        upper_green = np.array([71, 255, 255])

        # Threshold the HSV image to get only orange colors
        mask = cv2.inRange(hsv, lower_green, upper_green)

        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]

        if contours:

            # cv2.drawContours(image, contours, -1, (0,255,0), 3)
            c = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(c)
            # draw the book contour (in green)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

            (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                                    padding=(8, 8), scale=1.05)

            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

            for (xA, yA, xB, yB) in pick:
                cv2.rectangle(image, (xA, yA), (xB, yB), (255, 0, 0), 2)
                C_pixel = h
                P_pixel = (yB - yA)
                # *0.9
                C_height = 22
                P_height = ((22 / C_pixel) * P_pixel)/12
                P_feet = int(P_height)
                P_inch = int((P_height - P_feet)*12)
                height = "height of person:"+str(P_feet)+"'"+str(P_inch)+"''"
                print(height)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, height, (xA, yB+15), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Window", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()