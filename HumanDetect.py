from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import cv2


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)

    while (True):
        ret, frame = cap.read()
        if ret:

            # initialize the HOG descriptor/pre-trained person detector
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

            # for image in frame:
            image = frame
            # resize it to (1) reduce detection time and (2) improve detection accuracy
            image = imutils.resize(image, width=min(1000, image.shape[1]))
            orig = image.copy()

            # detect people in the image
            (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                                    padding=(8, 8), scale=1.05)

            # draw the original bounding boxes
            # for (x, y, w, h) in rects:
            #     cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # apply non-maxima suppression
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

            for (xA, yA, xB, yB) in pick:
                crop_img = image[yA:yB, xA:xB]
                gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                haar_cascade = cv2.CascadeClassifier('data/cascadetrain.xml')
                logo = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=20);
                for (x, y, w, h) in logo:
                    cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
                    cv2.rectangle(image, (xA + x, yA + y), (xA + x + w, yA + y + h), (255, 255, 0), 2)

                    haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
                    faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);
                    for (x, y, w, h) in faces:
                        cv2.rectangle(image, (xA + x, yA + y), (xA + x + w, yA + y + h), (255, 0, 0), 2)
                    haar_eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
                    eyes = haar_eye_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);
                    for (x, y, w, h) in eyes:
                        cv2.rectangle(image, (xA + x, yA + y), (xA + x + w, yA + y + h), (0, 0, 255), 2)

            # gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #
            # # load cascade classifier training file for haarcascade
            # haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
            #
            # # let's detect multiscale (some images may be closer to camera than others) images
            # faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);
            #
            # # print the number of faces found
            # print('Faces found: ', len(faces))
            #
            # for (x, y, w, h) in faces:
            #     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #
            # haar_eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
            # eyes = haar_eye_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);
            # for (x, y, w, h) in eyes:
            #     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

            cv2.imshow("Window", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
