import cv2
import pickle
import imutils
import numpy as np
from keras.preprocessing.image import img_to_array

face_detection_model = 'haarcascade_frontalface_default.xml'

face_detection = cv2.CascadeClassifier(face_detection_model)
model = pickle.load(open('finalized_model.sav', 'rb'))
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
            "neutral"]

# video prediction
cv2.namedWindow('your_face')
camera = cv2.VideoCapture(0)
while True:
    frame = camera.read()[1]
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbours=1, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)

    canvas = np.zeros((250, 300, 3), dtype='uint8')
    frameCopy = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True, key=lambda x: ((x[2] - x[0]) * (x[3] - x[1]))[0])
        (fX, fY, fW, fH) = faces
        # region of interests
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        prediction = model.predict(roi)[0]
        emotion_probability = np.max(prediction)
        label = EMOTIONS[prediction.argmax()]
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, prediction)):
        text = '{}: {:.2f}%'.format(emotion, prob * 100)
        w = int(prob * 300)
        cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
        cv2.putText(canvas, text, (10, (i * 35) + 23),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        cv2.putText(frameCopy, label, (fX, fY - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameCopy, (fX, fY), (fX + fW, fY + fH),
                  (0, 0, 255), 2)
    cv2.imshow('your_face', frameCopy)
    cv2.imshow("probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
