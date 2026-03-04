import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 224
EMOTIONS = ["Angry", "Happy", "Sad"]
GENDERS = ["Male", "Female", "Transgender"]

model = load_model("model/emotion_gender_model.h5")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_adult(image_path):

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No Face", "No Face", 0, 0

    (x, y, w, h) = faces[0]
    face = img[y:y+h, x:x+w]

    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face / 255.0
    face = np.reshape(face, (1, IMG_SIZE, IMG_SIZE, 3))

    emotion_pred, gender_pred = model.predict(face, verbose=0)

    emo_idx = np.argmax(emotion_pred[0])
    gen_idx = np.argmax(gender_pred[0])

    emo_conf = float(emotion_pred[0][emo_idx] * 100)
    gen_conf = float(gender_pred[0][gen_idx] * 100)

    return (
        GENDERS[gen_idx],
        EMOTIONS[emo_idx],
        round(gen_conf, 1),
        round(emo_conf, 1)
    )