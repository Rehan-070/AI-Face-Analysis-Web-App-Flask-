import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, render_template, request, Response, jsonify
import cv2
import numpy as np

from detect import detect_adult, model, face_cascade as detect_face_cascade, EMOTIONS, GENDERS, IMG_SIZE
from Deepface import detect_baby

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

capture_done_adult = False


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# ADULT IMAGE UPLOAD
# =========================================================

@app.route("/adult")
def adult():
    return render_template("adult_upload.html")


@app.route("/adult_result", methods=["POST"])
def adult_result():

    file = request.files.get("image")

    if not file:
        return "No file uploaded"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    try:
        gender, emotion, gen_conf, emo_conf = detect_adult(filepath)

    except Exception as e:
        return f"Adult detection error: {str(e)}"

    return render_template(
        "adult_result.html",
        image=filepath,
        gender=gender,
        emotion=emotion,
        gen_conf=gen_conf,
        emo_conf=emo_conf
    )


# =========================================================
# BABY IMAGE UPLOAD
# =========================================================

@app.route("/baby")
def baby():
    return render_template("baby_upload.html")


@app.route("/baby_result", methods=["POST"])
def baby_result():

    file = request.files.get("image")

    if not file:
        return "No file uploaded"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    try:
        result = detect_baby(filepath)

    except Exception as e:
        return f"DeepFace error: {str(e)}"

    return render_template(
        "baby_result.html",
        image=filepath,
        result=result
    )


# =========================================================
# ADULT LIVE CAMERA (capture + analyze)
# =========================================================

@app.route("/adult_live")
def adult_live():
    global capture_done_adult
    capture_done_adult = False
    return render_template("adult_live.html")


def generate_adult_frames():

    global capture_done_adult

    camera = cv2.VideoCapture(0)

    while camera.isOpened():

        success, frame = camera.read()

        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0 and not capture_done_adult:

            path = os.path.join(app.config["UPLOAD_FOLDER"], "live_adult.jpg")
            cv2.imwrite(path, frame)

            capture_done_adult = True
            camera.release()
            break

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()


@app.route("/adult_video")
def adult_video():
    return Response(
        generate_adult_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route("/adult_status")
def adult_status():
    return jsonify({"done": capture_done_adult})


@app.route("/adult_live_result")
def adult_live_result():

    image = "static/uploads/live_adult.jpg"

    if not os.path.exists(image):
        return "Image not captured yet"

    try:
        gender, emotion, gen_conf, emo_conf = detect_adult(image)

    except Exception as e:
        return f"Adult detection error: {str(e)}"

    return render_template(
        "adult_live_result.html",
        image=image,
        gender=gender,
        emotion=emotion,
        gen_conf=gen_conf,
        emo_conf=emo_conf
    )


# =========================================================
# ADULT LIVE FRAMING (real-time annotated stream)
# =========================================================

@app.route("/adult_live_frame")
def adult_live_frame():
    return render_template("adult_live_frame.html")


def generate_adult_live_frame_feed():
    """
    Streams annotated frames: green bounding box on each detected face
    with Gender and Emotion labels drawn using the custom model.
    """
    camera = cv2.VideoCapture(0)

    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]

            try:
                face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
                face_norm = face_resized / 255.0
                face_input = np.reshape(face_norm, (1, IMG_SIZE, IMG_SIZE, 3))

                emotion_pred, gender_pred = model.predict(face_input, verbose=0)

                emo_idx = np.argmax(emotion_pred[0])
                gen_idx = np.argmax(gender_pred[0])

                emo_label = EMOTIONS[emo_idx]
                gen_label = GENDERS[gen_idx]
                emo_conf  = round(float(emotion_pred[0][emo_idx] * 100), 1)
                gen_conf  = round(float(gender_pred[0][gen_idx] * 100), 1)

            except Exception:
                emo_label = "?"
                gen_label = "?"
                emo_conf  = 0.0
                gen_conf  = 0.0

            # Draw bounding box (green)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Background banner above box for labels
            banner_y = max(y - 50, 0)
            cv2.rectangle(frame, (x, banner_y), (x+w, y), (0, 0, 0), -1)

            # Gender label (cyan/blue)
            cv2.putText(
                frame,
                f"Gender: {gen_label} ({gen_conf}%)",
                (x + 4, banner_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 200, 0),
                1,
                cv2.LINE_AA
            )

            # Emotion label (pink)
            cv2.putText(
                frame,
                f"Emotion: {emo_label} ({emo_conf}%)",
                (x + 4, banner_y + 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (200, 100, 255),
                1,
                cv2.LINE_AA
            )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()


@app.route("/adult_live_frame_feed")
def adult_live_frame_feed():
    return Response(
        generate_adult_live_frame_feed(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# =========================================================
# BABY LIVE FRAMING (real-time annotated stream via DeepFace)
# =========================================================

@app.route("/baby_live_frame")
def baby_live_frame():
    return render_template("baby_live_frame.html")


def generate_baby_live_frame_feed():
    """
    Streams annotated frames: pink bounding box on each detected face
    with Gender and Emotion labels using DeepFace (sampled every N frames
    to keep the stream smooth).
    """
    from deepface import DeepFace

    camera = cv2.VideoCapture(0)
    frame_count = 0
    cached_results = {}   # key: face index, value: dict of labels

    # How often to run DeepFace (every N frames)
    ANALYZE_EVERY = 15

    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Run DeepFace every ANALYZE_EVERY frames for performance
        if frame_count % ANALYZE_EVERY == 0:
            cached_results = {}
            for i, (x, y, w, h) in enumerate(faces):
                face_roi = frame[y:y+h, x:x+w]
                try:
                    tmp_path = os.path.join(app.config["UPLOAD_FOLDER"], f"_tmp_face_{i}.jpg")
                    cv2.imwrite(tmp_path, face_roi)

                    result = DeepFace.analyze(
                        img_path=tmp_path,
                        actions=['gender', 'emotion'],
                        enforce_detection=False,
                        silent=True
                    )
                    r = result[0]

                    # dominant gender
                    gen_data = r.get('gender', {})
                    gen_label = max(gen_data, key=gen_data.get) if gen_data else "?"
                    gen_conf  = round(gen_data.get(gen_label, 0), 1)

                    # dominant emotion
                    emo_label = r.get('dominant_emotion', '?')
                    emo_data  = r.get('emotion', {})
                    emo_conf  = round(emo_data.get(emo_label, 0), 1)

                    cached_results[i] = {
                        'gen_label': gen_label,
                        'gen_conf':  gen_conf,
                        'emo_label': emo_label,
                        'emo_conf':  emo_conf,
                    }
                except Exception:
                    cached_results[i] = {
                        'gen_label': '?', 'gen_conf': 0,
                        'emo_label': '?', 'emo_conf': 0
                    }

        # Draw annotations on every frame using cached results
        for i, (x, y, w, h) in enumerate(faces):
            info = cached_results.get(i, {})
            gen_label = info.get('gen_label', '?')
            gen_conf  = info.get('gen_conf', 0)
            emo_label = info.get('emo_label', '?')
            emo_conf  = info.get('emo_conf', 0)

            # Bounding box (pink for baby)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (180, 100, 255), 2)

            # Background banner above box
            banner_y = max(y - 50, 0)
            cv2.rectangle(frame, (x, banner_y), (x+w, y), (0, 0, 0), -1)

            # Gender label
            cv2.putText(
                frame,
                f"Gender: {gen_label} ({gen_conf}%)",
                (x + 4, banner_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 200, 0),
                1,
                cv2.LINE_AA
            )

            # Emotion label
            cv2.putText(
                frame,
                f"Emotion: {emo_label} ({emo_conf}%)",
                (x + 4, banner_y + 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (100, 220, 255),
                1,
                cv2.LINE_AA
            )

        frame_count += 1

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()


@app.route("/baby_live_frame_feed")
def baby_live_frame_feed():
    return Response(
        generate_baby_live_frame_feed(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)