import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, render_template, request, Response, jsonify
import cv2

from detect import detect_adult
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
# ADULT LIVE CAMERA
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
# RUN APP
# =========================================================

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)