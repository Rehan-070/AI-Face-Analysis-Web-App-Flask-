from flask import Flask, render_template, request
import os
from detect import detect_adult
from Deepface import detect_baby

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------- ADULT ----------------
@app.route("/adult")
def adult():
    return render_template("adult_upload.html")

@app.route("/adult_result", methods=["POST"])
def adult_result():
    file = request.files["image"]

    if not file:
        return "No file uploaded"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    gender, emotion, gen_conf, emo_conf = detect_adult(filepath)

    return render_template(
        "adult_result.html",
        image=filepath,
        gender=gender,
        emotion=emotion,
        gen_conf=gen_conf,
        emo_conf=emo_conf
    )

# ---------------- BABY ----------------
@app.route("/baby")
def baby():
    return render_template("baby_upload.html")

@app.route("/baby_result", methods=["POST"])
def baby_result():
    file = request.files["image"]

    if not file:
        return "No file uploaded"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    result = detect_baby(filepath)

    return render_template(
        "baby_result.html",
        image=filepath,
        result=result
    )

if __name__ == "__main__":
    app.run(debug=True)