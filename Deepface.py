from deepface import DeepFace
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def detect_baby(image_path):

    result = DeepFace.analyze(
        img_path=image_path,
        actions=["gender", "age", "race", "emotion"],
        enforce_detection=False
    )

    data = result[0]

    return {
        "gender": data["gender"],
        "age": data["age"],
        "race": data["race"],
        "emotion": data["emotion"]
    }