from deepface import DeepFace

def detect_baby(img_path):

    result = DeepFace.analyze(
        img_path=img_path,
        actions=['gender','emotion','race'],
        enforce_detection=False
    )

    return result[0]