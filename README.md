# 🧠 AI Face Analysis Web App (Flask)

A Flask-based web application that analyzes uploaded images to detect **adult gender & emotions** and **baby detection** using AI models.

---

## 🚀 Features

- 🧑 Adult gender detection
- 😀 Emotion detection
- 👶 Baby detection
- 🖼 Image upload system
- ⚡ Flask web interface
- 📊 AI-powered analysis

---

## 🧠 How It Works

1. User uploads an image.
2. Flask saves the image in `static/uploads`.
3. The image is processed using AI models:
   - `detect_adult()` → Gender + Emotion
   - `detect_baby()` → Baby detection
4. Results are rendered on the result page with the uploaded image.

---

## 🗂️ Project Structure

app.py  
detect.py  
Deepface.py  

templates/  
static/  

static/uploads/  

Main backend logic is handled inside **app.py**.

---

## 🛠️ Requirements

Make sure you have **Python 3.x** installed.

Install required libraries:

`pip install flask opencv-python deepface numpy`

---

## 📦 Large Model Files

This project uses **Git LFS (Large File Storage)** to store large AI model files.

Install Git LFS before cloning the repository.

---

## ▶️ How to Run

# install git lfs  
`git lfs install`

# clone the repository  
`git clone <your-repo-url>`

# move into project folder  
`cd your_project_folder`

# install dependencies  
`pip install flask opencv-python deepface numpy`

# run the flask app  
`python app.py`

Open browser and visit:

`http://127.0.0.1:5000/`

---

## 🧪 Example Workflow

1. Open homepage  
2. Choose **Adult Detection** or **Baby Detection**  
3. Upload an image  
4. The system analyzes the image  
5. Result page shows:

Adult Detection:
- Gender
- Emotion
- Confidence scores

Baby Detection:
- Baby detection result

---

## 📚 Libraries Used

`Flask` – Web framework  
`OpenCV` – Image processing  
`DeepFace` – Face analysis  
`NumPy` – Numerical processing  

---

## 🤝 Contributing

- Fork the repository
- Improve AI accuracy
- Add real-time webcam detection
- Add age detection
- Improve UI
- Create a pull request

---

## 📜 License

This project is open-source and free to use for learning purposes.


---

## Author

Rehan Shaikh
