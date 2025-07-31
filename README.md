# 👤 Face Detection and Recognition System

A real-time **Face Detection and Recognition** system using Python, OpenCV, and the `face_recognition` library.  
It detects faces from a webcam feed and recognizes known faces using precomputed encodings.

---

🚀 Features

- 🖼️ Real-time face detection from webcam
- 🔹 Face recognition with name labels
- 📂 Load pre-saved face encodings from a pickle file
- ⚡ Lightweight and fast using `dlib` and `OpenCV`
- 📝 Easily add new faces to the database

---

🛠️ Tech Stack

- **Python** 3.8+
- **OpenCV** for real-time image processing
- **dlib** and **face_recognition** for encoding & recognition
- **pickle** for saving/loading face encodings

---

## 📁 Project Overview

This project demonstrates a simple and efficient face recognition pipeline using the following:

- **Face Encoding**: Creates numerical encodings of known faces using the `face_recognition` library.
- **Live Face Recognition**: Captures video via webcam and matches faces in real-time using previously saved encodings.
- **Use Case**: Can be extended for attendance systems, security access, or smart door lock automation.

---

## 🏗️ Project Structure

face_recognition_project/
├── dataset/ # Images of known people (training data)
├── train_encode.py # Script to encode faces and save them
├── recognize_face.py # Live face recognition using webcam
└── encodings.pickle # Stored face encodings (generated


💡 Future Improvements
-Add face registration via webcam
-Integrate attendance system
-Use CNN-based deep learning for higher accuracy
-Deploy as a web application with Flask or Streamlit


📝 License
This project is for educational purposes only.

