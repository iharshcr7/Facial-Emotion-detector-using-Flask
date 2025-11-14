📌 Facial Emotion Recognition System

A machine learning–powered application capable of detecting human facial expressions in real time using a webcam feed. The system classifies emotions such as Happy, Sad, Angry, Neutral, Fear, Surprise, and Disgust using a deep learning model trained on facial expression datasets.

🎯 Features

✔ Real-time facial detection using OpenCV
✔ CNN-based emotion classification
✔ Pre-trained model included (or optional download)
✔ Support for live camera feed and image input
✔ Lightweight & fast prediction
✔ Modular and readable architecture

🧠 Technologies Used
Component	Technology
Programming Language	Python 3.13
Computer Vision	OpenCV
Deep Learning	TensorFlow / Keras
Visualization	Matplotlib
Dataset	FER2013 (optional / training stage)
📁 Project Structure
Facial Emotion Detection/
│
├─ model/
│   ├─ emotion_model.h5        → Pre-trained neural network model
│   └─ labels.json             → Maps model output to emotion classes
│
├─ dataset/ (Optional)
│   ├─ train/                  → Folder used while training
│   ├─ test/
│   └─ metadata.csv            → Dataset details
│
├─ src/
│   ├─ train.py                → Script to train the CNN model
│   ├─ detect.py               → Real-time emotion recognition using webcam
│   ├─ preprocess.py           → Image normalization & preprocessing functions
│   ├─ model_builder.py        → CNN architecture and compilation
│   └─ utils.py                → Helper functions (logging, visualization)
│
├─ haarcascade/
│   └─ haarcascade_frontalface_default.xml → Face detection classifier
│
├─ requirements.txt            → Dependencies
├─ README.md                   → Main documentation (you are here)
└─ .gitignore                  → Excludes unwanted files from GitHub

🧪 Emotion Classes
ID	Emotion
0	Angry
1	Disgust
2	Fear
3	Happy
4	Sad
5	Surprise
6	Neutral
🚀 How to Run the Project
1️⃣ Create Virtual Environment (Recommended)
python -m venv venv


Activate:

Windows

venv\Scripts\activate


Mac/Linux

source venv/bin/activate

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Live Detection
python src/detect.py


Once the program runs, your webcam will open and display predictions in real time.

4️⃣ Train Model (Optional)

If you want to retrain using dataset:

python src/train.py

🧬 Model Architecture Summary

The CNN model consists of:

Convolution layers for feature extraction

MaxPooling layers to reduce spatial dimensions

Dropout for overfitting prevention

Dense & Softmax output layer for classification

📦 Output Examples
Example	Description
📷 Webcam feed	Displays bounding box and predicted emotion
📊 Training log	Shows accuracy/loss curves
🧾 Saved Model	Stored at /model/emotion_model.h5
🛡️ Known Limitations & Future Enhancements
Current Limitation	Planned Solution
Lower accuracy in low lighting	Add histogram equalization
Only frontal faces detected	Integrate Dlib or YOLO face detector
Limited emotion classes	Expand dataset and classes
👨‍💻 Contribution Guide

Fork the repo

Create a feature branch

Make changes

Submit pull request 🚀

📄 License

This project is released under the MIT License — free for personal and commercial use.

⭐ Support

If this project helped you, consider starring the repository to support development ❤️
