# =====================================================
# Real-Time Webcam Emotion Detection with EfficientNet-B0
# Saves faces in 'captures' and emotions in 'emotions.csv'
# =====================================================

import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import os
import csv
from datetime import datetime

# ------------------------------
# 1️⃣ Device setup
# ------------------------------
device = torch.device("cpu")

# ------------------------------
# 2️⃣ Emotion mapping
# ------------------------------
emotion_map = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# ------------------------------
# 3️⃣ Load trained EfficientNet-B0
# ------------------------------
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 7)
model.load_state_dict(torch.load("best_efficientnet_rafdb_cpu.pth", map_location=device))
model = model.to(device)
model.eval()

# ------------------------------
# 4️⃣ Preprocessing
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ------------------------------
# 5️⃣ Prepare folders and CSV
# ------------------------------
os.makedirs("captures", exist_ok=True)
csv_file = "emotions.csv"

# If CSV doesn’t exist, create it with header
if not os.path.isfile(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Emotion", "Image_Path"])

# ------------------------------
# 6️⃣ Load face detector
# ------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ------------------------------
# 7️⃣ Start webcam
# ------------------------------
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        img_tensor = transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, pred = torch.max(outputs, 1)
            emotion = emotion_map[pred.item()]

        # Draw rectangle and emotion label
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # Save face image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        img_path = f"captures/face_{timestamp}.jpg"
        cv2.imwrite(img_path, face_img)

        # Save emotion to CSV
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, emotion, img_path])

    # Show webcam
    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
