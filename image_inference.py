# =====================================================
# Facial Emotion Recognition - Single Image Inference
# =====================================================

import torch
from torchvision import transforms, models
from PIL import Image

# ------------------------------
# 1️⃣ Device setup
# ------------------------------
device = torch.device("cpu")  # CPU
print("Using device:", device)

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
# 3️⃣ Load the trained model
# ------------------------------
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 7)
model.load_state_dict(torch.load("best_efficientnet_rafdb_cpu.pth", map_location=device))
model = model.to(device)
model.eval()

# ------------------------------
# 4️⃣ Preprocessing for the image
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ------------------------------
# 5️⃣ Load the image
# ------------------------------
image_path = r"C:\Users\harsh\OneDrive\Documents\Desktop\Facial Emotion\archive\DATASET\train\fear\train_09380_aligned.jpg"  # Replace with your image path
img = Image.open(image_path)
img_tensor = transform(img).unsqueeze(0).to(device)  # add batch dimension

# ------------------------------
# 6️⃣ Prediction
# ------------------------------
with torch.no_grad():
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs, 1)
    emotion = emotion_map[predicted.item()]

print(f"Predicted emotion for {image_path}: {emotion}")
