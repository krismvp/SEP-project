import cv2
import torch
import numpy as np
import sys
import os
import time
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Dynamic path setup to ensure project-wide model imports work
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../..')))
from src.models.resnet34_small import ResNet34

COLOR_MAIN = (180, 255, 180)   
COLOR_ACCENT = (100, 255, 100) 
COLOR_BG = (35, 30, 25)       
COLOR_TEXT = (240, 240, 240)    
COLOR_BAR_BG = (55, 50, 45)     
FONT = cv2.FONT_HERSHEY_SIMPLEX 
AA = cv2.LINE_AA               

# Utilize Apple Silicon (MPS) or CPU for real-time inference
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
class_names = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']

# ResNet34: Deeper architecture used for more precise feature extraction
model = ResNet34(num_classes=6, in_channels=1).to(device)
model_path = os.path.join(BASE_DIR, "../../inference/resnet34_best.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Checkpoint not found: {model_path}. "
        "Place resnet34_best.pth inside the repository's inference/ folder."
    )
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Grad-CAM for Explainable AI
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Preprocessing to match training input (64x64, Normalized)
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

cap = cv2.VideoCapture(0)
prev_time = 0
print("Live-Demo is starting... Press 'q' to quit.")
prob_buffer = [] # Prevents UI flicker by averaging predictions


while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    
    # Performance Monitoring
    curr_time = time.time()
    time_diff =  (curr_time - prev_time)
    fps = 1 / time_diff if time_diff > 0 else 0
    prev_time = curr_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Isolate face to remove background noise
        roi = frame[y:y+h, x:x+w]
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess(Image.fromarray(rgb_roi)).unsqueeze(0).to(device)

        with torch.no_grad(): # Disable gradients for faster inference
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
        
        # Temporal Smoothing for stable visualization
        prob_buffer.append(probs)
        if len(prob_buffer) > 5:
            prob_buffer.pop(0)
        avg_probs = np.mean(prob_buffer, axis=0)
        pred_idx = np.argmax(avg_probs)

        # Explainability Map
        try: 
            targets = [ClassifierOutputTarget(pred_idx)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            img_float = np.float32(cv2.resize(rgb_roi, (64, 64))) / 255
            cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
            cam_image_bgr = cv2.resize(cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR), (w, h))
            face_zone = frame[y:y+h, x:x+w]
            frame[y:y+h, x:x+w] = cv2.addWeighted(face_zone, 0.5, cam_image_bgr, 0.5, 0)
        
        except Exception as e:
            pass

        cv2.rectangle(frame, (x, y), (x+w, y+h), COLOR_MAIN, 2)
        label_text = class_names[pred_idx].upper()
        (l_w, l_h), _ = cv2.getTextSize(label_text, FONT, 0.6, 2)
        cv2.rectangle(frame, (x, y-35), (x + l_w + 20, y), COLOR_MAIN, -1)
        cv2.putText(frame, label_text, (x+10, y-10), FONT, 0.6, COLOR_BG, 2, AA)

        p_w, p_h = 320, 300 
        p_x, p_y = width - p_w - 20, 40
        
        p_y = max(0, p_y)
        p_x = max(0, p_x)
        
        sub_roi = frame[p_y:p_y+p_h, p_x:p_x+p_w]
        overlay = np.zeros_like(sub_roi)
        cv2.rectangle(overlay, (0, 0), (p_w, p_h), COLOR_BG, -1)
        frame[p_y:p_y+p_h, p_x:p_x+p_w] = cv2.addWeighted(sub_roi, 0.3, overlay, 0.7, 0)

        cv2.putText(frame, "EMOTION ANALYSIS", (p_x + 15, p_y + 35), FONT, 0.6, COLOR_MAIN, 1, AA)

        for i, (name, prob) in enumerate(zip(class_names, avg_probs)):
            ry = p_y + 75 + (i * 35)
            bw = int(prob * 140)
            is_active = (i == pred_idx)
            
            curr_clr = COLOR_ACCENT if is_active else COLOR_MAIN
            thick = 2 if is_active else 1

            cv2.putText(frame, name, (p_x + 15, ry + 12), FONT, 0.5, COLOR_TEXT, thick, AA)
            cv2.rectangle(frame, (p_x + 100, ry), (p_x + 240, ry + 15), COLOR_BAR_BG, -1)
            cv2.rectangle(frame, (p_x + 100, ry), (p_x + 100 + bw, ry + 15), curr_clr, -1)      
            cv2.putText(frame, f"{prob*100:.1f}%", (p_x + 250, ry + 12), FONT, 0.45, COLOR_TEXT, thick, AA)

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), FONT, 0.7, COLOR_ACCENT, 2, AA)
    cv2.imshow('Emotion Recognition Demo', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
