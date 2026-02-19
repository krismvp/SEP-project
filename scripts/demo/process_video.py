import sys
import os
import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
class_names = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']

# Model Init (ResNet34 for deeper feature extraction)
model = ResNet34(num_classes=6, in_channels=1).to(device)
model_path = os.path.join(BASE_DIR, "../../outputs/resnet34_best.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Grad-CAM (Target last conv layer for spatial explanation)
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Input Pipeline (Match training resolution/normalization)
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def process_video(input_path, output_path):
    # Support both absolute (dialog) and relative (manual) paths
    full_input_path = input_path if os.path.isabs(input_path) else os.path.abspath(os.path.join(BASE_DIR, "../../", input_path))
    full_output_path = output_path if os.path.isabs(output_path) else os.path.abspath(os.path.join(BASE_DIR, "../../", output_path))

    os.makedirs(os.path.dirname(full_output_path), exist_ok=True)

    cap = cv2.VideoCapture(full_input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {full_input_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(full_output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))

    # Smoothing buffer (Reduce flicker in UI bars)
    prob_buffer = []
    buffer_size = 12

    print(f"Processing: {os.path.basename(full_input_path)}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Inference on localized face region
            roi = frame[y:y+h, x:x+w]
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(Image.fromarray(rgb_roi)).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
            
            # temporal averaging for stability
            prob_buffer.append(probs)
            if len(prob_buffer) > buffer_size: prob_buffer.pop(0)
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
            except Exception: pass
    
            cv2.rectangle(frame, (x, y), (x+w, y+h), COLOR_MAIN, 2) 
            cv2.rectangle(frame, (x, y-35), (x + int(w*0.6), y), COLOR_MAIN, -1)
            cv2.putText(frame, f"{class_names[pred_idx].upper()}", (x+10, y-10), FONT, 0.7, COLOR_BG, 2, AA)
            
            # Sidebar Panel (Translucent background for readability)
            panel_w, panel_h = 320, 300 
            panel_x, panel_y = width - panel_w - 20, 40
            sub_roi = frame[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w]
            black_bg = np.zeros_like(sub_roi)
            cv2.rectangle(black_bg, (0, 0), (panel_w, panel_h), COLOR_BG, -1)
            frame[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w] = cv2.addWeighted(sub_roi, 0.3, black_bg, 0.7, 0)

            cv2.putText(frame, "EMOTION METRICS", (panel_x + 15, panel_y + 35), FONT, 0.6, COLOR_MAIN, 1, AA)

            for i, (name, prob) in enumerate(zip(class_names, avg_probs)):
                ry = panel_y + 75 + (i * 35)
                bw = int(prob * 140)
                is_active = (i == pred_idx)

                curr_clr = COLOR_ACCENT if is_active else COLOR_MAIN
                cv2.putText(frame, f"{name}", (panel_x + 15, ry + 12), FONT, 0.5, COLOR_TEXT, (2 if is_active else 1), AA)
                cv2.rectangle(frame, (panel_x + 100, ry), (panel_x + 240, ry + 15), COLOR_BAR_BG, -1)
                cv2.rectangle(frame, (panel_x + 100, ry), (panel_x + 100 + bw, ry + 15), curr_clr, -1)
                cv2.putText(frame, f"{prob*100:.1f}%", (panel_x + 250, ry + 12), FONT, 0.45, COLOR_TEXT, (2 if is_active else 1), AA)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Export Complete: {full_output_path}")

if __name__ == "__main__":
    # OS-Native file picker for cross-platform usability
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    
    selected_video = filedialog.askopenfilename(
        title="Select Video for Emotion Analysis",
        filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")]
    )
    if selected_video:
        output_name = os.path.join(BASE_DIR, "../../outputs", f"processed_{os.path.basename(selected_video)}")
        process_video(selected_video, output_name)
    else:
        print("Selection cancelled.")
