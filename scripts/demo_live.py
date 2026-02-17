import cv2
import torch
import numpy as np
import sys
import os
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.resnet_small import ResNet18

COLOR_MAIN = (180, 255, 180)   
COLOR_ACCENT = (100, 255, 100) 
COLOR_BG = (35, 30, 25)       
COLOR_TEXT = (240, 240, 240)    
COLOR_BAR_BG = (55, 50, 45)     
FONT = cv2.FONT_HERSHEY_SIMPLEX 
AA = cv2.LINE_AA               

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
class_names = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']

model = ResNet18(num_classes=6, in_channels=1).to(device)
model.load_state_dict(torch.load("outputs/resnet18_best-2.pth", map_location=device))
model.eval()

target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

cap = cv2.VideoCapture(0)
print("Live-Demo is starting... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess(Image.fromarray(rgb_roi)).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
        
        pred_idx = np.argmax(probs)

        targets = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        img_float = np.float32(cv2.resize(rgb_roi, (64, 64))) / 255
        cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        cam_image_bgr = cv2.resize(cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR), (w, h))
        
        face_zone = frame[y:y+h, x:x+w]
        frame[y:y+h, x:x+w] = cv2.addWeighted(face_zone, 0.5, cam_image_bgr, 0.5, 0)

        cv2.rectangle(frame, (x, y), (x+w, y+h), COLOR_MAIN, 2)
        label = f"{class_names[pred_idx].upper()}"
        cv2.rectangle(frame, (x, y-35), (x + int(w*0.6), y), COLOR_MAIN, -1)
        cv2.putText(frame, label, (x+10, y-10), FONT, 0.6, COLOR_BG, 2, AA)

        p_w, p_h = 320, 300 
        p_x, p_y = width - p_w - 20, 40
        
        p_y = max(0, p_y)
        p_x = max(0, p_x)
        
        sub_roi = frame[p_y:p_y+p_h, p_x:p_x+p_w]
        black_bg = np.zeros_like(sub_roi)
        cv2.rectangle(black_bg, (0, 0), (p_w, p_h), COLOR_BG, -1)
        frame[p_y:p_y+p_h, p_x:p_x+p_w] = cv2.addWeighted(sub_roi, 0.3, black_bg, 0.7, 0)

        cv2.putText(frame, "EMOTION ANALYSIS", (p_x + 15, p_y + 35), FONT, 0.6, COLOR_MAIN, 1, AA)

        for i, (name, prob) in enumerate(zip(class_names, probs)):
            ry = p_y + 75 + (i * 35)
            bw = int(prob * 140)
            is_active = (i == pred_idx)
            
            curr_clr = COLOR_ACCENT if is_active else COLOR_MAIN
            thick = 2 if is_active else 1

            cv2.putText(frame, name, (p_x + 15, ry + 12), FONT, 0.5, COLOR_TEXT, thick, AA)
            cv2.rectangle(frame, (p_x + 100, ry), (p_x + 240, ry + 15), COLOR_BAR_BG, -1)
            cv2.rectangle(frame, (p_x + 100, ry), (p_x + 100 + bw, ry + 15), curr_clr, -1)
            cv2.putText(frame, f"{prob*100:.1f}%", (p_x + 250, ry + 12), FONT, 0.45, COLOR_TEXT, thick, AA)

    cv2.imshow('Emotion Recognition Demo', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
