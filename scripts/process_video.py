import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import torch
import numpy as np
from src.models.resnet_small import ResNet18
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
class_names = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Anger']

model = ResNet18(num_classes=6, in_channels=1).to(device)
model.load_state_dict(torch.load("outputs/resnet18_best.pth", map_location=device))
model.eval()
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)
cascade_path = "data/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Verarbeitetes Video: {input_path}...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_roi)

            input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

            outputs = model(input_tensor)
            pred_idx = outputs.argmax(dim=1).item()

            targets = [ClassifierOutputTarget(pred_idx)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

            img_float = np.float32(cv2.resize(rgb_roi, (64, 64))) / 255
            cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
            cam_image = cv2.resize(cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR), (w, h))
        
            frame[y:y+h, x:x+w] = cam_image
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, class_names[pred_idx], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        out.write(frame)
    cap.release()
    out.release()
    print("Video erfolgreich gespeichert!")

if __name__ == "__main__":
    process_video("data/input_video4.mp4", "outputs/processed_video.mp4")
