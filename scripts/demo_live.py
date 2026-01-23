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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
class_names = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']

model = ResNet18(num_classes=6, in_channels=1).to(device)
model.load_state_dict(torch.load("outputs/resnet18_best.pth", map_location=device))
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

print("Live-Demo startet... Drücke 'q' zum Beenden.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_roi)

        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        outputs = model(input_tensor)

        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        conf = conf.item()
        pred_idx = pred_idx.item()

        targets = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        img_float = np.float32(cv2.resize(rgb_roi, (64, 64))) / 255
        cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        cam_image = cv2.resize(cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR), (w, h))

        frame[y:y+h, x:x+w] = cam_image
        label = f"{class_names[pred_idx]} ({conf*100:.1f}%)"
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Emotion Recognition Live', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()