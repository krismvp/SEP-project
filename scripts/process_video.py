# System path setup for importing custom modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Computer vision and deep learning libraries
import cv2
import torch
import numpy as np
from src.models.resnet_small import ResNet18
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

# UI color scheme for video overlay visualization
COLOR_MAIN = (180, 255, 180)    # Light green for primary elements
COLOR_ACCENT = (100, 255, 100)  # Bright green for highlighted elements
COLOR_BG = (35, 30, 25)         # Dark background color
COLOR_TEXT = (240, 240, 240)    # Light gray text
COLOR_BAR_BG = (55, 50, 45)     # Background color for probability bars

FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA                # Anti-aliasing flag for smooth text               

# Select device: MPS (Apple Silicon) if available, otherwise CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Define 6 emotion classes
class_names = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']

# Load pretrained ResNet18 model for emotion recognition
model = ResNet18(num_classes=6, in_channels=1).to(device)
model.load_state_dict(torch.load("outputs/resnet18_best-2.pth", map_location=device))
model.eval()  # Set to evaluation mode

# Initialize Grad-CAM for model explainability (visualize attention)
target_layers = [model.layer4[-1]]  # Use last convolutional layer
cam = GradCAM(model=model, target_layers=target_layers)

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

# Define image preprocessing pipeline (resize, grayscale, normalize)
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def process_video(input_path, output_path):
    """Process video file: detect faces, predict emotions, overlay Grad-CAM and probability panel.
    
    Args:
        input_path: Path to input video file
        output_path: Path to save processed video with emotion analysis overlays
    """
    # Open input video and extract properties
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize video writer with H.264 codec (avc1)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"processing...")

    # Process each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break  # Exit when video ends

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces

        # Process each detected face in the frame
        for (x, y, w, h) in faces:
            # Extract face region of interest (ROI)
            roi = frame[y:y+h, x:x+w]
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(Image.fromarray(rgb_roi)).unsqueeze(0).to(device)

            # Predict emotion probabilities
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
            
            pred_idx = np.argmax(probs)  # Get predicted emotion class

            # Generate Grad-CAM attention map for predicted class
            targets = [ClassifierOutputTarget(pred_idx)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            img_float = np.float32(cv2.resize(rgb_roi, (64, 64))) / 255
            cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
            cam_image_bgr = cv2.resize(cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR), (w, h))
            
            # Overlay Grad-CAM heatmap on face region (50% blend)
            face_zone = frame[y:y+h, x:x+w]
            frame[y:y+h, x:x+w] = cv2.addWeighted(face_zone, 0.5, cam_image_bgr, 0.5, 0)

            # Draw bounding box around detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), COLOR_MAIN, 2) 
            
            # Display predicted emotion label above face
            label = f"{class_names[pred_idx].upper()}"
            cv2.rectangle(frame, (x, y-35), (x + int(w*0.5), y), COLOR_MAIN, -1)
            cv2.putText(frame, label, (x+10, y-10), FONT, 0.7, COLOR_BG, 2, AA)

            # Define position for probability panel (top-right corner)
            panel_w, panel_h = 320, 300 
            panel_x, panel_y = width - panel_w - 20, 40
            
            # Create semi-transparent dark background for probability panel
            sub_roi = frame[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w]
            black_bg = np.zeros_like(sub_roi)
            cv2.rectangle(black_bg, (0, 0), (panel_w, panel_h), COLOR_BG, -1)
            frame[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w] = cv2.addWeighted(sub_roi, 0.3, black_bg, 0.7, 0)

            # Display panel title
            cv2.putText(frame, "EMOTION ANALYSIS", (panel_x + 15, panel_y + 35), 
                        FONT, 0.6, COLOR_MAIN, 1, AA)

            # Draw probability bars for each emotion class
            for i, (name, prob) in enumerate(zip(class_names, probs)):
                ry = panel_y + 75 + (i * 35)  # Vertical position for this emotion
                bar_max_width = 140
                bar_width = int(prob * bar_max_width)  # Bar width proportional to probability
                is_active = (i == pred_idx)  # Highlight predicted class
                
                # Use accent color and thicker lines for predicted emotion
                current_color = COLOR_ACCENT if is_active else COLOR_MAIN
                txt_thick = 2 if is_active else 1

                # Display emotion name, probability bar, and percentage
                cv2.putText(frame, f"{name}", (panel_x + 15, ry + 12), FONT, 0.5, COLOR_TEXT, txt_thick, AA), 
                cv2.rectangle(frame, (panel_x + 100, ry), (panel_x + 100 + bar_max_width, ry + 15), COLOR_BAR_BG, -1)  # Background bar
                cv2.rectangle(frame, (panel_x + 100, ry), (panel_x + 100 + bar_width, ry + 15), current_color, -1)  # Filled bar
                cv2.putText(frame, f"{prob*100:.1f}%", (panel_x + 100 + bar_max_width + 10, ry + 12), 
                            FONT, 0.45, COLOR_TEXT, txt_thick, AA)

        # Write processed frame to output video
        out.write(frame)

    # Cleanup: Release video capture and writer
    cap.release()
    out.release()
    print("Export Complete")

# Script entry point - process default video file
if __name__ == "__main__":
    process_video("data/input_video5.mp4", "outputs/processed_video.mp4")
