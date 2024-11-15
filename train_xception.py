import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score, precision_score
from torchvision.models import EfficientNet_B0_Weights
import json
import os
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np


# Face extraction from videos and saving as images
class FaceExtractor:
    def __init__(self, output_dir="extracted_faces", face_cascade_path="haarcascade_frontalface_default.xml"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_path)
        self.labels = {}

    def extract_faces(self, video_path, class_label, max_frames=60, max_faces=30):
        cap = cv2.VideoCapture(video_path)
        count = 0
        frame_count = 0
        video_name = os.path.basename(video_path)

        while cap.isOpened() and frame_count < max_frames and count < max_faces:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % (cap.get(cv2.CAP_PROP_FRAME_COUNT) // max_frames or 1) != 0:
                frame_count += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                if count >= max_faces:
                    break
                face = frame[y:y + h, x:x + w]
                face_filename = f"{video_name}_face_{count}.jpg"
                face_image_path = os.path.join(self.output_dir, face_filename)
                cv2.imwrite(face_image_path, face)
                self.labels[face_filename] = class_label
                print(f"Saved: {face_image_path}")
                count += 1

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

    def save_labels(self, label_file="labels/labels.json"):
        os.makedirs(os.path.dirname(label_file), exist_ok=True)
        with open(label_file, "w") as f:
            json.dump(self.labels, f)
        print(f"Labels saved to {label_file}")


# Custom dataset for loading images and labels
class FaceDataset(Dataset):
    def __init__(self, image_dir, label_json, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        with open(label_json) as f:
            self.labels = json.load(f)
        self.image_paths = list(self.labels.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(image_path).convert("RGB")
        label = self.labels[self.image_paths[idx]]

        if self.transform:
            image = self.transform(image)

        return image, label


# Load pretrained Xception model (adapted as Xception-like with EfficientNet-b0)
def get_xception_model(num_classes, pretrained=True):
    if pretrained:
        weights = EfficientNet_B0_Weights.DEFAULT
    else:
        weights = None
    model = models.efficientnet_b0(weights=weights)  # Explicitly set weights
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


# Load data
def load_data(image_dir, label_json, batch_size, train_val_test_split=(0.7, 0.15, 0.15)):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = FaceDataset(image_dir=image_dir, label_json=label_json, transform=transform)
    train_size = int(train_val_test_split[0] * len(dataset))
    val_size = int(train_val_test_split[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# Train model
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for iteration, (images, labels) in enumerate(tqdm(train_loader), start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            print(f"Training Iteration {iteration}: Loss = {loss.item():.4f}")

        val_accuracy, val_precision = evaluate_model(model, val_loader, device)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}")


# Evaluate model with accuracy and precision
def evaluate_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    return accuracy, precision


# Main execution
if __name__ == "__main__":
    # Face extraction example
    extractor = FaceExtractor()
    video_dir = "final"  # Directory containing videos organized by class
    class_dirs = os.listdir(video_dir)  # Each subdirectory corresponds to a class (e.g., "class_0", "class_1")

    for class_dir in class_dirs:
        class_path = os.path.join(video_dir, class_dir)
        if os.path.isdir(class_path):
            class_label = int(class_dir.split('_')[-1])  # Assumes subdirectory names like "class_0", "class_1"
            for video_file in os.listdir(class_path):
                if video_file.endswith(".mp4"):  # Modify as needed for video format
                    extractor.extract_faces(os.path.join(class_path, video_file), class_label)

    extractor.save_labels()

    # Model and data setup
    image_dir = "extracted_faces"
    label_json = "labels/labels.json"
    batch_size = 32
    num_classes = 2  # Change based on your classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_xception_model(num_classes=num_classes, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader, val_loader, test_loader = load_data(image_dir, label_json, batch_size)

    # Training
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)

    # Testing
    test_accuracy, test_precision = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}")
