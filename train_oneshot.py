import os
import glob
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import numpy as np
import pretrainedmodels
import ssl
from random import sample

ssl._create_default_https_context = ssl._create_unverified_context

# Custom dataset to load CSV files and JSON labels
class VideoDataset(Dataset):
    def __init__(self, csv_folder, json_labels, scaler=None):
        self.csv_files = sorted(glob.glob(os.path.join(csv_folder, "*.csv")))
        self.labels = self.load_labels(json_labels)
        self.scaler = scaler if scaler else StandardScaler()
        
        # Load and scale all data once for simplicity
        self.data = []
        self.targets = []
        for csv_file in self.csv_files:
            df = pd.read_csv(csv_file, index_col=0, nrows=300)
            features = self.preprocess_data(df.values)
            features = self.scaler.fit_transform(features)  # Scale features
            self.data.append(torch.tensor(features, dtype=torch.float32))
            
            # Extract label based on the filename
            video_name = os.path.basename(csv_file).replace('.csv', '')
            self.targets.append(self.labels[video_name])
            
    def preprocess_data(self, data):
        if data.shape[0] > 300:
            return data[:300]
        elif data.shape[0] < 300:
            padding = np.zeros((300 - data.shape[0], data.shape[1]))
            return np.vstack((data, padding))
        return data

    def load_labels(self, json_path):
        with open(json_path, 'r') as f:
            labels = json.load(f)
        return {k: int(v) for k, v in labels.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = torch.tensor(self.targets[idx], dtype=torch.long)
        data = data.expand(32, data.size(-2), data.size(-1))
        return data, label

# Xception model with pretrained weights for few-shot learning
class CustomXception(nn.Module):
    def __init__(self, xception_model, num_classes):
        super(CustomXception, self).__init__()
        self.xception = xception_model
        self.xception.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.xception.last_linear = nn.Linear(self.xception.last_linear.in_features, num_classes)

    def forward(self, x):
        return self.xception(x)

# Few-shot episodic training function
def few_shot_train(model, dataset, criterion, optimizer, device, num_classes, num_shots=5, num_query=15, num_episodes=10):
    model.to(device)
    model.train()
    
    for episode in range(num_episodes):
        # Sample data for few-shot training
        support_set, query_set = [], []
        for cls in range(num_classes):
            cls_indices = [i for i, label in enumerate(dataset.targets) if label == cls]
            sampled_indices = sample(cls_indices, num_shots + num_query)
            support_set.extend(sampled_indices[:num_shots])
            query_set.extend(sampled_indices[num_shots:])

        # Prepare support and query DataLoader
        support_loader = DataLoader([dataset[i] for i in support_set], batch_size=num_shots, shuffle=True)
        query_loader = DataLoader([dataset[i] for i in query_set], batch_size=num_query, shuffle=True)
        
        # Inner-loop training on the support set
        for inputs, labels in support_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Evaluate on the query set for this episode
        query_loss, correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in query_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                query_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
        
        query_loss /= len(query_loader)
        accuracy = correct / len(query_set)
        print(f"Episode [{episode+1}/{num_episodes}], Query Loss: {query_loss:.4f}, Accuracy: {accuracy:.4f}")

# Save checkpoint with timestamp
def save_checkpoint(model, optimizer, episode, folder="checkpoint"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(folder, f"model_{timestamp}_episode_{episode}.pth")
    torch.save({
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

# Main function for few-shot learning
def main(csv_folder, json_labels, num_classes, num_shots=5, num_query=15, num_episodes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Dataset
    dataset = VideoDataset(csv_folder, json_labels)
    
    # Model, loss function, optimizer
    xception = pretrainedmodels.__dict__['xception'](pretrained='imagenet')
    model = CustomXception(xception, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train using few-shot episodes
    few_shot_train(model, dataset, criterion, optimizer, device, num_classes, num_shots, num_query, num_episodes)
    
    # Save the final checkpoint
    save_checkpoint(model, optimizer, num_episodes)

if __name__ == "__main__":
    csv_folder = "final"  # Folder with CSV files
    json_labels = "labels/labels.json"  # JSON file with labels
    num_classes = 2  # Set based on the number of unique classes in the labels
    main(csv_folder, json_labels, num_classes)
