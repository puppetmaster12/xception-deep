import os
import glob
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
            features = self.scaler.fit_transform(df.values)  # Scale features
            self.data.append(torch.tensor(features, dtype=torch.float32))

            # Extract label based on the filename
            video_name = os.path.basename(csv_file).replace('.csv', '')
            self.targets.append(self.labels[video_name])

            if features.shape[0] != 300:
                print(f"Warning: File {csv_file} has {features.shape[0]} rows after preprocessing.")
    
    def preprocess_data(self, data):
        if data.shape[0] > 300:
            return data[:300]
        elif data.shape[0] < 300:
            # Pad with zeros to reach 300 rows
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

# Cross-Attention Mechanism
class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, x1, x2):
        Q = self.query_proj(x1)  # (batch, seq_len, embed_dim)
        K = self.key_proj(x2)
        V = self.value_proj(x2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output

# Xception model with cross-attention
class CustomXception(nn.Module):
    def __init__(self, xception_model, embed_dim=1280):
        super(CustomXception, self).__init__()
        self.xception = xception_model
        self.cross_attention = CrossAttention(embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)  # To combine cross-attention output
        self.output_fc = nn.Linear(embed_dim, 2)  # Output layer for classification
        
        # Adjust the input channel of the first conv layer in Xception
        self.xception.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        # Duplicate input for cross-attention as an example
        x1 = x
        x2 = x

        # Extract features using Xception
        feat1 = self.xception(x1)
        feat2 = self.xception(x2)

        # Apply cross-attention between the two feature maps
        cross_attn_output = self.cross_attention(feat1, feat2)

        # Integrate cross-attention result and Xception output
        combined = feat1 + self.fc(cross_attn_output)
        out = self.output_fc(combined)
        
        return out

# Training function
def train(model, dataloader, criterion, optimizer, device, num_epochs=5):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")

# Save checkpoint with timestamp
def save_checkpoint(model, optimizer, epoch, folder="checkpoint"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(folder, f"model_{timestamp}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

# Main function to set everything up
def main(csv_folder, json_labels, num_classes, batch_size=32, num_epochs=5):
    device = torch.device("cpu")
    # Dataset and DataLoader
    dataset = VideoDataset(csv_folder, json_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, loss function, optimizer
    xception = pretrainedmodels.__dict__['xception'](pretrained='imagenet')
    model = CustomXception(xception)
    model.output_fc = nn.Linear(model.xception.last_linear.in_features, num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train(model, dataloader, criterion, optimizer, num_epochs, device)

    # Save the checkpoint
    save_checkpoint(model, optimizer, num_epochs)

if __name__ == "__main__":
    csv_folder = "final"  # Folder with CSV files
    json_labels = "labels/labels.json"  # JSON file with labels
    num_classes = 2  # Set based on the number of unique classes in the labels
    main(csv_folder, json_labels, num_classes)
