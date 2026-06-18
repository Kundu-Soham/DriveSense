import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as T
import matplotlib.pyplot as plt

# 1. Sequential Temporal Dataset Handling (Reads windows of exactly 10 consecutive frames)
class JAADPreprocessedSequentialDataset(Dataset):
    def __init__(self, base_dir, split='train'):
        self.split = split
        self.img_dir = os.path.join(base_dir, 'Processed JAAD Dataset', 'images', split)
        self.label_dir = os.path.join(base_dir, 'Processed JAAD Dataset', 'labels', split)
        self.img_files = sorted(os.listdir(self.img_dir))
        
        # Group frames into blocks of 10 consecutive frames per sequence window
        # FIXED: Only keep complete sequence chunks of length 10 to ensure batch stacking consistency
        self.sequences = []
        for i in range(0, len(self.img_files), 10):
            seq_chunk = self.img_files[i:i+10]
            if len(seq_chunk) == 10:
                self.sequences.append(seq_chunk)

        # Augmentation transforms (applied consistently across all frames in a clip)
        self.train_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_frames = self.sequences[idx]
        frames = []
        
        # Seed generator for consistent frame-to-frame data augmentations within a sequence
        if self.split == 'train':
            state = torch.get_rng_state()
            
        for frame_name in seq_frames:
            img_path = os.path.join(self.img_dir, frame_name)
            # Read 224x224 image directly and normalize
            image = read_image(img_path).float() / 255.0
            
            if self.split == 'train':
                torch.set_rng_state(state)
                image = self.train_transforms(image)
                
            frames.append(image)
            
        # Stack frames along the time dimension: shape (10, 3, 224, 224)
        sequence_tensor = torch.stack(frames, dim=0)
        
        # Load the isolated binary classification label from the final frame anchor
        final_frame_name = seq_frames[-1]
        label_path = os.path.join(self.label_dir, final_frame_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(label_path, 'r') as f:
            label = float(f.readline().strip())
            
        return sequence_tensor, torch.tensor(label)


# 2. Primary Hybrid CNN-LSTM Model 
class PrimaryCNNLSTM(nn.Module):
    def __init__(self, hidden_dim=128, lstm_layers=1):
        super(PrimaryCNNLSTM, self).__init__()
        
        # Three convolutional blocks for spatial feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 16 x 112 x 112
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 32 x 56 x 56
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # Output: 64 x 28 x 28
        )
        
        self.flatten_dim = 64 * 28 * 28
        
        # Recurrent layer to capture temporal walking kinematics and transitions
        self.lstm = nn.LSTM(
            input_size=self.flatten_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # Classification head mapping final temporal state to a single crossing probability logit
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Input shape: (Batch, Time, Channels, Height, Width)
        batch_size, time_steps, C, H, W = x.size()
        
        # Collapse batch and time dimensions to pass through spatial CNN
        c_in = x.view(batch_size * time_steps, C, H, W)
        c_out = self.features(c_in)
        
        # Flatten spatial maps to a sequence of vectors
        r_in = c_out.view(batch_size, time_steps, self.flatten_dim)
        
        # Pass features through the recurrent network
        lstm_out, (h_n, c_n) = self.lstm(r_in)
        
        # Take the final temporal state (last time-step output) for decision making
        final_temporal_state = lstm_out[:, -1, :]
        
        # Classify crossing intent
        logits = self.classifier(final_temporal_state)
        return logits


# 3. Model Configuration & Instantiation
model = PrimaryCNNLSTM(hidden_dim=128, lstm_layers=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
epochs = 20
batch_size = 16

BASE_DIRECTORY = "."

train_dataset = JAADPreprocessedSequentialDataset(BASE_DIRECTORY, split='train')
val_dataset = JAADPreprocessedSequentialDataset(BASE_DIRECTORY, split='val')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

history = {
    'train_loss': [], 'val_loss': [],
    'train_error': [], 'val_error': []
}

best_val_loss = float('inf')


# 4. Integrated Sequential Training Loop
for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device) # Shape: (B, 10, 3, 224, 224)
        labels = labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        preds = (outputs >= 0.0).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    epoch_error = 100.0 - epoch_acc
    
    # Validation Phase
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device).float().unsqueeze(1)
            
            val_outputs = model(val_inputs)
            v_loss = criterion(val_outputs, val_labels)
            
            val_loss += v_loss.item() * val_inputs.size(0)
            val_preds = (val_outputs >= 0.0).float()
            val_correct += (val_preds == val_labels).sum().item()
            val_total += val_labels.size(0)
            
    valid_loss = val_loss / val_total
    valid_acc = (val_correct / val_total) * 100
    valid_error = 100.0 - valid_acc
    
    history['train_loss'].append(epoch_loss)
    history['val_loss'].append(valid_loss)
    history['train_error'].append(epoch_error)
    history['val_error'].append(valid_error)
    
    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        torch.save(model.state_dict(), 'best_primary_model.pth')
    
    print(f"Epoch {epoch+1}/{epochs} - "
          f"Train Loss: {epoch_loss:.4f}, Train Err: {epoch_error:.2f}% | "
          f"Val Loss: {valid_loss:.4f}, Val Err: {valid_error:.2f}%")

print(f"\nTraining finished. Best validation loss encountered: {best_val_loss:.4f}")


# 5. Graph Generation
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(7, 5))
plt.plot(epochs_range, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
plt.plot(epochs_range, history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
plt.title('Primary Model: Training vs Validation Loss', fontsize=12, fontweight='bold')
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss (BCEWithLogits)', fontsize=10)
plt.xticks(range(2, epochs + 1, 2))
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('primary_loss_curves.png', dpi=300)
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(epochs_range, history['train_error'], 'b-', label='Training Error', linewidth=2)
plt.plot(epochs_range, history['val_error'], 'r--', label='Validation Error', linewidth=2)
plt.title('Primary Model: Training vs Validation Classification Error', fontsize=12, fontweight='bold')
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Error Rate (%)', fontsize=10)
plt.xticks(range(2, epochs + 1, 2))
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('primary_error_curves.png', dpi=300)
plt.show()