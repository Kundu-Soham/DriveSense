import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from collections import defaultdict

# 1. Variable-Length Temporal Dataset Handling (Groups whole videos dynamically)
class JAADVariableSequentialDataset(Dataset):
    def __init__(self, base_dir, split='train', max_len=15):
        self.split = split
        self.max_len = max_len
        self.img_dir = os.path.join(base_dir, 'Processed JAAD Dataset', 'images', split)
        self.label_dir = os.path.join(base_dir, 'Processed JAAD Dataset', 'labels', split)
        self.img_files = sorted(os.listdir(self.img_dir))
        
        # Group ALL frames belonging to the exact same video ID
        video_groups = defaultdict(list)
        for filename in self.img_files:
            match = re.match(r'(video_\d+)', filename)
            if match:
                video_id = match.group(1)
                video_groups[video_id].append(filename)
        
        # Keep whole videos as individual sequences
        self.sequences = []
        for video_id, frames in video_groups.items():
            # Sort frames chronologically by the frame number integer
            frames.sort(key=lambda x: int(re.search(r'frame_(\d+)', x).group(1)))
            
            # CRITICAL OPTIMIZATION: Cap the maximum sequence length to prevent pipeline hanging
            if len(frames) > self.max_len:
                frames = frames[:self.max_len]
                
            self.sequences.append(frames)

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
            image = read_image(img_path).float() / 255.0
            
            if self.split == 'train':
                torch.set_rng_state(state)
                image = self.train_transforms(image)
                
            frames.append(image)
            
        # Stack frames along the time dimension: shape (num_frames, 3, 224, 224)
        sequence_tensor = torch.stack(frames, dim=0)
        
        # Load the isolated binary classification label from the final frame anchor
        final_frame_name = seq_frames[-1]
        label_path = os.path.join(self.label_dir, final_frame_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(label_path, 'r') as f:
            label = float(f.readline().strip())
            
        return sequence_tensor, torch.tensor(label)


# Custom Collate Function to batch variable-length sequences with padding
def pad_sequence_collate(batch):
    # Sort batch by sequence length in descending order (strongly required by PyTorch packing utilities)
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    
    sequences, labels = zip(*batch)
    lengths = torch.tensor([seq.shape[0] for seq in sequences])
    
    # Pad sequences with 0.0 to match the length of the longest video inside this specific batch
    padded_seqs = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels, dim=0)
    
    return padded_seqs, labels, lengths


# 2. Variable Hybrid CNN-LSTM Model 
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

    def forward(self, x, lengths):
        # Input shape: (Batch, Max_Time_Steps, Channels, Height, Width)
        batch_size, max_time_steps, C, H, W = x.size()
        
        # Collapse batch and time dimensions to pass through spatial CNN
        c_in = x.view(batch_size * max_time_steps, C, H, W)
        c_out = self.features(c_in)
        
        # Flatten spatial maps to a sequence of vectors
        r_in = c_out.view(batch_size, max_time_steps, self.flatten_dim)
        
        # Pack the padded sequence so the recurrent units ignore trailing padding values
        packed_in = nn.utils.rnn.pack_padded_sequence(r_in, lengths.cpu(), batch_first=True)
        packed_out, (h_n, c_n) = self.lstm(packed_in)
        
        # Unpack output sequence back to regular padded structure
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        # Extract the exact final valid step output for each unique video length
        idx = (lengths - 1).view(-1, 1, 1).expand(batch_size, 1, lstm_out.size(2)).to(x.device)
        final_temporal_state = lstm_out.gather(1, idx).squeeze(1)
        
        # Classify crossing intent
        logits = self.classifier(final_temporal_state)
        return logits


# 3. Model Configuration & Instantiation
model = PrimaryCNNLSTM(hidden_dim=128, lstm_layers=1)

# INTEL XPU PERFORMANCE UPDATE: Target Intel Iris Graphics accelerator
if torch.xpu.is_available():
    device = torch.device("xpu")
    print("Using Intel GPU Acceleration via XPU backend.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA backend.")
else:
    device = torch.device("cpu")
    print("No discrete/integrated target GPU detected. Defaulting to CPU execution.")

model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
epochs = 20
batch_size = 2  # Low batch size prevents RAM/VRAM overflow with variable steps

BASE_DIRECTORY = "."

# Configured dataset with a safety length ceiling of 15 frames
train_dataset = JAADVariableSequentialDataset(BASE_DIRECTORY, split='train', max_len=15)
val_dataset = JAADVariableSequentialDataset(BASE_DIRECTORY, split='val', max_len=15)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_sequence_collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_sequence_collate)

history = {
    'train_loss': [], 'val_loss': [],
    'train_error': [], 'val_error': []
}

best_val_loss = float('inf')


# 4. Integrated Sequential Training Loop
for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    # Unpack lengths alongside inputs and labels
    for inputs, labels, lengths in train_loader:
        inputs = inputs.to(device) 
        labels = labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        # Pass both inputs and sequence lengths into forward pass
        outputs = model(inputs, lengths)
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
        for val_inputs, val_labels, val_lengths in val_loader:
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device).float().unsqueeze(1)
            
            val_outputs = model(val_inputs, val_lengths)
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