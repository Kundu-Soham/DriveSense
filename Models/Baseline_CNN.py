import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as T
import matplotlib.pyplot as plt

# 1. Cleaner Dataset Handling (Reads already cropped & resized 224x224 images)
class JAADPreprocessedDataset(Dataset):
    def __init__(self, base_dir, split='train'):
        self.split = split
        # Pointing to the new preprocessed directory structure
        self.img_dir = os.path.join(base_dir, 'Processed JAAD Dataset', 'images', split)
        self.label_dir = os.path.join(base_dir, 'Processed JAAD Dataset', 'labels', split)
        self.img_files = sorted(os.listdir(self.img_dir))
        self.sequences = [self.img_files[i:i+10] for i in range(0, len(self.img_files), 10)]

        # Pure augmentation transforms (no resizing transformations needed)
        self.train_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_frames = self.sequences[idx]
        final_frame_name = seq_frames[-1]
        img_path = os.path.join(self.img_dir, final_frame_name)
        
        # Read the 224x224 pre-cropped image directly and normalize pixel intensities
        image = read_image(img_path).float() / 255.0
        
        if self.split == 'train':
            image = self.train_transforms(image)
        
        # Load the isolated label file
        label_path = os.path.join(self.label_dir, final_frame_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(label_path, 'r') as f:
            label = float(f.readline().strip())
            
        return image, torch.tensor(label)


# 2. Simple Baseline CNN Model
class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),  
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5), 
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# 3. Model Configuration & Instantiation
model = BaselineCNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4) 
epochs = 20  
batch_size = 16

BASE_DIRECTORY = "." 

train_dataset = JAADPreprocessedDataset(BASE_DIRECTORY, split='train')
val_dataset = JAADPreprocessedDataset(BASE_DIRECTORY, split='val')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

history = {
    'train_loss': [], 'val_loss': [],
    'train_error': [], 'val_error': []
}

best_val_loss = float('inf')


# 4. Integrated Training Loop
for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device) 
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
        torch.save(model.state_dict(), 'best_baseline_model.pth')
    
    print(f"Epoch {epoch+1}/{epochs} - "
          f"Train Loss: {epoch_loss:.4f}, Train Err: {epoch_error:.2f}% | "
          f"Val Loss: {valid_loss:.4f}, Val Err: {valid_error:.2f}%")

print(f"\nTraining finished. Best validation loss encountered: {best_val_loss:.4f}")


# 5. Graph Generation
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(7, 5))
plt.plot(epochs_range, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
plt.plot(epochs_range, history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
plt.title('Baseline Model: Training vs Validation Loss', fontsize=12, fontweight='bold')
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss (BCEWithLogits)', fontsize=10)
plt.xticks(range(2, epochs + 1, 2))
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('baseline_loss_curves.png', dpi=300)
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(epochs_range, history['train_error'], 'b-', label='Training Error', linewidth=2)
plt.plot(epochs_range, history['val_error'], 'r--', label='Validation Error', linewidth=2)
plt.title('Baseline Model: Training vs Validation Classification Error', fontsize=12, fontweight='bold')
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Error Rate (%)', fontsize=10)
plt.xticks(range(2, epochs + 1, 2))
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('baseline_error_curves.png', dpi=300)
plt.show()