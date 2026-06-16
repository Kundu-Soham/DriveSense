import torch
import torch.nn as nn

class BaselineSingleFrameCNN(nn.Module):
    def __init__(self):
        super(BaselineSingleFrameCNN, self).__init__()
        # Three sequential convolutional blocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: 32 x 112 x 112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: 64 x 56 x 56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Output: 128 x 28 x 28
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Expecting input shape: (Batch_Size, 3, 224, 224) 
        # Extract features from the final frame of a sequence only
        x = self.features(x)
        x = self.classifier(x)
        return x

# Validation verification
if __name__ == "__main__":
    model = BaselineSingleFrameCNN()
    sample_input = torch.randn(8, 3, 224, 224)
    output = model(sample_input)
    print("Baseline output shape:", output.shape) # Expected: [8, 1]