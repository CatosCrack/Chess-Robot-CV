import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch import optim
from torchvision import transforms, datasets
import os
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / 'data' / 'parsed_images'
MODEL_WEIGHTS_PATH = SCRIPT_DIR / 'weigths.pth'



class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        # Feature extractor with convolutional layers
        # TODO: Update kernel, stride, and padding once image size is determined
        self.features = nn.Sequential(
            # Uses padding=2 and stride=1 to preserve dimensions
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=2, stride=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Reduces spatial dimensions by half
            
            # Uses padding=1 and stride=1 to preserve even dimensions
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1, padding_mode='replicate'),
            nn.ReLU(),
        )

        # Classifier with fully connected layers
        # 50x50 -> Conv1 -> 52x52 -> Pool1 -> 26x26
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 26 * 26, 120),
            nn.ReLU(),
            nn.Linear(120, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def stack_images(self, image_array: np.ndarray) -> torch.Tensor:
        if image_array.ndim == 3: # Add channel dimension if missing
            formatted_array = np.expand_dims(image_array, axis=1) # Add channel dimension (num_images, 1, width, height)
        else:
            formatted_array = image_array
            
        formatted_array = torch.from_numpy(formatted_array).float() # Convert to PyTorch tensor
        return formatted_array / 255.0 # Normalize pixel values to [0, 1]
        
# Implment the tranining loop using the labelled data in data/parsed_images
# TODO: Implement TensorBoard for data visualization
def train():

    # Load model
    model = Model()

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define transformations for the datasets
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((50, 50)), # Resize images to a consistent size (e.g., 50x50)
        transforms.ToTensor()
    ])

    # Load the dataset and split into training and validation
    dataset = datasets.ImageFolder(root=str(DATA_DIR), transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Train the model
    epochs = 10
    previous_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()

        # Training loop
        running_training_loss = 0.0
        training_correct_predictions = 0
        training_total_predictions = 0

        for images, labels in train_loader:
            labels = labels.float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            predictions = (outputs > 0.5).float() # Convert probabilities to 1s (>0.5) and 0s (<=0.5)
            training_correct_predictions += (predictions == labels).sum().item()
            training_total_predictions += labels.size(0)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_training_loss += loss.item()

        average_training_loss = running_training_loss / len(train_loader)
        training_accuracy = 100 * training_correct_predictions / training_total_predictions
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {average_training_loss:.4f}, Training Accuracy: {training_accuracy:.4f}')

        # Validation loop
        model.eval()
        running_validation_loss = 0.0
        validation_correct_predictions = 0
        validation_total_predictions = 0

        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels.float().unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_validation_loss += loss.item()
                predictions = (outputs > 0.5).float() # Convert probabilities to 1s (>0.5) and 0s (<=0.5)
                validation_correct_predictions += (predictions == labels).sum().item()
                validation_total_predictions += labels.size(0)

        average_validation_loss = running_validation_loss / len(val_loader)
        validation_accuracy = 100 * validation_correct_predictions / validation_total_predictions
        print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {average_validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}')

        if average_validation_loss < previous_val_loss:
            previous_val_loss = average_validation_loss
            torch.save(model.state_dict(), str(MODEL_WEIGHTS_PATH))
            print('Model saved!')


def predict(image_array: np.ndarray) -> np.ndarray:
    model = Model()
    # Load weights for inference
    model.load_state_dict(torch.load(str(MODEL_WEIGHTS_PATH), weights_only=True))
    model.eval() 

    input_tensor = model.stack_images(image_array)

    with torch.no_grad(): 
        outputs = model(input_tensor)
        # Convert probabilities to binary 0/1
        predictions = (outputs > 0.5).int()
        
    return predictions.numpy().flatten()

if __name__ == "__main__":
    train()