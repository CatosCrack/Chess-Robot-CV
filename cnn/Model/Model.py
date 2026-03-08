import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch import optim
from torchvision import transforms, datasets


class CNNClassifier():
    def __init__(self) -> None:
        
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
            nn.MaxPool2d(kernel_size=2, stride=2) # Reduces spatial dimensions by half
        )

        # Classifier with fully connected layers
        self.classifier = nn.Sequential()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# The input image_array expects an array with shape (num_images, width, height)
def format_input(image_array: np.ndarray) -> torch.Tensor:
    formatted_array = np.expand_dims(image_array, axis=1) # Add channel dimension (num_images, 1, width, height)
    formatted_array = torch.from_numpy(formatted_array).float() # Convert to PyTorch tensor
    return formatted_array / 255.0 # Normalize pixel values to [0, 1]
    
# Implment the tranining loop using the labelled data in data/parsed_images
# TODO: Implement TensorBoard for data visualization
def train():

    # Load model
    model = CNNClassifier()

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
    dataset = datasets.ImageFolder(root='data/parsed_images', transform=transform)
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
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        average_validation_loss = running_validation_loss / len(val_loader)
        validation_accuracy = 100 * validation_correct_predictions / validation_total_predictions
        print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {average_validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}')

        if average_validation_loss < previous_val_loss:
            previous_val_loss = average_validation_loss
            torch.save(model.state_dict(), 'Model/weigths.pth')
            print('Model saved!')


def predict(image_array: np.ndarray) -> np.ndarray:
    pass

if __name__ == "__main__":
    train()