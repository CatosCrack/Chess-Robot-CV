import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms, datasets
import kagglehub
import matplotlib.pyplot as plt
import time
import random
from model import Model  


def run_benchmark():
    start_time = time.time()

    dataset_path = kagglehub.dataset_download("geovanafranca/chess-board-occupancy-binary")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(45) 
    model = Model().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Higher threshold reduces false positives for "piece" predictions
    THRESHOLD = 0.85

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }


    transform = transforms.Compose([
        transforms.Grayscale(),
        # BRIGHTNESS JITTER: Makes "Black" squares look gray and "White" squares look gray
        # This forces the model to stop relying on pixel intensity
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        # BLUR: Removes the sharp "lines" of the board that confuse the model
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        transforms.Resize((50, 50)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    # Balance classes during training to reduce bias toward "pieces".
    train_targets = [dataset.targets[i] for i in train_data.indices]
    class_counts = torch.bincount(torch.tensor(train_targets), minlength=2).float()
    class_weights = 1.0 / class_counts.clamp(min=1.0)
    sample_weights = torch.tensor([class_weights[target] for target in train_targets], dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_data, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    epochs = 10

    for epoch in range(epochs): 
        model.train()
        running_loss = 0.0
        train_correct, train_total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            train_probs = outputs.detach().flatten()
            train_predictions = (train_probs > THRESHOLD).int()
            train_labels = labels.flatten().int()
            train_correct += (train_predictions == train_labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        correct, total = 0, 0
        fp, fn, tp, tn = 0, 0, 0, 0
        val_running_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss = criterion(outputs, labels.float().unsqueeze(1))
                val_running_loss += val_loss.item()
                probs = outputs.flatten()
                predictions = (probs > THRESHOLD).int()

                tp += ((predictions == 1) & (labels == 1)).sum().item()
                tn += ((predictions == 0) & (labels == 0)).sum().item()
                fp += ((predictions == 1) & (labels == 0)).sum().item()
                fn += ((predictions == 0) & (labels == 1)).sum().item()

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_running_loss / len(val_loader)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        piece_precision = tp / (tp + fp) if (tp + fp) else 0.0
        empty_recall = tn / (tn + fp) if (tn + fp) else 0.0
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
            f"| Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}% "
            f"| Empty->Piece FP: {fp}"
        )

    print(f"\nTotal benchmark time: {(time.time() - start_time)/60:.2f} minutes")


    # Visualization
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_acc'], marker='o', color='b', label='Train')
    plt.plot(range(1, epochs + 1), history['val_acc'], marker='o', color='g', label='Validation')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history['train_loss'], marker='s', color='r', label='Train')
    plt.plot(range(1, epochs + 1), history['val_loss'], marker='s', color='m', label='Validation')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

   
    print("\nDisplaying 3 random predictions (Real vs Predicted)...")
    plt.figure(figsize=(10, 12))

    for i in range(3):
        idx = random.randint(0, len(val_data) - 1)
        image, label = val_data[idx]
        input_tensor = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            prob = output.item()  # Model already outputs probability
            prediction = 1 if prob > THRESHOLD else 0
            display_prob = prob if prediction == 1 else 1 - prob

        actual_name = "Piece" if label == 1 else "Empty"
        pred_name = "Piece" if prediction == 1 else "Empty"

        # Left: real label
        plt.subplot(3, 2, 2 * i + 1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f"Real: {actual_name}", color='black')
        plt.axis('off')

        # Right: predicted label 
        plt.subplot(3, 2, 2 * i + 2)
        plt.imshow(image.squeeze(), cmap='gray')
        color = 'green' if prediction == label else 'red'
        plt.title(f"Predicted: {pred_name} ({display_prob:.1%})", color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_benchmark()