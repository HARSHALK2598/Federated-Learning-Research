import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from accelerate import Accelerator
from cnn_model import get_resnet18
from cnn_model import get_resnet50
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR


# Define a custom learning rate schedule
def lr_lambda(epoch):
    """
    Custom learning rate schedule:
    Adjusts the learning rate based on the epoch number.
    """
    if epoch < 20:
        return 0.1  # Initial learning rate
    elif 20 <= epoch < 30:
        return 0.01  # Reduced learning rate
    elif 30 <= epoch < 40:
        return 0.001  # Further reduction
    else:
        return 0.0001  # Minimal learning rate

# Function to plot training metrics over time
def plot_metrics_with_time(train_errors, train_accuracies, test_accuracies, times, save_path="metrics_plot.png"):
    """
    Plots training error, training accuracy, and test accuracy over time.
    Saves the plot to the specified path.
    """
    plt.figure(figsize=(12, 8))

    # Plot training error vs time
    plt.subplot(2, 2, 1)
    plt.plot(times, train_errors, marker='o', label="Train Error")
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.title('Training Error vs Time')
    plt.legend()

    # Plot training accuracy vs time
    plt.subplot(2, 2, 2)
    plt.plot(times, train_accuracies, marker='o', label="Train Accuracy")
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy vs Time')
    plt.legend()

    # Plot test accuracy vs time
    plt.subplot(2, 2, 3)
    plt.plot(times, test_accuracies, marker='o', label="Test Accuracy")
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy vs Time')
    plt.legend()

    # Plot cumulative training time per epoch
    plt.subplot(2, 2, 4)
    plt.plot(range(len(times)), times, marker='o', label="Cumulative Time")
    plt.xlabel('Epochs')
    plt.ylabel('Time (s)')
    plt.title('Cumulative Time vs Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Metrics plot saved to {save_path}")

# Main training and evaluation function
def train_and_evaluate():
    """
    Trains and evaluates a ResNet18 model on the CIFAR-10 dataset.
    Uses FP16 precision for efficiency and Hugging Face Accelerator for distributed training.
    """
    # Initialize Accelerator with FP16 precision
    accelerator = Accelerator(mixed_precision="fp16")

    # Initialize the model, optimizer, and loss function
    model = get_resnet18(num_classes=10, pretrained=False)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = LambdaLR(optimizer, lr_lambda)  # Learning rate scheduler

    # Define data augmentations and transformations for training and testing
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Data augmentation: random horizontal flip
        transforms.RandomCrop(32, padding=4),  # Data augmentation: random cropping
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalize with CIFAR-10 stats
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR-10 datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)

    # Create data loaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)

    # Prepare model, optimizer, and data loaders using Accelerator
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)

    # Variables to store metrics
    train_errors = []
    train_accuracies = []
    test_accuracies = []
    cumulative_times = []  # Tracks cumulative time per epoch
    total_time = 0  # Total time taken
    epochs = 50  # Number of training epochs

    start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        correct = 0
        total = 0
        total_loss = 0
        for inputs, targets in train_loader:
            # Move data to the appropriate device and match FP16 precision
            inputs = inputs.to(accelerator.device, dtype=torch.float16)
            targets = targets.to(accelerator.device)

            # Zero the gradients, perform forward and backward passes, and update the weights
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            accelerator.backward(loss)  # Backpropagation with FP16 support
            optimizer.step()

            # Accumulate statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Log training metrics
        train_errors.append(total_loss / len(train_loader))
        train_accuracies.append(100.0 * correct / total)
        scheduler.step()  # Update the learning rate

        # Testing phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                # Move data to the appropriate device and match FP16 precision
                inputs = inputs.to(accelerator.device, dtype=torch.float16)
                targets = targets.to(accelerator.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # Log test accuracy
        test_accuracy = 100.0 * correct / total
        test_accuracies.append(test_accuracy)

        # Calculate epoch time and update cumulative time
        epoch_time = time.time() - epoch_start_time
        total_time += epoch_time
        cumulative_times.append(total_time)

        # Print metrics for the current epoch
        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {train_errors[-1]:.4f}, "
                  f"Train Accuracy = {train_accuracies[-1]:.2f}%, Test Accuracy = {test_accuracy:.2f}%, "
                  f"Epoch Time = {epoch_time:.2f}s, Total Time = {total_time:.2f}s")

    # Plot and save metrics
    if accelerator.is_main_process:
        plot_metrics_with_time(train_errors, train_accuracies, test_accuracies, cumulative_times, save_path="training_metrics_with_time.png")


# Entry point of the script
if __name__ == "__main__":
    train_and_evaluate()
