import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from accelerate import Accelerator
from cnn_model import get_resnet18
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR

# Define a custom learning rate scheduler
def lr_lambda(epoch):
    """
    Custom learning rate schedule:
    Adjusts the learning rate based on the epoch number.
    """
    if epoch < 20:
        return 1.0  # Keep the learning rate constant
    elif 20 <= epoch < 30:
        return 0.1  # Reduce the learning rate by a factor of 10
    elif 30 <= epoch < 40:
        return 0.01  # Further reduce the learning rate
    elif 40 <= epoch < 50:
        return 0.001
    else:
        return 0.0001  # Final reduction in learning rate

# Plot training and testing metrics over time
def plot_metrics_with_time(train_errors, train_accuracies, test_accuracies, times, save_path="metrics_plot.png"):
    """
    Creates plots for train/test metrics over time and saves them to a file.
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

    # Plot cumulative time per epoch
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
    Trains and evaluates the ResNet18 model on CIFAR-10 using Hugging Face Accelerate for distributed training.
    Includes support for learning rate scheduling and plotting performance metrics.
    """
    # Initialize Accelerator for distributed training
    accelerator = Accelerator()

    # Initialize the model, optimizer, and criterion
    model = get_resnet18(num_classes=10, pretrained=False)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    # Define a learning rate scheduler
    scheduler = LambdaLR(optimizer, lr_lambda)

    # Define data transformations for CIFAR-10
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

    # Load training and testing datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)

    # Create data loaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)

    # Prepare model, optimizer, and data loaders with Accelerator
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)

    # Initialize variables to store training and testing metrics
    train_errors = []
    train_accuracies = []
    test_accuracies = []
    cumulative_times = []  # Tracks cumulative training time
    total_time = 0
    epochs = 50  # Number of epochs to train

    start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        correct = 0
        total = 0
        total_loss = 0
        for inputs, targets in train_loader:
            # Move inputs and targets to the device
            inputs, targets = inputs.to(accelerator.device), targets.to(accelerator.device)

            # Zero out the gradients, perform forward and backward passes, and update the optimizer
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            accelerator.backward(loss)
            optimizer.step()

            # Track statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Update training metrics
        train_errors.append(total_loss / len(train_loader))
        train_accuracies.append(100.0 * correct / total)

        # Step the learning rate scheduler
        scheduler.step()

        # Testing phase
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(accelerator.device), targets.to(accelerator.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # Update test accuracy
        test_accuracy = 100.0 * correct / total
        test_accuracies.append(test_accuracy)

        # Calculate time taken for this epoch
        epoch_time = time.time() - epoch_start_time
        total_time += epoch_time
        cumulative_times.append(total_time)

        # Log metrics for the current epoch
        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {train_errors[-1]:.4f}, "
                  f"Train Accuracy = {train_accuracies[-1]:.2f}%, Test Accuracy = {test_accuracy:.2f}%, "
                  f"Epoch Time = {epoch_time:.2f}s, Total Time = {total_time:.2f}s")

    # Save and plot metrics if this is the main process
    if accelerator.is_main_process:
        plot_metrics_with_time(train_errors, train_accuracies, test_accuracies, cumulative_times, save_path="training_metrics_with_time.png")


if __name__ == "__main__":
    train_and_evaluate()
