import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from .config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    MODEL_DIR,
    NUM_EPOCHS,
    WEIGHT_DECAY,
)
from .data_loaders import get_cifar10_loaders
from .evaluate import evaluate_model
from .model import build_model


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.show()


def train(args):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds, train_loader, test_loader, dip_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(pretrained=True, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_accuracy = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        model.train()
        running_loss, running_correct = 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()

        scheduler.step()
        train_loss = running_loss / len(train_ds)
        train_accuracy = running_correct / len(train_ds)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%")

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.inference_mode():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss = val_loss / len(test_loader.dataset)
        val_accuracy = val_correct / len(test_loader.dataset)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), args.checkpoint)
            print(f"Saved best model with validation accuracy {best_accuracy * 100:.2f}%")

    if args.plot:
        plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

    print("\n===== Evaluation on Original Test Set =====")
    evaluate_model(model, test_loader, train_ds.classes, device)
    print("\n===== Evaluation on DIP Test Set =====")
    evaluate_model(model, dip_loader, train_ds.classes, device)


def main():
    parser = argparse.ArgumentParser(description="Train a ResNet-50 CIFAR-10 classifier.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
