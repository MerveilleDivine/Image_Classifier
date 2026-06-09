import argparse

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .config import BATCH_SIZE, CHECKPOINT_PATH
from .data_loaders import get_cifar10_loaders
from .model import build_model, load_checkpoint


def evaluate_model(model, loader, class_names, device):
    model.eval()
    predictions, labels = [], []

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            batch_predictions = outputs.argmax(1).cpu().numpy()
            predictions.extend(batch_predictions)
            labels.extend(targets.numpy())

    accuracy = accuracy_score(labels, predictions)
    weighted_f1 = f1_score(labels, predictions, average="weighted")
    print(f"Accuracy: {accuracy * 100:.2f}% | Weighted F1: {weighted_f1 * 100:.2f}%")
    print(classification_report(labels, predictions, target_names=class_names))
    return accuracy, weighted_f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate the CIFAR-10 image classifier.")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH), help="Path to trained model checkpoint.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, _, test_loader, dip_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(pretrained=False, freeze_backbone=False)
    model = load_checkpoint(model, checkpoint_path=args.checkpoint, map_location=device)
    model.to(device)

    print("\n===== Evaluation on Original Test Set =====")
    evaluate_model(model, test_loader, train_ds.classes, device)

    print("\n===== Evaluation on DIP Test Set =====")
    evaluate_model(model, dip_loader, train_ds.classes, device)


if __name__ == "__main__":
    main()
