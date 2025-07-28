# ===== Colab Setup: Install dependencies =====
# Run this cell first to install all required libraries
!pip install --upgrade torch torchvision gradio opencv-python scikit-learn matplotlib

# ===== Imports =====
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
import numpy as np
from matplotlib import pyplot as plt
plt.ion()  # Enable interactive mode
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from PIL import Image
import cv2
import gradio as gr
import torch.nn.functional as F  # For softmax probabilities
# ===== Constants =====
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

# ===== 1. DIP Enhancement, Segmentation, Edges, Morphology =====
def enhance_image(cv_img):
    """
    Apply Gaussian blur and histogram equalization (contrast enhancement).
    """
    blur = cv2.GaussianBlur(cv_img, (5, 5), 0)
    yuv = cv2.cvtColor(blur, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def segment_image(cv_img):
    """Simple segmentation via Otsu's thresholding."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.merge([mask, mask, mask])

def edge_detect(cv_img, method='canny'):
    """Return edge map using Canny or Sobel."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    if method == 'sobel':
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        edges = np.uint8(np.clip(mag / mag.max() * 255, 0, 255))
    else:
        med = np.median(gray)
        lower = int(max(0, 0.7 * med))
        upper = int(min(255, 1.3 * med))
        edges = cv2.Canny(gray, lower, upper)
    return edges

def morph_cleanup(mask, use_open=True, kernel_size=3, iterations=1):
    """Apply opening (erosion then dilation) or closing."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if use_open:
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    else:
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

# Combined DIP pipeline for PIL images
def transform_pil(img):
    """Enhance, segment, and clean via morphology."""
    img_np = np.array(img)
    enhanced = enhance_image(img_np)
    mask = segment_image(enhanced)
    clean_mask = morph_cleanup(mask[:,:,0], use_open=True)
    clean_mask = cv2.merge([clean_mask]*3)
    segmented = cv2.bitwise_and(enhanced, clean_mask)
    return Image.fromarray(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))

# Compare original vs equalized contrast
import matplotlib.pyplot as plt
def compare_eq_vs_raw(pil_img):
    img_np = np.array(pil_img)
    yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    eq_rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    fig, axs = plt.subplots(1, 2, figsize=(8,4))
    axs[0].imshow(pil_img); axs[0].set_title('Original'); axs[0].axis('off')
    axs[1].imshow(eq_rgb); axs[1].set_title('Histogram Equalized'); axs[1].axis('off')
    plt.tight_layout(); plt.show()
 #===== 2. Transforms & Augmentation =====
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
    transforms.RandomErasing(p=0.5)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

dip_test_transform = transforms.Compose([
    transforms.Lambda(transform_pil),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])
# ===== 3. Plot Metrics =====
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss over Epochs')
    plt.legend(); plt.show()
    plt.figure(figsize=(8,4))
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Validation Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy over Epochs')
    plt.legend(); plt.show()
    # ===== 4. Evaluation =====
def evaluate_model(model, loader, class_names, device):
    model.eval()
    preds, labels = [], []
    with torch.inference_mode():
        for imgs, labs in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            pred = out.argmax(1).cpu().numpy()
            preds.extend(pred); labels.extend(labs.numpy())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    print(f'Accuracy: {acc*100:.2f}% | F1: {f1*100:.2f}%')
    print(classification_report(labels, preds, target_names=class_names))
    return acc, f1
# ===== 5. Visualization =====
def visualize_predictions(model, loader, class_names, device, num_images=16):
    model.eval()
    imgs, labs = next(iter(loader))
    imgs, labs = imgs.to(device), labs.to(device)
    with torch.inference_mode():
        out = model(imgs)
        pred = out.argmax(1)
    imgs, pred, labs = imgs.cpu(), pred.cpu(), labs.cpu()
    n = min(num_images, len(imgs)); cols = 4; rows = (n + cols - 1) // cols
    plt.figure(figsize=(12, 3 * rows))
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        img = imgs[i].permute(1,2,0).numpy()
        img = np.clip(img * imagenet_std + imagenet_mean, 0, 1)
        plt.imshow(img)
        plt.title(f"T:{class_names[labs[i]]}\nP:{class_names[pred[i]]}")
        plt.axis('off')
    plt.tight_layout(); plt.show()
# ===== 6. Main Training =====
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_ds  = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    dip_ds   = datasets.CIFAR10(root='./data', train=False, download=False, transform=dip_test_transform)

    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    test_ld  = torch.utils.data.DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=4)
    dip_ld   = torch.utils.data.DataLoader(dip_ds,  batch_size=128, shuffle=False, num_workers=4)

    # Build and freeze parts of ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for name, param in model.named_parameters():
        if 'layer3' not in name and 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    num_epochs = 100
    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    checkpoint_path = 'best_resnet50_cifar10.pth'

    for epoch in range(1, num_epochs + 1):
        print(f"🚀 Epoch {epoch}/{num_epochs}")
        # --- Training ---
        model.train()
        running_loss, running_correct = 0.0, 0
        for imgs, labels in train_ld:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
        scheduler.step()
        train_loss = running_loss / len(train_ds)
        train_acc = running_correct / len(train_ds)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(f"✅ Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")

        # --- Validation ---
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.inference_mode():
            for imgs, labels in test_ld:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
        val_loss = val_loss / len(test_ds)
        val_acc = val_correct / len(test_ds)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"🧪 Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"🔖 Saved Best Model (Val Acc: {best_acc*100:.2f}%)")

    # Plot training metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

    # Load best model for final evaluation
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("\n===== Evaluation on Original Test Set =====")
    acc_raw, f1_raw = evaluate_model(model, test_ld, train_ds.classes, device)
    print("\n===== Evaluation on DIP Test Set =====")
    acc_dip, f1_dip = evaluate_model(model, dip_ld, train_ds.classes, device)
    print(f"\n✅ Improvement with DIP: {(acc_dip - acc_raw)*100:.2f}%")

    # Visualize predictions
    visualize_predictions(model, test_ld, train_ds.classes, device)
    visualize_predictions(model, dip_ld, train_ds.classes, device)

# ===== 7. Execute Training =====
if __name__ == '__main__':
    main()
# ===== 8. Gradio Demo Interface =====
# Load trained model for inference
inference_model = models.resnet50(weights=None)
inference_model.fc = nn.Linear(inference_model.fc.in_features, 10)
inference_model.load_state_dict(torch.load('best_resnet50_cifar10.pth', map_location='cpu'))
inference_model.eval()

# Inference preprocessing
def infer_transform(img):
    img = img.convert('RGB')
    img = img.resize((32, 32))
    tensor = transforms.ToTensor()(img)
    tensor = transforms.Normalize(imagenet_mean,imagenet_std)(tensor)
    return tensor.unsqueeze(0)

classes = datasets.CIFAR10(root='./data', train=False, download=True).classes

def predict(image, use_DIP=False):
    """Predict CIFAR-10 labels with optional DIP preprocessing."""
    if use_DIP:
        image = transform_pil(image)
    x = infer_transform(image)
    with torch.inference_mode():
        out = inference_model(x)
        probs = F.softmax(out[0], dim=0)
    top_p, top_i = torch.topk(probs, 3)
    return {classes[i]: float(top_p[idx]) for idx, i in enumerate(top_i)}

# Launch Gradio interface
gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload CIFAR-10 Image"),
        gr.Checkbox(label="Use DIP Preprocessing")
    ],
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="CIFAR-10 Classifier with Augmentation & DIP",
    description="Toggle DIP preprocessing to compare raw vs enhanced+segmented."
).launch(share=True)
