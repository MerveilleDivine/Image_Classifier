# CIFAR-10 Image Classifier with DIP Enhancement and Gradio UI

This project is a comprehensive image classification pipeline that combines deep learning with Digital Image Processing (DIP) techniques to classify images from the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. It includes training a fine-tuned ResNet-50 model, applying image preprocessing techniques like segmentation and edge detection, and deploying a Gradio-based web interface for live prediction.

---

## 🌍 Live Demo

You can launch a live demo using the [Gradio](https://www.gradio.app/) interface at the end of the script.

---

## 🔧 Features

* Deep learning with ResNet-50
* Data augmentation and normalization using torchvision
* DIP-based preprocessing (histogram equalization, segmentation, morphology)
* Metric plotting and model evaluation (Accuracy, F1, Confusion Matrix)
* Gradio UI for user-friendly image prediction

---

## 📚 Project Structure

* `enhance_image` / `segment_image`: Basic DIP techniques
* `train_transform`, `dip_test_transform`: Image preprocessing pipelines
* `main()`: Full model training and evaluation logic
* `predict()`: Gradio-compatible prediction logic
* `evaluate_model()`, `visualize_predictions()`: Metrics and visual outputs

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
https://github.com/MerveilleDivine/Image_Classifier.git
cd Image_Classifier
```

### 2. Install Dependencies

Use Google Colab (recommended) or install locally:

```bash
pip install --upgrade torch torchvision gradio opencv-python scikit-learn matplotlib
```

---

## 🔬 How It Works

### DIP Pipeline

Each input image can optionally go through:

* Gaussian blur + Histogram Equalization (Contrast)
* Otsu's Threshold-based Segmentation
* Morphological Opening to clean noise

### Training Workflow

1. Loads CIFAR-10 dataset (original and DIP-enhanced test set)
2. Applies heavy data augmentation during training
3. Fine-tunes a pretrained ResNet-50 on CIFAR-10
4. Tracks metrics, saves best model, and evaluates both original and DIP-enhanced test sets
5. Visualizes training loss/accuracy curves and sample predictions

---

## 📊 Model Performance

> Note: Actual metrics are printed after training completes.

---

## 📷 Gradio Inference

Launch an interactive UI for image classification:

```bash
python ImageEnhancement.py
```

Or use in Colab:

```python
# At the end of the notebook
Interface.launch(share=True)
```

### Inputs:

* Upload an image (32x32 CIFAR-10 format)
* Optionally apply DIP preprocessing

### Outputs:

* Top 3 predicted classes with confidence scores

---

## 🩼 Credits

Developed by Mervine Muganguzi.
Inspired by academic research on image enhancement and its effect on classification accuracy.

---

## 🌟 TODO / Improvements

* Support for custom datasets
* Export trained model to ONNX
* Add adversarial robustness testing
* Improve segmentation accuracy

---

## 🚫 Disclaimer

This project is meant for educational and experimental purposes. CIFAR-10 images are low-resolution and results may vary in real-world tasks.

---

## ✈️ License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.
