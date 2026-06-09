# CIFAR-10 Image Classifier with DIP Enhancement and Gradio UI

A modular computer vision project that fine-tunes a ResNet-50 model on CIFAR-10 and compares standard classification with Digital Image Processing (DIP) enhanced inputs.

This project began as a Computer Vision course project and is being refactored into a cleaner ML engineering portfolio project with separate training, evaluation, preprocessing, model, and Gradio inference modules.

---

## Highlights

- Fine-tunes a pretrained ResNet-50 model on CIFAR-10.
- Uses torchvision augmentation and ImageNet normalization.
- Includes DIP preprocessing with histogram equalization, segmentation, edge detection, and morphology helpers.
- Evaluates accuracy, weighted F1, and classification reports.
- Provides a Gradio interface for interactive prediction.
- Keeps training, evaluation, and app launch separate for better maintainability.

---

## Current Result

| Experiment | Result |
|---|---:|
| Best validation accuracy | 84.27% |
| Dataset | CIFAR-10 |
| Model | Fine-tuned ResNet-50 |

The original project reached a best validation accuracy of **84.27%** after 39 epochs. More detailed raw-vs-DIP evaluation metrics should be added after rerunning the refactored pipeline.

---

## Project Structure

```text
Image_Classifier/
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── data_loaders.py
│   ├── datasets.py
│   ├── dip.py
│   ├── evaluate.py
│   ├── model.py
│   └── train.py
├── models/
│   └── .gitkeep
├── results/
│   └── .gitkeep
├── ImageEnhancement.py
├── LICENSE
├── README.md
├── requirements.txt
└── .gitignore
```

> `ImageEnhancement.py` is now a compatibility entrypoint. New development should use the modules inside `src/`.

---

## Installation

```bash
git clone https://github.com/MerveilleDivine/Image_Classifier.git
cd Image_Classifier
pip install -r requirements.txt
```

For GPU training, install a PyTorch build that matches your CUDA setup from the official PyTorch instructions.

---

## Train

```bash
python -m src.train
```

Optional quick test run:

```bash
python -m src.train --epochs 1 --batch-size 64
```

The best checkpoint is saved to:

```text
models/best_resnet50_cifar10.pth
```

---

## Evaluate

```bash
python -m src.evaluate --checkpoint models/best_resnet50_cifar10.pth
```

The evaluation command reports accuracy, weighted F1, and a per-class classification report on both the original and DIP-enhanced CIFAR-10 test sets.

---

## Run the Gradio App

```bash
python -m src.app --checkpoint models/best_resnet50_cifar10.pth
```

Or use the legacy compatibility entrypoint:

```bash
python ImageEnhancement.py
```

If the checkpoint is missing, the app now shows a clear error instead of failing silently.

---

## DIP Pipeline

The preprocessing module includes:

- Gaussian blur
- Histogram equalization
- Otsu threshold-based segmentation
- Morphological cleanup
- Canny or Sobel edge detection helpers

The DIP transform can be used during evaluation or toggled in the Gradio app.

---

## Engineering Notes

This refactor intentionally separates responsibilities:

| Module | Responsibility |
|---|---|
| `src/config.py` | shared paths, constants, class names, training defaults |
| `src/dip.py` | image enhancement and segmentation helpers |
| `src/data_loaders.py` | CIFAR-10 datasets, transforms, and loaders |
| `src/model.py` | ResNet-50 construction and checkpoint loading |
| `src/train.py` | training loop and checkpoint saving |
| `src/evaluate.py` | evaluation metrics and reports |
| `src/app.py` | Gradio inference UI |

---

## Limitations

- CIFAR-10 images are low-resolution, so real-world image performance may differ.
- The trained checkpoint is not committed to GitHub because model artifacts can be large.
- Raw-vs-DIP comparison metrics should be rerun and documented after the refactor.

---

## Future Improvements

- Add lightweight unit tests for DIP functions and inference transforms.
- Add a Python CI workflow for syntax checks.
- Export the model to ONNX.
- Add CSV/JSON evaluation reports.
- Add confusion matrix and prediction-sample images to the README.

---

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.
