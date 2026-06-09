import argparse

import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms

from .config import CHECKPOINT_PATH, CIFAR10_CLASSES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from .dip import transform_pil
from .model import build_inference_model


def build_infer_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def create_predict_fn(checkpoint_path=CHECKPOINT_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_inference_model(checkpoint_path=checkpoint_path, device=device)
    infer_transform = build_infer_transform()

    def predict(image, use_dip=False):
        if image is None:
            raise gr.Error("Please upload an image first.")

        if use_dip:
            image = transform_pil(image)

        image = image.convert("RGB")
        tensor = infer_transform(image).unsqueeze(0).to(device)

        with torch.inference_mode():
            output = model(tensor)
            probabilities = F.softmax(output[0], dim=0)

        top_probabilities, top_indices = torch.topk(probabilities, 3)
        return {
            CIFAR10_CLASSES[index]: float(top_probabilities[position].cpu())
            for position, index in enumerate(top_indices.cpu().tolist())
        }

    return predict


def launch_app(checkpoint_path=CHECKPOINT_PATH, share=False):
    try:
        predict = create_predict_fn(checkpoint_path=checkpoint_path)
    except FileNotFoundError as error:
        message = str(error)

        def predict_missing_checkpoint(image, use_dip=False):
            raise gr.Error(message)

        predict = predict_missing_checkpoint

    interface = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="pil", label="Upload an image"),
            gr.Checkbox(label="Use DIP preprocessing"),
        ],
        outputs=gr.Label(num_top_classes=3, label="Predictions"),
        title="CIFAR-10 Classifier with DIP Enhancement",
        description="Classify images using a ResNet-50 CIFAR-10 model. Toggle DIP preprocessing to compare raw and enhanced inputs.",
    )
    interface.launch(share=share)


def main():
    parser = argparse.ArgumentParser(description="Launch the Gradio inference app.")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--share", action="store_true", help="Create a public Gradio sharing link.")
    args = parser.parse_args()
    launch_app(checkpoint_path=args.checkpoint, share=args.share)


if __name__ == "__main__":
    main()
