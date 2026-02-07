"""Inference pipeline supporting PyTorch and ONNX backends.

Usage:
    python predict.py --image path/to/image.jpg --backend pytorch
    python predict.py --image path/to/image.jpg --backend onnx
    python predict.py --image_dir path/to/images/ --backend onnx --batch_size 16
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from src.features.augmentation import get_val_transforms
from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logger
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory of images")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/best_swin_classifier.pth",
    )
    parser.add_argument(
        "--onnx_model", type=str, default="models/final/swin_classifier.onnx"
    )
    parser.add_argument(
        "--backend", type=str, default="pytorch", choices=["pytorch", "onnx"]
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    return parser.parse_args()


class PyTorchPredictor:
    """PyTorch inference backend."""

    def __init__(self, model, device, transform, class_names):
        self.model = model
        self.device = device
        self.transform = transform
        self.class_names = class_names

    def predict_single(self, image_path: str) -> Dict:
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        tensor = self.transform(image=img_np)["image"].unsqueeze(0).to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
        latency = (time.perf_counter() - start) * 1000

        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

        return {
            "image": image_path,
            "predicted_class": self.class_names[pred_idx],
            "confidence": round(confidence, 4),
            "all_probs": {
                name: round(probs[0, i].item(), 4)
                for i, name in enumerate(self.class_names)
            },
            "latency_ms": round(latency, 2),
            "backend": "pytorch",
        }


class ONNXPredictor:
    """ONNX Runtime inference backend."""

    def __init__(self, onnx_path: str, transform, class_names):
        import onnxruntime as ort

        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.transform = transform
        self.class_names = class_names

    def predict_single(self, image_path: str) -> Dict:
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        tensor = self.transform(image=img_np)["image"].unsqueeze(0).numpy()

        start = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: tensor})
        latency = (time.perf_counter() - start) * 1000

        logits = outputs[0][0]
        probs = np.exp(logits) / np.exp(logits).sum()
        pred_idx = probs.argmax()
        confidence = probs[pred_idx]

        return {
            "image": image_path,
            "predicted_class": self.class_names[pred_idx],
            "confidence": round(float(confidence), 4),
            "all_probs": {
                name: round(float(probs[i]), 4)
                for i, name in enumerate(self.class_names)
            },
            "latency_ms": round(latency, 2),
            "backend": "onnx",
        }


def get_class_names(data_dir: str, skip_nested: str = "PlantVillage") -> List[str]:
    """Get class names from dataset directory."""
    dataset_dir = Path(data_dir)
    names = []
    for d in sorted(dataset_dir.iterdir()):
        if d.is_dir() and d.name != skip_nested:
            names.append(d.name)
    return names


def main():
    args = parse_args()
    cfg = ConfigManager(args.config)
    set_seed(cfg.seed)

    logger = setup_logger("solinfitec", level=cfg.logging.level)

    # Get class names
    data_dir = Path(cfg.data.raw_dir) / cfg.data.dataset_name
    class_names = get_class_names(str(data_dir))
    if not class_names:
        class_names = [f"class_{i}" for i in range(cfg.model.num_classes)]

    transform = get_val_transforms(img_size=cfg.data.img_size[0])

    # Create predictor
    if args.backend == "pytorch":
        from src.models.swin_classifier import SwinClassifier

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SwinClassifier(
            model_name=cfg.model.variant,
            num_classes=len(class_names),
            pretrained=False,
            hidden_dim=cfg.model.head.hidden_dim,
            feature_dim=cfg.model.feature_dim,
        ).to(device)

        if Path(args.checkpoint).exists():
            ckpt = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        predictor = PyTorchPredictor(model, device, transform, class_names)
    else:
        predictor = ONNXPredictor(args.onnx_model, transform, class_names)

    # Collect images
    image_paths = []
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        img_dir = Path(args.image_dir)
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            image_paths.extend(str(p) for p in img_dir.glob(ext))

    if not image_paths:
        logger.error("No images found. Use --image or --image_dir.")
        return

    logger.info(f"Running inference on {len(image_paths)} images ({args.backend})")

    # Run predictions
    results = []
    total_latency = 0

    for img_path in image_paths:
        result = predictor.predict_single(img_path)
        results.append(result)
        total_latency += result["latency_ms"]
        logger.info(
            f"{Path(img_path).name}: {result['predicted_class']} "
            f"({result['confidence']:.3f}) [{result['latency_ms']:.1f}ms]"
        )

    # Grad-CAM (PyTorch only)
    if args.gradcam and args.backend == "pytorch":
        from src.visualization.gradcam import SwinGradCAM

        gradcam = SwinGradCAM(predictor.model)
        for i, (img_path, result) in enumerate(zip(image_paths, results)):
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            tensor = transform(image=img_np)["image"].unsqueeze(0).to(device)

            heatmap = gradcam.generate(tensor)
            out_path = f"reports/gradcam_{Path(img_path).stem}.png"
            gradcam.visualize(img_np, heatmap, save_path=out_path)
            results[i]["gradcam_path"] = out_path

    # Summary
    avg_latency = total_latency / len(results) if results else 0
    logger.info(f"\nAverage latency: {avg_latency:.1f}ms")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
