"""Export trained Swin classifier to ONNX format."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.swin_classifier import SwinClassifier
from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logger
from src.utils.onnx_export import ONNXExporter
from src.utils.seed import set_seed

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/best_swin_classifier.pth",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--quantize", type=str, default=None, choices=["int8", "fp16"])
    parser.add_argument("--benchmark", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = ConfigManager(args.config)
    set_seed(cfg.seed)

    logger = setup_logger("solinfitec", level=cfg.logging.level)
    logger.info("Starting ONNX export")

    device = torch.device("cpu")  # Export on CPU

    # Load model
    model = SwinClassifier(
        model_name=cfg.model.variant,
        num_classes=cfg.model.num_classes,
        pretrained=False,
        hidden_dim=cfg.model.head.hidden_dim,
        dropout=0.0,  # No dropout at inference
        feature_dim=cfg.model.feature_dim,
    )

    if Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    else:
        logger.warning(f"Checkpoint not found: {args.checkpoint}, using random weights")

    model.eval()

    # Export
    output_dir = Path(args.output_dir or cfg.export.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = str(output_dir / "swin_classifier.onnx")

    exporter = ONNXExporter(
        opset_version=cfg.export.opset_version,
        max_diff=cfg.export.max_diff,
    )

    exporter.export_swin(
        model,
        onnx_path,
        input_size=(1, 3, cfg.data.img_size[0], cfg.data.img_size[1]),
    )

    # Validate
    if cfg.export.validate_output:
        valid = exporter.validate(model, onnx_path)
        if not valid:
            logger.error("ONNX validation FAILED")
            return

    # Quantize
    quant_path = None
    if args.quantize or cfg.export.quantization.get("enabled"):
        quant_type = args.quantize or cfg.export.quantization.get("type", "fp16")
        try:
            quant_path = exporter.quantize(onnx_path, quant_type=quant_type)
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")

    # Benchmark
    if args.benchmark or cfg.export.benchmark_latency:
        bench_path = quant_path or onnx_path
        results = exporter.benchmark_latency(bench_path)

        # Save results
        with open(output_dir / "benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)

    logger.info("Export complete.")


if __name__ == "__main__":
    main()
