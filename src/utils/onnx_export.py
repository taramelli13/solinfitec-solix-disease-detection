"""ONNX export utilities: export, validate, quantize, and benchmark."""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("solinfitec.onnx_export")


class ONNXExporter:
    """Export PyTorch models to ONNX with validation and benchmarking.

    Args:
        opset_version: ONNX opset version (14 for Swin compatibility).
        max_diff: Maximum allowed difference between PyTorch and ONNX outputs.
    """

    def __init__(self, opset_version: int = 14, max_diff: float = 1e-5):
        self.opset_version = opset_version
        self.max_diff = max_diff

    def export_swin(
        self,
        model: nn.Module,
        output_path: str,
        input_size: Tuple[int, ...] = (1, 3, 224, 224),
        dynamic_batch: bool = True,
    ) -> str:
        """Export Swin classifier to ONNX.

        Args:
            model: SwinClassifier model.
            output_path: Path for output .onnx file.
            input_size: Input tensor shape.
            dynamic_batch: Whether to allow dynamic batch size.

        Returns:
            Path to exported ONNX file.
        """
        model.eval()
        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_size, device=device)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dynamic_axes = {}
        if dynamic_batch:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }

        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            opset_version=self.opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

        logger.info(f"ONNX model exported to {output_path}")

        # Get file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"ONNX model size: {size_mb:.1f} MB")

        return str(output_path)

    def validate(
        self,
        pytorch_model: nn.Module,
        onnx_path: str,
        input_size: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> bool:
        """Validate ONNX output matches PyTorch output.

        Returns:
            True if outputs match within max_diff tolerance.
        """
        import onnxruntime as ort

        pytorch_model.eval()
        device = next(pytorch_model.parameters()).device
        dummy_input = torch.randn(*input_size, device=device)

        # PyTorch inference
        with torch.no_grad():
            pytorch_output = pytorch_model(dummy_input).cpu().numpy()

        # ONNX inference
        session = ort.InferenceSession(onnx_path)
        ort_input = {session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        onnx_output = session.run(None, ort_input)[0]

        # Compare
        diff = np.abs(pytorch_output - onnx_output).max()
        passed = diff < self.max_diff

        logger.info(
            f"ONNX validation: max_diff={diff:.2e}, "
            f"threshold={self.max_diff:.2e}, "
            f"{'PASSED' if passed else 'FAILED'}"
        )
        return passed

    def quantize(
        self,
        onnx_path: str,
        output_path: Optional[str] = None,
        quant_type: str = "fp16",
    ) -> str:
        """Quantize ONNX model to INT8 or FP16.

        Args:
            onnx_path: Input ONNX model path.
            output_path: Output path (default: adds suffix).
            quant_type: 'int8' or 'fp16'.

        Returns:
            Path to quantized model.
        """
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic

        if output_path is None:
            p = Path(onnx_path)
            output_path = str(p.parent / f"{p.stem}_{quant_type}{p.suffix}")

        if quant_type == "int8":
            quantize_dynamic(
                onnx_path,
                output_path,
                weight_type=QuantType.QInt8,
            )
        elif quant_type == "fp16":
            model = onnx.load(onnx_path)
            from onnxruntime.transformers import float16

            model_fp16 = float16.convert_float_to_float16(model)
            onnx.save(model_fp16, output_path)
        else:
            raise ValueError(f"Unknown quantization type: {quant_type}")

        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        logger.info(f"Quantized model ({quant_type}): {output_path} ({size_mb:.1f} MB)")
        return output_path

    def benchmark_latency(
        self,
        onnx_path: str,
        input_size: Tuple[int, ...] = (1, 3, 224, 224),
        num_runs: int = 100,
        warmup: int = 10,
    ) -> dict:
        """Benchmark ONNX inference latency.

        Returns:
            Dict with mean, std, min, max latency in ms.
        """
        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path)
        dummy_input = np.random.randn(*input_size).astype(np.float32)
        input_name = session.get_inputs()[0].name

        # Warmup
        for _ in range(warmup):
            session.run(None, {input_name: dummy_input})

        # Benchmark
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            session.run(None, {input_name: dummy_input})
            latencies.append((time.perf_counter() - start) * 1000)

        results = {
            "mean_ms": np.mean(latencies),
            "std_ms": np.std(latencies),
            "min_ms": np.min(latencies),
            "max_ms": np.max(latencies),
            "p50_ms": np.percentile(latencies, 50),
            "p95_ms": np.percentile(latencies, 95),
            "num_runs": num_runs,
        }

        logger.info(
            f"Latency benchmark: mean={results['mean_ms']:.1f}ms, "
            f"p95={results['p95_ms']:.1f}ms, "
            f"min={results['min_ms']:.1f}ms, max={results['max_ms']:.1f}ms"
        )
        return results
