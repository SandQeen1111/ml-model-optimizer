"""
ML Model Optimizer
==================
Automated model optimization toolkit that compares PyTorch, ONNX, 
TensorRT, and CoreML formats for latency, size, and accuracy.

Author: SQ1111
License: MIT
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from pathlib import Path


# ============================================================
# Data Classes
# ============================================================
@dataclass
class OptimizationResult:
    """Result of a single optimization pass."""
    format_name: str
    file_path: str
    size_mb: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_fps: float
    accuracy: Optional[float] = None
    compression_ratio: Optional[float] = None


@dataclass
class OptimizationReport:
    """Complete optimization report across all formats."""
    model_name: str
    input_shape: list
    results: List[OptimizationResult] = field(default_factory=list)
    best_latency: Optional[str] = None
    best_size: Optional[str] = None
    recommendation: Optional[str] = None


# ============================================================
# Base Optimizer
# ============================================================
class BaseOptimizer:
    """Base class for model format optimizers."""
    
    def __init__(self, output_dir: str = "optimized"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def benchmark(self, predict_fn, input_data, num_runs=200, warmup=20):
        """Generic benchmark for any predict function."""
        # Warmup
        for _ in range(warmup):
            predict_fn(input_data)
        
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            predict_fn(input_data)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
        
        latencies.sort()
        n = len(latencies)
        mean = sum(latencies) / n
        
        return {
            "p50_ms": round(latencies[n // 2], 3),
            "p95_ms": round(latencies[int(n * 0.95)], 3),
            "p99_ms": round(latencies[int(n * 0.99)], 3),
            "mean_ms": round(mean, 3),
            "min_ms": round(latencies[0], 3),
            "max_ms": round(latencies[-1], 3),
            "throughput_fps": round(1000 / mean, 1),
        }


# ============================================================
# PyTorch Optimizer
# ============================================================
class PyTorchOptimizer(BaseOptimizer):
    """Optimize and benchmark PyTorch models."""
    
    def optimize(self, model: nn.Module, input_shape: tuple, 
                 device: str = "cpu") -> OptimizationResult:
        """Save and benchmark PyTorch model."""
        model.eval()
        
        # Save model
        save_path = self.output_dir / "model_pytorch.pth"
        torch.save(model.state_dict(), save_path)
        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        
        # TorchScript trace for optimization
        dummy = torch.randn(1, *input_shape, device=device)
        traced = torch.jit.trace(model, dummy)
        traced_path = self.output_dir / "model_traced.pt"
        torch.jit.save(traced, str(traced_path))
        
        # Benchmark
        def predict(x):
            with torch.no_grad():
                return model(x)
        
        dummy_bench = torch.randn(1, *input_shape, device=device)
        metrics = self.benchmark(predict, dummy_bench)
        
        return OptimizationResult(
            format_name="PyTorch FP32",
            file_path=str(save_path),
            size_mb=round(size_mb, 2),
            latency_p50_ms=metrics["p50_ms"],
            latency_p95_ms=metrics["p95_ms"],
            latency_p99_ms=metrics["p99_ms"],
            throughput_fps=metrics["throughput_fps"],
        )


# ============================================================
# ONNX Optimizer
# ============================================================
class ONNXOptimizer(BaseOptimizer):
    """Export, optimize, and benchmark ONNX models."""
    
    def optimize(self, model: nn.Module, input_shape: tuple,
                 device: str = "cpu", quantize: bool = True) -> List[OptimizationResult]:
        """Export to ONNX with optional INT8 quantization."""
        import onnxruntime as ort
        
        model.eval()
        results = []
        
        # Export FP32
        fp32_path = self.output_dir / "model_fp32.onnx"
        dummy = torch.randn(1, *input_shape, device=device)
        
        torch.onnx.export(
            model, dummy, str(fp32_path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            do_constant_folding=True,
        )
        
        fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
        
        # Benchmark FP32
        sess_fp32 = ort.InferenceSession(str(fp32_path))
        dummy_np = dummy.cpu().numpy()
        
        def predict_fp32(x):
            return sess_fp32.run(None, {"input": x})
        
        metrics_fp32 = self.benchmark(predict_fp32, dummy_np)
        
        results.append(OptimizationResult(
            format_name="ONNX FP32",
            file_path=str(fp32_path),
            size_mb=round(fp32_size, 2),
            latency_p50_ms=metrics_fp32["p50_ms"],
            latency_p95_ms=metrics_fp32["p95_ms"],
            latency_p99_ms=metrics_fp32["p99_ms"],
            throughput_fps=metrics_fp32["throughput_fps"],
        ))
        
        # Quantize to INT8
        if quantize:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            int8_path = self.output_dir / "model_int8.onnx"
            quantize_dynamic(
                model_input=str(fp32_path),
                model_output=str(int8_path),
                weight_type=QuantType.QInt8,
            )
            
            int8_size = os.path.getsize(int8_path) / (1024 * 1024)
            
            sess_int8 = ort.InferenceSession(str(int8_path))
            
            def predict_int8(x):
                return sess_int8.run(None, {"input": x})
            
            metrics_int8 = self.benchmark(predict_int8, dummy_np)
            
            results.append(OptimizationResult(
                format_name="ONNX INT8",
                file_path=str(int8_path),
                size_mb=round(int8_size, 2),
                latency_p50_ms=metrics_int8["p50_ms"],
                latency_p95_ms=metrics_int8["p95_ms"],
                latency_p99_ms=metrics_int8["p99_ms"],
                throughput_fps=metrics_int8["throughput_fps"],
                compression_ratio=round(fp32_size / int8_size, 1),
            ))
        
        return results


# ============================================================
# Report Generator
# ============================================================
class ReportGenerator:
    """Generate optimization comparison reports."""
    
    @staticmethod
    def generate(report: OptimizationReport) -> str:
        """Generate formatted comparison report."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"  MODEL OPTIMIZATION REPORT — {report.model_name}")
        lines.append(f"  Input Shape: {report.input_shape}")
        lines.append("=" * 70)
        lines.append("")
        
        # Table header
        header = (
            f"  {'Format':<16} {'Size':>8} {'P50':>8} {'P95':>8} "
            f"{'P99':>8} {'FPS':>8} {'Ratio':>8}"
        )
        lines.append(header)
        lines.append("  " + "─" * 64)
        
        # Find reference size
        ref_size = report.results[0].size_mb if report.results else 1
        
        for r in report.results:
            ratio = f"{ref_size/r.size_mb:.1f}x" if r.size_mb > 0 else "—"
            lines.append(
                f"  {r.format_name:<16} {r.size_mb:>6.1f}MB "
                f"{r.latency_p50_ms:>6.1f}ms {r.latency_p95_ms:>6.1f}ms "
                f"{r.latency_p99_ms:>6.1f}ms {r.throughput_fps:>6.0f}/s "
                f"{ratio:>7}"
            )
        
        lines.append("")
        
        # Find best
        if report.results:
            best_lat = min(report.results, key=lambda r: r.latency_p50_ms)
            best_size = min(report.results, key=lambda r: r.size_mb)
            best_fps = max(report.results, key=lambda r: r.throughput_fps)
            
            lines.append(f"  🏆 Fastest:    {best_lat.format_name} ({best_lat.latency_p50_ms}ms)")
            lines.append(f"  📦 Smallest:   {best_size.format_name} ({best_size.size_mb}MB)")
            lines.append(f"  ⚡ Best FPS:   {best_fps.format_name} ({best_fps.throughput_fps}/s)")
            
            report.best_latency = best_lat.format_name
            report.best_size = best_size.format_name
        
        lines.append("=" * 70)
        return "\n".join(lines)
    
    @staticmethod
    def save_json(report: OptimizationReport, path: str):
        """Save report as JSON."""
        data = {
            "model_name": report.model_name,
            "input_shape": report.input_shape,
            "results": [asdict(r) for r in report.results],
            "best_latency": report.best_latency,
            "best_size": report.best_size,
            "recommendation": report.recommendation,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ============================================================
# Main Optimizer Pipeline
# ============================================================
class ModelOptimizer:
    """Main optimizer that runs all format comparisons."""
    
    def __init__(self, output_dir: str = "optimized"):
        self.output_dir = output_dir
        self.pytorch_opt = PyTorchOptimizer(output_dir)
        self.onnx_opt = ONNXOptimizer(output_dir)
    
    def optimize(self, model: nn.Module, input_shape: tuple,
                 model_name: str = "model", device: str = "cpu") -> OptimizationReport:
        """Run complete optimization across all formats."""
        report = OptimizationReport(
            model_name=model_name,
            input_shape=list(input_shape),
        )
        
        print(f"\n🔧 Optimizing {model_name}...")
        print(f"   Input: {input_shape} | Device: {device}\n")
        
        # PyTorch
        print("  [1/3] PyTorch baseline...")
        pytorch_result = self.pytorch_opt.optimize(model, input_shape, device)
        report.results.append(pytorch_result)
        print(f"        ✓ {pytorch_result.size_mb}MB | {pytorch_result.latency_p50_ms}ms")
        
        # ONNX FP32 + INT8
        print("  [2/3] ONNX FP32 export...")
        onnx_results = self.onnx_opt.optimize(model, input_shape, device)
        for r in onnx_results:
            report.results.append(r)
            print(f"        ✓ {r.format_name}: {r.size_mb}MB | {r.latency_p50_ms}ms")
        
        # Generate report
        print("  [3/3] Generating report...")
        report_text = ReportGenerator.generate(report)
        print(report_text)
        
        # Save
        ReportGenerator.save_json(
            report, 
            os.path.join(self.output_dir, "optimization_report.json")
        )
        
        return report


# ============================================================
# CLI Entry Point
# ============================================================
def main():
    """Demo: Optimize EfficientNet-B0."""
    from torchvision import models
    
    # Load model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.eval()
    
    # Run optimization
    optimizer = ModelOptimizer(output_dir="optimized")
    report = optimizer.optimize(
        model=model,
        input_shape=(3, 224, 224),
        model_name="EfficientNet-B0",
        device="cpu",
    )
    
    print(f"\n✓ All results saved to: optimized/")
    print(f"  Best latency: {report.best_latency}")
    print(f"  Best size:    {report.best_size}")


if __name__ == "__main__":
    main()
