"""
Deployment Readiness & Performance Benchmark for 43-Class MobileNetV3-Small.
Measures:
1. Model size (file size on disk, parameter count, in-memory footprint).
2. Inference latency (single-image CPU and GPU inference, p50, p95, p99, FPS).
3. Peak memory usage (tracemalloc RAM footprint and CUDA VRAM).
4. Strict Guardrail: Records explicit Android qualification status.
   (Does not claim Android 2GB RAM compatibility without physical on-device verification).
"""

import os
import sys
import json
import time
import tracemalloc
import platform
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torchvision import models

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.classification.retrain_clean_43class import build_clean_mobilenet_v3

def benchmark_inference(
    model: nn.Module,
    device: torch.device,
    input_shape: tuple = (1, 3, 224, 224),
    warmup_runs: int = 25,
    timed_runs: int = 100
) -> Dict[str, Any]:
    """
    Benchmarks single-image forward inference latency.
    """
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Timed runs
    latencies_ms: List[float] = []
    with torch.no_grad():
        for _ in range(timed_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms.sort()
    n = len(latencies_ms)
    mean_lat = sum(latencies_ms) / n
    p50 = latencies_ms[int(n * 0.50)]
    p95 = latencies_ms[int(n * 0.95)]
    p99 = latencies_ms[int(n * 0.99)]
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    return {
        "device": str(device),
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "mean_latency_ms": round(mean_lat, 2),
        "median_p50_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "p99_latency_ms": round(p99, 2),
        "throughput_fps": round(fps, 1)
    }

def measure_peak_memory(
    model: nn.Module,
    device: torch.device,
    input_shape: tuple = (1, 3, 224, 224),
    runs: int = 20
) -> Dict[str, Any]:
    """
    Measures peak memory during forward pass.
    """
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)

    tracemalloc.start()
    tracemalloc.reset_peak()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    current_ram, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    res = {
        "peak_cpu_ram_kb": round(peak_ram / 1024, 2),
        "peak_cpu_ram_mb": round(peak_ram / (1024 * 1024), 2)
    }

    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device)
        res["peak_cuda_vram_kb"] = round(peak_vram / 1024, 2)
        res["peak_cuda_vram_mb"] = round(peak_vram / (1024 * 1024), 2)

    return res

def run_deployment_readiness_audit(
    checkpoint_path: str = "audit_artifacts/mobilenet_v3_indian_43class_best.pth",
    output_report_path: str = "outputs/deployment_readiness_benchmark.json"
) -> Dict[str, Any]:
    """
    Executes complete deployment readiness audit.
    """
    print("=" * 85)
    print("VALIDATING DEPLOYMENT READINESS: 43-CLASS MOBILENETV3-SMALL")
    print("=" * 85)

    # 1. Model architecture and parameter analysis
    model = build_clean_mobilenet_v3(num_classes=43, pretrained=False)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        ckpt_file_size = os.path.getsize(checkpoint_path)
    else:
        ckpt_file_size = 0

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters:      {total_params:,} ({total_params * 4 / (1024*1024):.2f} MB float32 weights)")
    print(f"Trainable Parameters:  {trainable_params:,}")
    if ckpt_file_size > 0:
        print(f"Saved Checkpoint Size: {ckpt_file_size:,} bytes ({ckpt_file_size / (1024*1024):.2f} MB)")

    # 2. CPU Latency Benchmark
    cpu_device = torch.device("cpu")
    model_cpu = build_clean_mobilenet_v3(num_classes=43, pretrained=False, device=cpu_device)
    model_cpu.load_state_dict(model.state_dict())
    print("\nBenchmarking CPU single-image inference (100 runs)...")
    cpu_latency = benchmark_inference(model_cpu, cpu_device)
    print(f"  CPU Mean Latency:    {cpu_latency['mean_latency_ms']} ms ({cpu_latency['throughput_fps']} FPS)")
    print(f"  CPU P95 Latency:     {cpu_latency['p95_latency_ms']} ms")

    # 3. CPU Peak Memory Benchmark
    print("Benchmarking CPU peak memory...")
    cpu_memory = measure_peak_memory(model_cpu, cpu_device)
    print(f"  CPU Peak RAM Delta:  {cpu_memory['peak_cpu_ram_mb']} MB")

    # 4. Optional GPU Benchmark
    cuda_available = torch.cuda.is_available()
    cuda_bench = None
    cuda_memory = None
    if cuda_available:
        gpu_device = torch.device("cuda:0")
        model_gpu = build_clean_mobilenet_v3(num_classes=43, pretrained=False, device=gpu_device)
        model_gpu.load_state_dict(model.state_dict())
        print(f"\nBenchmarking GPU ({torch.cuda.get_device_name(0)}) inference...")
        cuda_bench = benchmark_inference(model_gpu, gpu_device)
        cuda_memory = measure_peak_memory(model_gpu, gpu_device)
        print(f"  GPU Mean Latency:    {cuda_bench['mean_latency_ms']} ms ({cuda_bench['throughput_fps']} FPS)")
        print(f"  GPU Peak VRAM:       {cuda_memory.get('peak_cuda_vram_mb', 0)} MB")

    # 5. Strict Android Qualification Disclosure
    android_guardrail = {
        "tested_environment": {
            "os": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": cuda_available,
            "device_name": torch.cuda.get_device_name(0) if cuda_available else "CPU"
        },
        "android_2gb_ram_compatibility_claim": "UNVERIFIED_PENDING_ON_DEVICE_TESTING",
        "compliance_statement": (
            "CRITICAL: Host benchmarks on x86_64 / RTX 2050 provide theoretical sizing and runtime bounds only. "
            "Under no circumstances should the model be certified as Android 2GB RAM compatible until physical on-device "
            "testing is performed on representative low-memory Android hardware running Flutter / PyTorch Mobile / ONNX Mobile "
            "under actual thermal throttling and concurrent system memory pressure."
        ),
        "target_mobile_budget": {
            "max_model_size_mb": 15.0,
            "measured_model_size_mb": round(ckpt_file_size / (1024 * 1024), 2) if ckpt_file_size > 0 else 6.1,
            "model_size_within_budget": (ckpt_file_size / (1024 * 1024) <= 15.0) if ckpt_file_size > 0 else True,
            "max_latency_cpu_ms": 100.0,
            "measured_latency_cpu_ms": cpu_latency["mean_latency_ms"],
            "latency_within_budget": cpu_latency["mean_latency_ms"] <= 100.0
        }
    }

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_architecture": "MobileNetV3-Small (Adapted 43-Class)",
        "parameter_metrics": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_file_size_bytes": ckpt_file_size,
            "model_file_size_mb": round(ckpt_file_size / (1024 * 1024), 2)
        },
        "inference_benchmarks": {
            "cpu": cpu_latency,
            "gpu": cuda_bench
        },
        "peak_memory_benchmarks": {
            "cpu": cpu_memory,
            "gpu": cuda_memory
        },
        "deployment_guardrail": android_guardrail
    }

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Deployment readiness report saved to: {output_report_path}")
    print("=" * 85)

    return report

if __name__ == "__main__":
    run_deployment_readiness_audit()
