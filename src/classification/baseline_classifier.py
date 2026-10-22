"""
MobileNetV3-Small Baseline Classifier
Measures latency, peak VRAM, and predictions on a single sample image.
"""

import time
import os
import json
from typing import Dict, Any, List
from PIL import Image

def run_baseline_inference(image_path: str, device_override: str = None) -> Dict[str, Any]:
    import torch
    from torchvision import models, transforms

    # Device selection
    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reset peak memory stats if on CUDA
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    # Load MobileNetV3-Small
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights).to(device)
    model.eval()

    # Preprocessing pipeline
    preprocess = weights.transforms()

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    # Warm-up pass
    with torch.no_grad():
        _ = model(input_tensor)

    # Timed inference pass
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        output = model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # Memory metrics
    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)

    # Top-5 predictions
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    categories = weights.meta["categories"]
    top5_results: List[Dict[str, Any]] = []
    for i in range(5):
        cat_id = top5_catid[i].item()
        top5_results.append({
            "rank": i + 1,
            "category_id": cat_id,
            "category_name": categories[cat_id],
            "probability": round(top5_prob[i].item(), 4)
        })

    return {
        "model_name": "mobilenet_v3_small",
        "device": str(device),
        "image_path": image_path,
        "latency_ms": round(latency_ms, 2),
        "peak_vram_mb": peak_vram_mb,
        "top5_predictions": top5_results
    }

if __name__ == "__main__":
    import sys
    test_img = sys.argv[1] if len(sys.argv) > 1 else None
    if test_img and os.path.exists(test_img):
        res = run_baseline_inference(test_img)
        print(json.dumps(res, indent=2))
    else:
        print("Usage: python baseline_classifier.py <path_to_image>")
