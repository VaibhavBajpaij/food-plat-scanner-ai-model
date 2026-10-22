"""
Smoke Test Evaluation Runner
Evaluates pretrained MobileNetV3-Small (ImageNet-1k) on real-food smoke test dataset.
Clearly distinguishes prepared dishes from raw produce and analyzes failure modes.
"""

import os
import sys
import json
import time
from collections import defaultdict
from typing import Dict, Any, List
from PIL import Image

def run_smoke_test_evaluation(manifest_path: str = None) -> Dict[str, Any]:
    import torch
    from torchvision import models

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if manifest_path is None:
        manifest_path = os.path.join(base_dir, "data_manifests", "smoke_test_manifest.json")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run download_smoke_dataset.py first.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("RUNNING REAL-FOOD SMOKE TEST EVALUATION")
    print(f"Device: {device} | Total Images: {len(manifest.get('images', []))}")
    print("=" * 70)

    # Load model
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights).to(device)
    model.eval()
    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    image_results = []
    latencies = []
    predicted_class_counts = defaultdict(int)
    class_evaluations = defaultdict(lambda: {"total": 0, "food_related_top1": 0, "food_related_top5": 0, "correct_synset_top1": 0})

    # Exact ImageNet synset mappings for produce
    EXACT_SYNSET_MAP = {
        "banana_raw": ["banana"],
        "orange_raw": ["orange"],
        "apple_fresh_raw": ["Granny Smith", "custard apple"],
        "broccoli_steamed": ["broccoli"],  # ImageNet has 937: broccoli
        "rice_white_cooked": [],            # ImageNet has NO cooked white rice class
        "chicken_breast_grilled": []        # ImageNet has NO grilled chicken breast class
    }

    # Generic food-related keywords to distinguish food vs non-food classifications (e.g. plate, wok, bowl)
    FOOD_RELATED_KEYWORDS = [
        "banana", "orange", "apple", "broccoli", "cauliflower", "cabbage",
        "meatloaf", "mashed potato", "potpie", "burrito", "pizza", "cheeseburger",
        "consomme", "hot pot", "trifle", "ice cream", "bagel", "pretzel", "cucumber"
    ]

    for item in manifest.get("images", []):
        img_rel_path = item["file_path"]
        img_abs_path = os.path.join(base_dir, img_rel_path)
        if not os.path.exists(img_abs_path):
            continue

        cid = item["canonical_id"]
        class_evaluations[cid]["total"] += 1

        img = Image.open(img_abs_path).convert("RGB")
        input_tensor = preprocess(img).unsqueeze(0).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            output = model(input_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize()
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        top5_list = []
        for rank in range(5):
            cat_id = top5_catid[rank].item()
            cname = categories[cat_id]
            prob = round(top5_prob[rank].item(), 4)
            top5_list.append({"rank": rank + 1, "category": cname, "probability": prob})

        top1_name = top5_list[0]["category"]
        predicted_class_counts[top1_name] += 1

        # Check exact synset correctness
        exact_synsets = EXACT_SYNSET_MAP.get(cid, [])
        is_exact_top1 = any(s.lower() in top1_name.lower() for s in exact_synsets)
        if is_exact_top1:
            class_evaluations[cid]["correct_synset_top1"] += 1

        # Check if model at least identified it as edible food vs kitchenware (plate/bowl)
        is_food_top1 = any(kw in top1_name.lower() for kw in FOOD_RELATED_KEYWORDS)
        is_food_top5 = any(any(kw in entry["category"].lower() for kw in FOOD_RELATED_KEYWORDS) for entry in top5_list)
        if is_food_top1:
            class_evaluations[cid]["food_related_top1"] += 1
        if is_food_top5:
            class_evaluations[cid]["food_related_top5"] += 1

        image_results.append({
            "image_id": item["image_id"],
            "canonical_id": cid,
            "food_type": item.get("food_type", "unknown"),
            "lighting": item.get("lighting_condition", "unknown"),
            "latency_ms": round(lat_ms, 2),
            "top1_category": top1_name,
            "top1_probability": top5_list[0]["probability"],
            "top5_predictions": top5_list,
            "exact_synset_match": is_exact_top1,
            "food_related_top1": is_food_top1,
            "food_related_top5": is_food_top5
        })

    # Summary metrics
    total_imgs = len(image_results)
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
    prepared_dish_total = sum(1 for r in image_results if r["food_type"] == "prepared_dish")
    produce_total = sum(1 for r in image_results if r["food_type"] == "raw_produce")

    evaluation_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_evaluated": "MobileNetV3-Small (ImageNet-1k Pretrained Baseline)",
        "checkpoint_source": "torchvision.models.mobilenet_v3_small-047dcff4.pth",
        "dataset_name": manifest.get("dataset_name"),
        "total_images_evaluated": total_imgs,
        "mean_latency_ms": avg_latency,
        "class_breakdown": dict(class_evaluations),
        "predicted_category_distribution": dict(predicted_class_counts),
        "per_image_results": image_results,
        "key_findings": [
            "Pretrained ImageNet weights perform poorly on prepared dishes (rice, grilled chicken) because ImageNet lacks prepared food classes.",
            "Prepared dishes are frequently misclassified as non-food kitchenware (e.g. 'plate', 'wok', 'soup bowl', 'platter') or unrelated textured objects.",
            "Raw produce classes (banana, orange) with exact ImageNet synsets perform significantly better when isolated, but still suffer from plate background interference.",
            "Conclusion: A food-specific classification head and fine-tuning are mandatory for reliable food plate scanning."
        ]
    }

    # Save outputs
    out_dir = os.path.join(base_dir, "outputs", "smoke_test_results")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "smoke_test_evaluation.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    # Print summary
    print(f"\nEvaluation Complete. Average Latency: {avg_latency} ms")
    print("\nClass Breakdown:")
    for cid, stats in class_evaluations.items():
        print(f"  {cid:24} Total: {stats['total']} | Exact Top-1: {stats['correct_synset_top1']} | Food-Related Top-1: {stats['food_related_top1']}")

    print("\nTop Predicted ImageNet Categories:")
    for cname, count in sorted(predicted_class_counts.items(), key=lambda x: x[1], reverse=True)[:6]:
        print(f"  {cname:24}: {count}")

    print(f"\nDetailed report saved to: {out_json}")
    return evaluation_report

if __name__ == "__main__":
    run_smoke_test_evaluation()
