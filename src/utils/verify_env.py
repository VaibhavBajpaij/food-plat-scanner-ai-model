"""
Automated Environment Verification Script
Validates Python runtime, GPU detection, CUDA compatibility, and memory budget.
"""

import sys
import os
import json
import shutil

def run_environment_audit() -> dict:
    audit = {
        "status": "PASS",
        "python_version": sys.version,
        "platform": sys.platform,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "disk_info_d_drive": {},
        "errors": []
    }

    # Check disk space on D: drive
    try:
        total, used, free = shutil.disk_usage("D:\\")
        audit["disk_info_d_drive"] = {
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2)
        }
        if free / (1024**3) < 10.0:
            audit["status"] = "WARNING"
            audit["errors"].append("Less than 10 GB free on D: drive.")
    except Exception as e:
        audit["errors"].append(f"Disk check failed: {str(e)}")

    # Check PyTorch and CUDA
    try:
        import torch
        audit["torch_version"] = torch.__version__
        audit["cuda_available"] = torch.cuda.is_available()
        audit["device_count"] = torch.cuda.device_count()

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_vram_mb = round(props.total_memory / (1024**2), 2)
                audit["devices"].append({
                    "device_id": i,
                    "name": props.name,
                    "total_vram_mb": total_vram_mb,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count
                })
        else:
            audit["status"] = "WARNING"
            audit["errors"].append("CUDA is not available to PyTorch. CPU fallback only.")
    except ImportError:
        audit["status"] = "FAIL"
        audit["errors"].append("PyTorch is not yet installed in this environment.")

    return audit

if __name__ == "__main__":
    result = run_environment_audit()
    print(json.dumps(result, indent=2))
    
    # Save manifest
    output_path = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "environment_audit.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
