"""Evaluate zero-shot prediction baselines (Chance vs Train Majority)."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

from src.metrics.geolocation import GeolocationDistanceError, GeolocationMissRate

# ---------------------------------------------------------------------------
# Imports & Task Specs
# ---------------------------------------------------------------------------

def _import_torchmetrics():
    from torchmetrics.classification import (
        MulticlassAccuracy,
        MulticlassF1Score,
    )
    from torchmetrics.regression import (
        MeanAbsoluteError, 
        PearsonCorrCoef,
        KendallRankCorrCoef
    )
    return {
        "MulticlassAccuracy": MulticlassAccuracy,
        "MulticlassF1Score": MulticlassF1Score,
        "MeanAbsoluteError": MeanAbsoluteError,
        "PearsonCorrCoef": PearsonCorrCoef,
        "KendallRankCorrCoef": KendallRankCorrCoef,
    }

class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GEOLOCATION = "geolocation"

@dataclass
class TaskSpec:
    task_type: TaskType
    num_classes: int
    min_value: int = 0
    max_value: int = 0
    pred_key: str = "processed_transcript"

TASK_SPECS: Dict[str, TaskSpec] = {
    "cmul2arcticl1": TaskSpec(TaskType.CLASSIFICATION, num_classes=7),
    "edacc": TaskSpec(TaskType.CLASSIFICATION, num_classes=13),
    "fleurs": TaskSpec(TaskType.CLASSIFICATION, num_classes=24),
    "ultrasuite_child": TaskSpec(TaskType.CLASSIFICATION, num_classes=2),
    "speechocean": TaskSpec(TaskType.REGRESSION, num_classes=11, min_value=0, max_value=10),
    "easycall": TaskSpec(TaskType.REGRESSION, num_classes=4, min_value=0, max_value=3),
    "uaspeech": TaskSpec(TaskType.REGRESSION, num_classes=5, min_value=0, max_value=4),
    "vaanigeo": TaskSpec(TaskType.GEOLOCATION, num_classes=2, pred_key="predicted_transcript"),
}

# ---------------------------------------------------------------------------
# Loading & Utilities
# ---------------------------------------------------------------------------

def normalize_record(record: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(record, dict): return None
    if "pred" in record or "passthrough" in record: return record
    if len(record) == 1:
        ((idx, payload),) = record.items()
        if isinstance(payload, dict):
            out = {"idx": idx}; out.update(payload); return out
    return record

def load_records(pattern: str) -> List[Dict[str, Any]]:
    records = []
    files = sorted(glob(pattern))
    for filepath in files:
        if ".cache" in filepath or ".error" in filepath: continue
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = normalize_record(json.loads(line.strip()))
                    if r: records.append(r)
                except: continue
    return records

def latlon_to_xyz(lat_rad, lon_rad):
    x = math.cos(lat_rad) * math.cos(lon_rad)
    y = math.cos(lat_rad) * math.sin(lon_rad)
    z = math.sin(lat_rad)
    return (x, y, z)

# ---------------------------------------------------------------------------
# 1. Statistics Calculation (The "Train Majority/Mean" Logic)
# ---------------------------------------------------------------------------

def compute_train_stats(records: List[Dict[str, Any]], spec: TaskSpec) -> Any:
    """Computes the Mode (Cls) or Mean (Reg/Geo) from the TRAIN split."""
    train_targets = []
    
    for rec in records:
        pt = rec.get("passthrough", {})
        if pt.get("split") != "train":
            continue
        
        target = pt.get("target")
        if target is None: continue
        
        # Geolocation target is [lat_rad, lon_rad]
        # Others are int or float
        train_targets.append(target)

    if not train_targets:
        print("Warning: No train records found. Falling back to defaults.")
        if spec.task_type == TaskType.GEOLOCATION: return (0.0, 0.0) # 0,0 rads
        return 0

    if spec.task_type == TaskType.CLASSIFICATION:
        # Return Mode
        return Counter(train_targets).most_common(1)[0][0]
    
    elif spec.task_type == TaskType.REGRESSION:
        # Return Mean
        return sum(train_targets) / len(train_targets)
    
    elif spec.task_type == TaskType.GEOLOCATION:
        # Return Mean Lat/Lon in Radians, 
        lats = [t[0] for t in train_targets]
        lons = [t[1] for t in train_targets]
        return (sum(lats)/len(lats), sum(lons)/len(lons))

# ---------------------------------------------------------------------------
# 2. Prediction & Metric Calculation
# ---------------------------------------------------------------------------

def evaluate_run(records: List[Dict[str, Any]], spec: TaskSpec, baseline_type: str, train_stat: Any):
    """Generates predictions for TEST split based on baseline_type and evaluates."""
    
    # Storage for valid pairs
    preds_cls, targets_cls = [], []
    raw_preds_reg, int_preds_reg, targets_reg = [], [], []
    preds_geo, targets_geo = [], []

    for rec in records:
        pt = rec.get("passthrough", {})
        # Only evaluate on TEST split
        if pt.get("split", "test") != "test": 
            continue

        target = pt.get("target")
        if target is None: continue

        # --- GENERATE PREDICTION ---
        pred_val = None
        
        if spec.task_type == TaskType.CLASSIFICATION:
            if baseline_type == "chance":
                pred_val = random.randint(0, spec.num_classes - 1)
            else: # majority
                pred_val = int(train_stat)
            
            preds_cls.append(pred_val)
            targets_cls.append(int(target))

        elif spec.task_type == TaskType.REGRESSION:
            if baseline_type == "chance":
                pred_val = random.uniform(spec.min_value, spec.max_value)
            else: # majority (mean)
                pred_val = float(train_stat)
            
            # Clamp and round for int metrics
            int_val = int(round(pred_val))
            int_val = max(spec.min_value, min(spec.max_value, int_val))
            
            raw_preds_reg.append(pred_val)
            int_preds_reg.append(int_val)
            targets_reg.append(int(target))

        elif spec.task_type == TaskType.GEOLOCATION:
            # Target is [lat_rad, lon_rad]
            if baseline_type == "chance":
                # Random lat/lon in degrees, convert to XYZ
                r_lat = math.radians(random.uniform(-90.0, 90.0))
                r_lon = math.radians(random.uniform(-180.0, 180.0))
                pred_val = latlon_to_xyz(r_lat, r_lon)
            else: # majority (mean)
                # train_stat is (mean_lat_rad, mean_lon_rad)
                pred_val = latlon_to_xyz(train_stat[0], train_stat[1])

            preds_geo.append(pred_val)
            targets_geo.append((float(target[0]), float(target[1])))

    # --- COMPUTE METRICS ---
    tm = _import_torchmetrics()
    
    if spec.task_type == TaskType.CLASSIFICATION:
        if not preds_cls: return {}
        preds_t, targets_t = torch.tensor(preds_cls), torch.tensor(targets_cls)
        return {
            "acc": tm["MulticlassAccuracy"](num_classes=spec.num_classes)(preds_t, targets_t).item(),
            "f1": tm["MulticlassF1Score"](num_classes=spec.num_classes, average="macro")(preds_t, targets_t).item(),
        }

    elif spec.task_type == TaskType.REGRESSION:
        if not targets_reg: return {}
        int_p, raw_p, t = torch.tensor(int_preds_reg), torch.tensor(raw_preds_reg), torch.tensor(targets_reg)
        return {
            "mae": tm["MeanAbsoluteError"]()(int_p.float(), t.float()).item(),
            "pcc": tm["PearsonCorrCoef"]()(raw_p, t.float()).item(),
            "kendall": tm["KendallRankCorrCoef"]()(raw_p, t.float()).item(),
        }

    elif spec.task_type == TaskType.GEOLOCATION:
        if not targets_geo: return {}
        p, t = torch.tensor(preds_geo), torch.tensor(targets_geo)
        err = GeolocationDistanceError()
        miss1 = GeolocationMissRate(k=1)
        err.update(p, t)
        miss1.update(p, t)
        return {
            "err_km": err.compute().item(),
            "hit_rate_top1": 100-miss1.compute().item(),
        }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(TASK_SPECS.keys()))
    parser.add_argument("--run_dir", type=str, help="Dir with predictions")
    parser.add_argument("--predictions", type=str, help="Glob for predictions")
    parser.add_argument("--baseline", choices=["chance", "majority"], default="chance", 
                        help="'chance': Random uniform. 'majority': Train Mean/Mode.")
    parser.add_argument("--seed", type=int, default=24)
    args = parser.parse_args()

    # Setup
    pattern = args.predictions if args.predictions else str(Path(args.run_dir) / "prediction*.jsonl")
    spec = TASK_SPECS[args.dataset]
    
    print(f"Loading records for {args.dataset}...")
    records = load_records(pattern)
    print(f"Total records: {len(records)}")

    # 1. Compute Train Stats (once, deterministic)
    train_stat = compute_train_stats(records, spec)
    if args.baseline == "majority":
        print(f"Baseline: Train Majority/Mean -> {train_stat}")
    else:
        print(f"Baseline: Random Chance")

    # 2. Run Evaluation Loop (5 seeds)
    # If baseline is deterministic (majority), we only need 1 run, but keeping logic uniform is fine.
    seeds = range(args.seed, args.seed + 5) if args.baseline == "chance" else [args.seed]
    aggregated_results = {}

    print(f"\nRunning {len(seeds)} seed: ({list(seeds)})...")
    
    for s in seeds:
        random.seed(s)
        metrics = evaluate_run(records, spec, args.baseline, train_stat)
        
        for k, v in metrics.items():
            aggregated_results.setdefault(k, []).append(v)

    # 3. Report Results
    print("\n" + "=" * 50)
    print(f"RESULTS: {args.dataset.upper()} | Baseline: {args.baseline.upper()}")
    print("=" * 50)
    
    for k, values in aggregated_results.items():
        mean_v = statistics.mean(values)
        std_v = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{k:<15}: {mean_v:.4f} ± {std_v:.4f}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()