import os
import sys
import numpy as np
from alibi_detect.cd import MMDDrift
from feature_extractor import SemanticFeatureExtractor

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
COCO_DIR = os.path.join(DATA_DIR, "coco_val100")
OOD_DIR = os.path.join(DATA_DIR, "ood_medmnist100")

def run_production_drift_detection(sample_size=50):
    print("=" * 60)
    print("TSK-503: Evaluating Covariate Shift via CLIP Embeddings & MMD")
    print("=" * 60)

    extractor = SemanticFeatureExtractor()

    # 1. Extract all COCO embeddings in one pass (100 images)
    print(f"\n[1/3] Extracting COCO Embeddings (Total N={sample_size * 2})...")
    coco_embeddings = extractor.extract_features(COCO_DIR, max_samples=sample_size * 2)
    if len(coco_embeddings) < sample_size * 2:
        print(f"ERROR: Expected {sample_size * 2} images in data/coco_val100, found {len(coco_embeddings)}")
        sys.exit(1)

    # Split into Baseline Reference (50) and Clean In-Distribution Traffic (50)
    baseline_embeddings = coco_embeddings[:sample_size]
    clean_test_embeddings = coco_embeddings[sample_size:sample_size * 2]

    # 2. Initialize MMD Drift Detector with Gaussian Kernel
    print("\n[2/3] Initializing Alibi-Detect MMD Detector (backend=pytorch, p_val=0.05)...")
    cd = MMDDrift(baseline_embeddings, backend="pytorch", p_val=0.05)

    # 3. Test In-Distribution Traffic (Clean COCO Images)
    print("Evaluating In-Distribution Traffic (Clean COCO)...")
    preds_clean = cd.predict(clean_test_embeddings)
    is_drift_clean = bool(preds_clean["data"]["is_drift"])
    p_val_clean = float(preds_clean["data"]["p_val"])
    print(f"Clean Traffic Result -> Drift Detected: {is_drift_clean} | P-value: {p_val_clean:.4f}")

    # 4. Test Out-of-Distribution Traffic (MedMNIST X-rays)
    print(f"\n[3/3] Testing Out-of-Distribution Traffic (MedMNIST X-rays N={sample_size})...")
    ood_embeddings = extractor.extract_features(OOD_DIR, max_samples=sample_size)
    
    preds_ood = cd.predict(ood_embeddings)
    is_drift_ood = bool(preds_ood["data"]["is_drift"])
    p_val_ood = float(preds_ood["data"]["p_val"])
    print(f"OOD Traffic Result   -> Drift Detected: {is_drift_ood} | P-value: {p_val_ood:.4f}")

    print("\n" + "=" * 60)
    print("Summary of Statistical Test:")
    print(f"  Clean COCO Traffic:  Drift = {is_drift_clean} (p = {p_val_clean:.4f})")
    print(f"  OOD MedMNIST Shift: Drift = {is_drift_ood} (p = {p_val_ood:.4f})")
    print("=" * 60)

    return is_drift_clean, is_drift_ood

if __name__ == "__main__":
    is_drift_clean, is_drift_ood = run_production_drift_detection(sample_size=50)

    # Quality Gate criteria: Clean traffic passes (no drift), OOD traffic is flagged (drift)
    if not is_drift_clean and is_drift_ood:
        print("\n✅ DRIFT QUALITY GATE VERIFIED: Clean data accepted, OOD domain shift successfully intercepted!")
        sys.exit(0)
    else:
        print("\n🚨 QUALITY GATE FAILED: Drift detector failed validation criteria.")
        sys.exit(1)