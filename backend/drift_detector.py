import numpy as np
from PIL import Image
from alibi_detect.cd import MMDDrift

def preprocess_image(image_path):
    # Open image, resize to a small resolution to keep math fast, convert to RGB
    img = Image.open(image_path).convert('RGB').resize((64, 64))
    # Convert to numpy array, normalize pixel values between 0 and 1, and flatten
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array.flatten()

def run_drift_detection():
    print("Initializing MMD Drift Detector...")
    
    # 1. Load the reference image
    try:
        base_img = preprocess_image("test.jpg")
    except FileNotFoundError:
        print("CRITICAL ERROR: 'test.jpg' not found. Please ensure it is in the backend/ folder.")
        return

    # 2. Simulate a "Reference Dataset" (e.g., your COCO training data)
    print("Generating reference distribution (Baseline)...")
    np.random.seed(42)
    reference_data = []
    for _ in range(20):
        noise = np.random.normal(0, 0.05, base_img.shape)
        reference_data.append(base_img + noise)
    reference_data = np.array(reference_data, dtype=np.float32)

    # 3. Initialize the MMD Drift Detector
    cd = MMDDrift(reference_data, backend='pytorch', p_val=0.05)

    # 4. Test Case A: Normal Traffic (No Drift)
    print("\n--- Test Case A: Normal Traffic (Standard Image) ---")
    test_normal = np.array([base_img, base_img]) 
    preds_normal = cd.predict(test_normal)
    is_drift_normal = bool(preds_normal['data']['is_drift'])
    # FIX: Removed the [0] from p_val
    print(f"Result -> is_drift: {is_drift_normal} (P-value: {preds_normal['data']['p_val']:.4f})")

    # 5. Test Case B: Bad Traffic (Drift)
    print("\n--- Test Case B: Anomalous Traffic (Pure Static Noise) ---")
    test_anomaly = np.random.uniform(0, 1, size=(2, base_img.shape[0])).astype(np.float32)
    preds_anomaly = cd.predict(test_anomaly)
    is_drift_anomaly = bool(preds_anomaly['data']['is_drift'])
    # FIX: Removed the [0] from p_val
    print(f"Result -> is_drift: {is_drift_anomaly} (P-value: {preds_anomaly['data']['p_val']:.4f})")
    
    # 6. Final Output
    print("\n" + "="*55)
    print("TSK-203 Drift Evaluation Complete!")
    print(f"Successfully returned boolean - Anomaly Blocked: {is_drift_anomaly}")
    print("="*55 + "\n")

if __name__ == "__main__":
    run_drift_detection()