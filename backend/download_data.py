import os
import json
import urllib.request
import zipfile
import numpy as np
from PIL import Image
import medmnist
from medmnist import PneumoniaMNIST

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
COCO_IMG_DIR = os.path.join(DATA_DIR, "coco_val100")
OOD_IMG_DIR = os.path.join(DATA_DIR, "ood_medmnist100")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")

os.makedirs(COCO_IMG_DIR, exist_ok=True)
os.makedirs(OOD_IMG_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_ANN_ZIP = os.path.join(DATA_DIR, "annotations_trainval2017.zip")
FILTERED_ANN_PATH = os.path.join(ANNOTATIONS_DIR, "captions_val100.json")

def download_coco_slice(sample_count=100):
    print(f"--- 1. Ingesting COCO 2017 Validation Slice (N={sample_count}) ---")
    
    if not os.path.exists(FILTERED_ANN_PATH):
        if not os.path.exists(COCO_ANN_ZIP):
            print("Downloading COCO annotations archive (~241 MB)...")
            urllib.request.urlretrieve(COCO_ANN_URL, COCO_ANN_ZIP)
            print("Downloaded.")

        print("Extracting captions_val2017.json...")
        with zipfile.ZipFile(COCO_ANN_ZIP, 'r') as zip_ref:
            zip_ref.extract("annotations/captions_val2017.json", DATA_DIR)

        full_ann_path = os.path.join(DATA_DIR, "annotations", "captions_val2017.json")
        with open(full_ann_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)

        # Select the first sample_count unique images
        selected_images = full_data["images"][:sample_count]
        selected_image_ids = {img["id"] for img in selected_images}
        selected_annotations = [
            ann for ann in full_data["annotations"] if ann["image_id"] in selected_image_ids
        ]

        filtered_manifest = {
            "images": selected_images,
            "annotations": selected_annotations
        }

        with open(FILTERED_ANN_PATH, 'w', encoding='utf-8') as f:
            json.dump(filtered_manifest, f, indent=2)

        # Clean up temporary large files to preserve disk space
        if os.path.exists(COCO_ANN_ZIP):
            os.remove(COCO_ANN_ZIP)
        if os.path.exists(full_ann_path):
            os.remove(full_ann_path)
        print(f"Created filtered annotation catalog at {FILTERED_ANN_PATH}")
    else:
        with open(FILTERED_ANN_PATH, 'r', encoding='utf-8') as f:
            filtered_manifest = json.load(f)

    # Download the 100 individual images directly from COCO AWS bucket
    print(f"Verifying/downloading {sample_count} individual COCO images...")
    for idx, img_entry in enumerate(filtered_manifest["images"], start=1):
        target_file = os.path.join(COCO_IMG_DIR, img_entry["file_name"])
        if not os.path.exists(target_file):
            print(f"[{idx}/{sample_count}] Downloading {img_entry['file_name']}...")
            urllib.request.urlretrieve(img_entry["coco_url"], target_file)
    print("COCO image slice ingestion complete.")

def download_medmnist_slice(sample_count=100):
    print(f"\n--- 2. Ingesting MedMNIST OOD Shift Slice (N={sample_count}) ---")
    dataset = PneumoniaMNIST(split='test', download=True, root=DATA_DIR)
    
    saved_count = 0
    for idx, (img_arr, label) in enumerate(dataset):
        if saved_count >= sample_count:
            break
        img_out = os.path.join(OOD_IMG_DIR, f"ood_med_{idx:03d}.png")
        if not os.path.exists(img_out):
            # FIX: img_arr is already a PIL Image, so we just convert it to RGB
            img = img_arr.convert("RGB")
            img.save(img_out)
        saved_count += 1
    print(f"Extracted {saved_count} real medical X-ray images into {OOD_IMG_DIR}")
    
if __name__ == "__main__":
    download_coco_slice(100)
    download_medmnist_slice(100)
    print("\n" + "="*50)
    print("TSK-501 Complete: Curated Benchmark Datasets Ready!")
    print("="*50)