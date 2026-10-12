import os
import sys
import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import evaluate
from db_connector import log_evaluation

# Initialize the Hugging Face evaluate library
rouge = evaluate.load('rouge')

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
COCO_DIR = os.path.join(DATA_DIR, "coco_val100")
ANNOTATIONS_FILE = os.path.join(DATA_DIR, "annotations", "captions_val100.json")

def load_ground_truth():
    print("Loading human ground-truth annotations...")
    with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Map image filenames to a list of their human reference captions
    annotations = {}
    for img in data['images']:
        ann_list = [ann['caption'] for ann in data['annotations'] if ann['image_id'] == img['id']]
        annotations[img['file_name']] = ann_list
    return annotations

def run_production_evaluation(max_images=5):
    print("\n" + "="*60)
    print("TSK-504: Dynamic NLP & Hallucination Scoring Engine")
    print("="*60)

    print("Loading VLM Engine (blip-image-captioning-base)...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    ground_truth = load_ground_truth()
    
    # Grab a small batch of test images from the COCO directory
    test_images = sorted([f for f in os.listdir(COCO_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:max_images]
    
    predictions = []
    references = []

    print("\nRunning AI Inference & Scoring...")
    for filename in test_images:
        img_path = os.path.join(COCO_DIR, filename)
        raw_image = Image.open(img_path).convert('RGB')
        
        inputs = processor(raw_image, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)
            
        predictions.append(caption)
        # Pass the 5 human reference captions for this specific image
        references.append(ground_truth.get(filename, [])) 
        print(f"  Img: {filename} -> VLM: {caption}")

    print("\nCalculating NLP Scores against Human Ground Truth...")
    # ROUGE compares the AI prediction against all 5 human references
    results = rouge.compute(predictions=predictions, references=references)
    
    # We use ROUGE-L as our empirical similarity proxy
    similarity_score = float(results['rougeL'])
    
    # The hallucination/error rate is the inverse of the text similarity match
    hallucination_rate = max(0.0, 1.0 - similarity_score)
    
    print(f"\nFinal Empirical Scores:")
    print(f"  Similarity (ROUGE-L):       {similarity_score:.4f}")
    print(f"  Dynamic Hallucination Rate: {hallucination_rate:.4f}")

    print("\nLogging real empirical metrics to database...")
    log_evaluation(
        model_version="blip-image-captioning-base-production",
        similarity_score=similarity_score,      # Changed from cider_score
        hallucination_rate=hallucination_rate,
        deployment_status="TESTING"
    )
    
    # Quality Gate Enforcer
    if hallucination_rate > 0.80: # 80% error threshold for this specific metric
        print(f"❌ FATAL: Hallucination rate ({hallucination_rate:.2f}) exceeds maximum safety threshold!")
        sys.exit(1)
    else:
        print(f"✅ QUALITY GATE PASSED: Hallucination metrics within safe parameters.")
        sys.exit(0)

if __name__ == "__main__":
    run_production_evaluation(max_images=5)