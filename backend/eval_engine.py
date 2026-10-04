import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from db_connector import log_evaluation # NEW: Import your DB script
import sys

def run_baseline_evaluation():
    print("Loading VLM Engine...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    image_path = "test.jpg"
    try:
        raw_image = Image.open(image_path).convert('RGB')
        print(f"Successfully loaded image: '{image_path}'")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Could not find '{image_path}'")
        return

    print("Running AI inference...")
    inputs = processor(raw_image, return_tensors="pt")

    if True:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)

    print("\n" + "="*50)
    print(f"VLM OUTPUT: {caption}")
    print("="*50 + "\n")

    # NEW: Automatically log to the database
    print("Logging results to metric store...")
    # NEW: Simulated Bad Update Metrics
    simulated_hallucination_rate = 0.35 # 35% hallucination rate
    threshold = 0.20 # Maximum allowable rate

    print("Logging results to metric store...")
    log_evaluation(
        model_version="blip-image-captioning-base-DEGRADED",
        cider_score=0.1500, 
        hallucination_rate=simulated_hallucination_rate, 
        deployment_status="BLOCKED"
    )

    # NEW: Quality Gate Blocking Logic
    if simulated_hallucination_rate > threshold:
        print(f"\n❌ FATAL: Hallucination rate ({simulated_hallucination_rate}) exceeds maximum safety threshold ({threshold})!")
        sys.exit(1) # This forces the GitHub Action to fail

if __name__ == "__main__":
    run_baseline_evaluation()