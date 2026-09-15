import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def run_baseline_evaluation():
    # 1. Load the Processor and Model from HuggingFace
    print("Loading VLM Engine (this may take a minute or two on the first run to download weights)...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # 2. Load the Local Test Image
    image_path = "test.jpg"
    try:
        raw_image = Image.open(image_path).convert('RGB')
        print(f"Successfully loaded image: '{image_path}'")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Could not find '{image_path}'. Make sure it is saved in the backend/ folder.")
        return

    # 3. Process the Image (Convert pixels to tensors for the AI)
    print("Running AI inference...")
    inputs = processor(raw_image, return_tensors="pt")

    # 4. Generate the Caption
    # We use torch.no_grad() because we are only evaluating, not training the model.
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50)
        
    caption = processor.decode(out[0], skip_special_tokens=True)

    # 5. Print the Final Result
    print("\n" + "="*50)
    print(f"VLM OUTPUT: {caption}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_baseline_evaluation()