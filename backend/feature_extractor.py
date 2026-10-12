import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel
from tqdm import tqdm

class SemanticFeatureExtractor:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        print(f"Loading Vision Backbone ({model_name})...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract_features(self, image_dir, max_samples=100):
        print(f"Extracting semantic embeddings from {image_dir}...")
        image_files = sorted([
            os.path.join(image_dir, f) 
            for f in os.listdir(image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])[:max_samples]

        embeddings = []
        for img_path in tqdm(image_files):
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use the pooled representation (CLS token output) -> shape: (512,)
                    image_embeds = outputs.pooler_output.cpu().numpy().squeeze()
                    embeddings.append(image_embeds)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        return np.array(embeddings, dtype=np.float32)

if __name__ == "__main__":
    extractor = SemanticFeatureExtractor()
    
    # Test extraction on a small subset to verify
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    coco_dir = os.path.join(data_dir, "coco_val100")
    
    features = extractor.extract_features(coco_dir, max_samples=5)
    print(f"Successfully extracted CLIP features! Shape: {features.shape}")