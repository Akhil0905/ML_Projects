#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from PIL import Image
from subSelectImages import subSelectImages 

def download_clipseg_model():
    print("Loading CLIPSeg model...")
    processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    print("Model loaded successfully.")
    return processor, model

def segmentImages():
    selected_images = subSelectImages()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor, model = download_clipseg_model()
    model.to(device)

    mask_filenames = []
    texts = ["a bird"]  

    for image_path in selected_images:
        print(f"Processing {image_path} for segmentation...")

        image = Image.open(image_path).convert("RGB")
        inputs = processor(
            text=texts, 
            images=[image] * len(texts),  
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits  

        mask_array = (logits[0, :, :].cpu().numpy() > -2) * 255  
        mask_image = Image.fromarray(mask_array.astype(np.uint8)).resize(image.size)

        mask_filename = image_path.rsplit(".", 1)[0] + "_mask.png"
        mask_image.save(mask_filename)
        mask_filenames.append(mask_filename)
        print(f"Saved mask: {mask_filename}")

    return mask_filenames

# Example usage
if __name__ == "__main__":
    mask_files = segmentImages()
    print("Generated mask files:", mask_files)
