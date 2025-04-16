#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import torch
import numpy as np
from PIL import Image
from subSelectImages import subSelectImages
from segmentImages import segmentImages
from diffusers import StableDiffusionInpaintPipeline
import random  

def load_sd_inpaint_model():
    print(" Loading Stable Diffusion Inpainting model...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    print(" Model loaded successfully.")
    return pipe

def substituteSquirrels():
    os.makedirs("images", exist_ok=True)

    selected_images = subSelectImages()
    mask_files = [img_path.rsplit(".", 1)[0] + "_mask.png" for img_path in selected_images]
    missing_masks = [mask for mask in mask_files if not os.path.exists(mask)]
    
    if missing_masks:
        print(" Some masks are missing! Generating missing masks...")
        segmentImages()
    else:
        print(" Using pre-generated masks.")

    assert all(os.path.exists(mask) for mask in mask_files), " Mask generation failed!"
    pipe = load_sd_inpaint_model()

    squirrel_images = []

    for img_path, mask_path in zip(selected_images, mask_files):
        print(f" Processing {img_path}...")

        image = Image.open(img_path).convert("RGB").resize((512, 512))
        mask = Image.open(mask_path).convert("L").resize((512, 512))

        text_prompts = [
            "A squirrel eating at a bird feeder, photorealistic, detailed fur, natural outdoor background",
            "A chipmunk sitting where the bird was, eating seeds at the feeder, realistic lighting",
            "A small squirrel holding a seed at the feeder, in a natural setting, ultra-detailed",
            "A fluffy squirrel replacing the bird, chewing on seeds, ultra-realistic",
            "A playful squirrel jumping onto the feeder, detailed fur, natural outdoor environment"
        ]

        text_prompt = random.choice(text_prompts)

        inpainted_image = pipe(
            prompt=text_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=50,
        ).images[0]

        base_name = os.path.basename(img_path)
        name_without_ext, ext = os.path.splitext(base_name)

        if name_without_ext.endswith("-NowWithSquirrels"):
            name_without_ext = name_without_ext.replace("-NowWithSquirrels", "")

        save_path = os.path.join("images", f"{name_without_ext}-NowWithSquirrels.jpg")
        inpainted_image.save(save_path)
        squirrel_images.append(save_path)
        print(f" Saved: {save_path}")

    return squirrel_images

#  Example usage
if __name__ == "__main__":
    squirrel_images = substituteSquirrels()
    print(" Birds replaced with squirrels successfully!")
    print(" Final Saved Images:", squirrel_images)
