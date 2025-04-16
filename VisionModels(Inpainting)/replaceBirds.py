#!/usr/bin/env python
# coding: utf-8

# In[31]:


import os
import torch
import numpy as np
from PIL import Image
from subSelectImages import subSelectImages
from segmentImages import segmentImages
from diffusers import StableDiffusionInpaintPipeline

def load_sd_inpaint_model():
    print(" Loading Stable Diffusion Inpainting model...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    print(" Model loaded successfully.")
    return pipe

def analyze_mask(mask):
    mask_np = np.array(mask)
    white_pixels = np.sum(mask_np > 128)
    total_pixels = mask_np.size
    mask_ratio = white_pixels / total_pixels

    rows = np.any(mask_np > 128, axis=1)
    cols = np.any(mask_np > 128, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]] if np.any(rows) else (0, 0)
    x_min, x_max = np.where(cols)[0][[0, -1]] if np.any(cols) else (0, 0)
    mask_height = y_max - y_min
    mask_width = x_max - x_min

    center_x = mask_np.shape[1] // 2
    center_y = mask_np.shape[0] // 2
    is_near_center = (x_min < center_x < x_max) and (y_min < center_y < y_max)

    if mask_ratio < 0.02:
        return "A small bird sitting on a feeder perch, looking natural"
    elif mask_ratio > 0.6:
        return "A seamless natural background with a bird feeder"
    elif mask_height > mask_width * 1.5:
        return "A bird in mid-flight with wings spread, soaring near a bird feeder"
    elif is_near_center and mask_ratio < 0.1:
        return (
            "A small bird perched inside a transparent feeder, visible through the glass, "
            "with soft natural reflections, realistic lighting"
        )
    else:
        return "A realistic bird perched on a feeder, blending with the outdoor scene"

def replaceBirdsWithText():
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
    replaced_images = []

    for img_path, mask_path in zip(selected_images, mask_files):
        print(f" Processing {img_path}...")

        image = Image.open(img_path).convert("RGB").resize((512, 512))
        mask = Image.open(mask_path).convert("L").resize((512, 512))
        text_prompt = analyze_mask(mask)

        if "transparent feeder" in text_prompt:
            text_prompt = (
                "A bird perched inside a transparent feeder, clearly visible, "
                "realistic lighting, soft glass reflections, bird eating seeds."
            )

        inpainted_image = pipe(
            prompt=[text_prompt, "A realistic outdoor background with seamless blending"],
            image=image,
            mask_image=mask,
            num_inference_steps=50,
            guidance_scale=7.5,
        ).images[0]

        base_name = os.path.basename(img_path)
        name_without_ext, ext = os.path.splitext(base_name)
        save_path = os.path.join("images", f"{name_without_ext}-birdsReplaced{ext}")
        inpainted_image.save(save_path)
        replaced_images.append(save_path)
        print(f" Saved: {save_path}")

    return replaced_images

#  Example usage
if __name__ == "__main__":
    replaced_bird_images = replaceBirdsWithText()
    print(" Birds replaced successfully!")
    print(" Final Saved Images:", replaced_bird_images)
