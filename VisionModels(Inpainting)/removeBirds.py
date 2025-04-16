#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
from subSelectImages import subSelectImages
from segmentImages import segmentImages

def removeBirds():
    selected_images = subSelectImages()
    print(f"Selected images: {selected_images}")

    mask_files = segmentImages()
    print(f"Generated masks: {mask_files}")

    assert len(selected_images) == len(mask_files), "Mismatch between selected images and generated masks!"

    inpainted_images = []

    for img_path, mask_path in zip(selected_images, mask_files):
        print(f"Processing {img_path}...")

        # Load images and masks
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask > 128, 255, 0).astype(np.uint8)

        inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        # Save the inpainted image
        save_path = img_path.replace(".jpg", "-birdsRemoved.jpg").replace(".jpeg", "-birdsRemoved.jpg")
        cv2.imwrite(save_path, inpainted_image)
        inpainted_images.append(save_path)
        print(f"Saved: {save_path}")

    return inpainted_images

# Example usage
if __name__ == "__main__":
    removed_bird_images = removeBirds()
    print("Birds removed successfully! Files saved:", removed_bird_images)
