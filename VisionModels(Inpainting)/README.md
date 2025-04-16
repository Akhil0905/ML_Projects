# 🖼️ VisionModels(Inpainting)

This project performs semantic image inpainting using Stable Diffusion. It includes segmentation-based masking and regenerating occluded regions of images to create realistic visuals.

---

## 🎯 Objectives

- Generate masked regions using image segmentation
- Use Stable Diffusion inpainting pipeline for restoration
- Apply prompts for guided reconstruction

---

## 📊 Main packages used:
 - diffusers
 - torchvision
 - PIL
 - torch
 - transformers

**Note: This project runs best with GPU due to diffusion model complexity.**

---

## 🧠 Notes
 - Used sub-selectors to identify and mask objects
 - Supports text-guided inpainting with prompts
 - Image preprocessing and mask handling included

---

## 📁 Folder Structure

```bash
VisionModels(Inpainting)/
├── subselectImages.ipynb           # Image Selection logic
├── segmentImages.ipynb             # Generates Bird Masks
├── removeBirds.ipynb               # Stable Diffusion inpainting
├── replaceBirds.ipynb              # Inpainted new bird image
├── substituteSquirrels.ipynb       # Inpainted squirrel images
├── mySampleImages                  # Includes all the related images 
└── README.md                       # This file
