
---

### ✅ `VisionModels(Inpainting)`

```markdown
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
├── segmentImages.py         # Segmentation logic
├── inpaint_pipeline.py      # Stable Diffusion inpainting
├── sample_images/           # Input test cases
├── outputs/                 # Inpainted image results
└── README.md                # This file
