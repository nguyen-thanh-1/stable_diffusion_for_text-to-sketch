# Mini-Stable-Diffusion: Lightweight Text-to-Sketch Generator

![Generated Sample](generated_image_v1.png)
![Generated Sample](generated_image_v2.png)

## Introduction

This project implements a lightweight **Text-to-Sketch** model based on the Stable Diffusion architecture, specifically optimized for generating **128x128 sketch/line-art images**.

Unlike standard Stable Diffusion models which require heavy computational resources, this project demonstrates that a **scaled-down U-Net trained from scratch**, combined with a **fine-tuned VAE**, can achieve effective text-to-image generation for specific domains (sketches) with significantly reduced parameter counts.

## Architecture & Technical Details

The model follows a two-stage training pipeline utilizing a custom `MiniDiffusionPipeline` defined in `stable_diffusion.py`.

### 1. Text Encoder (Frozen)
* **Model:** `runwayml/stable-diffusion-v1-5` (CLIP ViT-L/14).
* **Status:** Frozen (weights are not updated). Used to convert text prompts into embeddings.

### 2. Variational Autoencoder - VAE (Fine-tuned)
* **Base Model:** `stabilityai/sd-vae-ft-mse`.
* **Method:** The VAE is fine-tuned on the sketch dataset to better handle high-frequency details typical in line art, ensuring clearer reconstruction at 128x128 resolution.

### 3. U-Net (Custom & Trained from Scratch)
* **Architecture:** A custom "Mini" U-Net design defined in `MiniDiffusionPipeline`.
* **Channel Configuration:** `(128, 256, 512)` — significantly smaller than standard SD (320, 640, 1280) to fit limited VRAM and speed up training.
* **Training:** Trained from scratch (random initialization) to learn the denoising process specifically for the 128x128 latent space.
