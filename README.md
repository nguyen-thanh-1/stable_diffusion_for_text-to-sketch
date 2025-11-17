# Mini-Stable-Diffusion: Lightweight Text-to-Sketch Generator

![Project Status](https://img.shields.io/badge/Status-Research_Prototype-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

## 📖 Introduction

This project implements a compact, lightweight version of the Stable Diffusion architecture, specifically optimized for generating **128x128 sketch/line-art images**. 

Unlike standard Stable Diffusion models which require heavy computational resources, this project demonstrates that a **scaled-down U-Net trained from scratch**, combined with a **fine-tuned VAE**, can achieve effective text-to-image generation for specific domains (sketches) with significantly reduced parameter counts.

## 🏗️ Architecture & Technical Details

The model follows a two-stage training pipeline utilizing a custom `MiniDiffusionPipeline`.

### 1. Text Encoder (Frozen)
* **Model:** `runwayml/stable-diffusion-v1-5` (CLIP ViT-L/14).
* **Status:** Frozen (weights are not updated). Used to convert text prompts into embeddings.

### 2. Variational Autoencoder - VAE (Fine-tuned)
* **Base Model:** `stabilityai/sd-vae-ft-mse`.
* **Method:** The VAE was fine-tuned on the sketch dataset to better handle high-frequency details typical in line art, ensuring clearer reconstruction at 128x128 resolution.

### 3. U-Net (Custom & Trained from Scratch)
* **Architecture:** A custom "Mini" U-Net design.
* **Channel Configuration:** `(128, 256, 512)` - significantly smaller than standard SD (320, 640, 1280).
* **Training:** Trained from scratch (random initialization) to learn the denoising process specifically for the 128x128 latent space.

---

## 📂 Project Structure

```text
.
├── stable_diffusion.py    # Custom Pipeline class (MiniDiffusionPipeline) & Model Config
├── train_vae.py           # Script for Stage 1: Fine-tuning the VAE
├── train_unet.py          # Script for Stage 2: Training the Mini U-Net
├── dataset.py             # Data loading and preprocessing logic
├── inference.py           # Script for generating images from trained weights
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
