import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from diffusers.optimization import get_scheduler
from diffusers import DDPMScheduler, StableDiffusionPipeline 
from PIL import Image
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from stable_diffusion import MiniDiffusionPipeline
from dataset import SketchDataset


from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


# --- Cấu hình ---
TRAIN_DATA_DIR = r"C:\Users\Admin\Desktop\scientific research\dataset\train" 
VAL_DATA_DIR = r"C:\Users\Admin\Desktop\scientific research\dataset\val"    
VAE_PATH = "./vae-finetuned.safetensors"      
IMAGE_SIZE = 128
EPOCHS = 100          
BATCH_SIZE = 16 * 5
LEARNING_RATE = 1e-4 * 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_UNET_PATH = "./unet-mini.safetensors" 

CHECKPOINT_PATH = "./unet_latest_checkpoint.pth" 
NUM_INFERENCE_STEPS = 50 

TINY_UNET_CONFIG = {
    "unet_block_out_channels": (128, 256, 512),
}

def plot_metrics(history, filename="unet_metrics_plot.png"):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    axs[0, 0].plot(history['train_loss'], label="Train Loss")
    axs[0, 0].plot(history['val_loss'], label="Validation Loss")
    axs[0, 0].set_title("Train vs Validation Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("MSE Loss")
    axs[0, 0].legend()
    
    axs[0, 1].plot(history['fid'], label="FID", color='green')
    axs[0, 1].set_title("Fréchet Inception Distance (FID)")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("FID (lower is better)")
    axs[0, 1].legend()

    axs[1, 0].plot(history['lpips'], label="LPIPS", color='red')
    axs[1, 0].set_title("Learned Perceptual Image Patch Similarity (LPIPS)")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("LPIPS (lower is better)")
    axs[1, 0].legend()

    axs[1, 1].plot(history['clip_score'], label="CLIP Score", color='purple')
    axs[1, 1].set_title("CLIP Score")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("CLIP Score (higher is better)")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Đã lưu biểu đồ metrics tại {filename}")

def evaluate(
    eval_pipeline, gen, val_loader, metrics, 
    unet, vae, text_encoder, scheduler, 
    vae_scale_factor, num_inference_steps
):
   
    unet.eval() 
    total_val_loss = 0.0

    for metric in metrics.values():
        metric.reset()

    def to_uint8(images):
        images = (images.clamp(-1, 1) + 1) / 2 
        images = (images * 255).type(torch.uint8)
        return images
    
    def to_lpips_format(images):
        return images.clamp(-1, 1)

    pbar = tqdm(val_loader, desc="[Validation & Evaluation]")
    for batch in pbar:
        images = batch["pixel_values"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        
        with torch.no_grad():
            # --- TÍNH VALIDATION LOSS ---
            latents = vae.encode(images).latent_dist.mean * vae_scale_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=DEVICE)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            text_embeds = text_encoder(input_ids)[0]
            
            noise_pred = unet(noisy_latents, timesteps, text_embeds).sample
            val_loss = F.mse_loss(noise_pred, noise)
            total_val_loss += val_loss.item()
            
            # --- SINH ẢNH (Dùng eval_pipeline) ---
            prompts = eval_pipeline.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            generated_output = eval_pipeline(
                prompt=prompts,
                num_inference_steps=num_inference_steps,
                output_type="pt",
                generator=gen 
            )
            
            generated_images = generated_output.images 
            generated_images_norm = (generated_images * 2) - 1 
            
            # --- CẬP NHẬT METRICS ---
            gt_images_uint8 = to_uint8(images)
            gt_images_lpips = to_lpips_format(images)
            gen_images_uint8 = to_uint8(generated_images_norm)
            gen_images_lpips = to_lpips_format(generated_images_norm)

            metrics["fid"].update(gt_images_uint8, real=True)
            metrics["fid"].update(gen_images_uint8, real=False)
            metrics["lpips"].update(gt_images_lpips, gen_images_lpips)
            metrics["clip_score"].update(gen_images_uint8, prompts)

    # --- TRẢ VỀ KẾT QUẢ ---
    results = {
        "val_loss": total_val_loss / len(val_loader),
        "fid": metrics["fid"].compute().item(),
        "lpips": metrics["lpips"].compute().item(),
        "clip_score": metrics["clip_score"].compute().item()
    }
    
    return results

def main():
    print("--- Giai đoạn 2: Huấn luyện UNet-mini ---")
    start_time_total = time.time()
    
    # Khởi tạo Pipeline
    pipeline = MiniDiffusionPipeline(
        base_model_id="runwayml/stable-diffusion-v1-5",
        device=DEVICE,
        config_overrides=TINY_UNET_CONFIG 
    )
    
    # Tải VAE đã fine-tune
    try:
        pipeline.vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
        print(f"Tải VAE đã fine-tune thành công từ {VAE_PATH}")
    except Exception as e:
        print(f"Lỗi: Không thể tải VAE từ {VAE_PATH}. {e}")
        print("Vui lòng chạy train_vae.py trước!")
        return
        
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    
    unet = pipeline.unet
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    noise_scheduler = pipeline.noise_scheduler
    vae_scale_factor = pipeline.config['latent_scale']

    # Tải Dữ liệu 
    train_dataset = SketchDataset(TRAIN_DATA_DIR, tokenizer, IMAGE_SIZE)
    val_dataset = SketchDataset(VAL_DATA_DIR, tokenizer, IMAGE_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Đã tải {len(train_dataset)} ảnh train và {len(val_dataset)} ảnh val.")

    print("Khởi tạo Evaluation Pipeline (một lần)...")
    eval_pipeline = StableDiffusionPipeline(
        unet=unet, 
        vae=vae, 
        text_encoder=text_encoder,
        tokenizer=tokenizer, 
        scheduler=noise_scheduler,
        safety_checker=None, 
        feature_extractor=None,
    ).to(DEVICE)
    l
    eval_pipeline.set_progress_bar_config(disable=True)

    eval_pipeline.unet.eval()
    eval_pipeline.vae.eval()
    eval_pipeline.text_encoder.eval()

    gen = torch.Generator(device=DEVICE).manual_seed(42)

    optimizer = AdamW(unet.parameters(), lr=LEARNING_RATE)
    lr_scheduler = get_scheduler(
        name=pipeline.config['scheduler'],
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_loader) * EPOCHS),
    )

    metrics = {
        "fid": FrechetInceptionDistance(feature=64).to(DEVICE),
        "lpips": LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(DEVICE),
        "clip_score": CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").to(DEVICE)
    }

    start_epoch = 0
    history = {
        "train_loss": [], "val_loss": [], 
        "fid": [], "lpips": [], "clip_score": []
    }
    best_clip_score = 0.0

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Phát hiện checkpoint. Đang tải từ {CHECKPOINT_PATH}...")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            unet.load_state_dict(checkpoint['unet_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            history = checkpoint['history']
            best_clip_score = checkpoint['best_clip_score']
            print(f"Resume training từ epoch {start_epoch}")
        except Exception as e:
            print(f"Lỗi khi tải checkpoint: {e}. Bắt đầu lại từ đầu.")
            start_epoch = 0
            history = {k: [] for k in history}
            best_clip_score = 0.0
    else:
        print("Không tìm thấy checkpoint. Bắt đầu training từ đầu.")


    for epoch in range(start_epoch, EPOCHS):
        start_time_epoch = time.time()
        unet.train()
        epoch_train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch in pbar:
            images = batch["pixel_values"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.mean * vae_scale_factor
                text_embeds = text_encoder(input_ids)[0]
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=DEVICE)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            noise_pred = unet(noisy_latents, timesteps, text_embeds).sample
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            epoch_train_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})
            
        avg_train_loss = epoch_train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        
        # ---Chạy Đánh giá (Evaluation) ---

        eval_results = evaluate(
            eval_pipeline, gen, val_loader, metrics,
            unet, vae, text_encoder, noise_scheduler,
            vae_scale_factor, NUM_INFERENCE_STEPS
        )
        
        history["val_loss"].append(eval_results["val_loss"])
        history["fid"].append(eval_results["fid"])
        history["lpips"].append(eval_results["lpips"])
        history["clip_score"].append(eval_results["clip_score"])
        
        epoch_time_min = (time.time() - start_time_epoch) / 60
        
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} Results (Thời gian: {epoch_time_min:.2f} phút) ---")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {eval_results['val_loss']:.6f}")
        print(f"  LPIPS:      {eval_results['lpips']:.4f} (↓)")
        print(f"  FID:        {eval_results['fid']:.4f} (↓)")
        print(f"  CLIP Score: {eval_results['clip_score']:.4f} (↑)")
        
        if eval_results['clip_score'] > best_clip_score:
            best_clip_score = eval_results['clip_score']
            torch.save(unet.state_dict(), SAVE_UNET_PATH)
            print(f"Đã lưu UNet *tốt nhất* mới tại {SAVE_UNET_PATH} (CLIP Score: {best_clip_score:.4f})")

        print(f"Đang lưu checkpoint cuối cùng tại {CHECKPOINT_PATH}...")
        checkpoint = {
            'epoch': epoch + 1, 
            'unet_state_dict': unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'history': history,
            'best_clip_score': best_clip_score
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
    
    
    total_time_min = (time.time() - start_time_total) / 60
    print(f"\n--- Hoàn thành Giai đoạn 2 ---")
    print(f"Tổng thời gian chạy (phiên này): {total_time_min:.2f} phút")
    print(f"UNet đã train (tốt nhất) được lưu tại: {SAVE_UNET_PATH}")
    
    if history['train_loss']:
        plot_metrics(history, "unet_metrics_plot.png")
    else:
        print("Không có dữ liệu history để vẽ biểu đồ.")

if __name__ == "__main__":
    main()