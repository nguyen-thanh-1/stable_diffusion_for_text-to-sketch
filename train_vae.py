import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import AdamW
from PIL import Image
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

# Giả sử file mini_pipeline.py và dataset.py nằm cùng thư mục
# MỚI: Đổi tên import khớp với file của bạn
from stable_diffusion import MiniDiffusionPipeline
from dataset import SketchDataset

# --- Cấu hình ---
TRAIN_DATA_DIR = r"C:\Users\Admin\Desktop\scientific research\dataset\train" # Thay đổi
VAL_DATA_DIR = r"C:\Users\Admin\Desktop\scientific research\dataset\val"     # Thay đổi
IMAGE_SIZE = 128
EPOCHS = 33           # Số epochs để fine-tune VAE
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
SAVE_PATH = "vae-finetuned.safetensors" # Sẽ lưu VAE *tốt nhất*
CHECKPOINT_PATH = "vae_latest_checkpoint.pth" # MỚI: File để resume
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def plot_losses(train_losses, val_losses, filename="vae_loss_plot.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("VAE Fine-tuning Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig(filename)
    print(f"Đã lưu biểu đồ loss tại {filename}")

def main():
    print("--- Giai đoạn 1: Fine-tuning VAE ---")
    
    # 1. Khởi tạo Pipeline (chỉ để lấy VAE và tokenizer)
    # Dùng VAE-MSE sắc nét làm VAE gốc
    pipeline = MiniDiffusionPipeline(
        base_model_id="runwayml/stable-diffusion-v1-5",
        vae_model_id="stabilityai/sd-vae-ft-mse",
        device=DEVICE
    )
    vae = pipeline.vae
    tokenizer = pipeline.tokenizer # Cần cho SketchDataset
    vae_scale_factor = pipeline.config['latent_scale']

    # 2. Tải Dữ liệu
    # Dùng Dataset nhưng chỉ lấy 'pixel_values'
    train_dataset = SketchDataset(TRAIN_DATA_DIR, tokenizer, IMAGE_SIZE)
    val_dataset = SketchDataset(VAL_DATA_DIR, tokenizer, IMAGE_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Đã tải {len(train_dataset)} ảnh train và {len(val_dataset)} ảnh val.")
    
    # 3. Thiết lập Training
    optimizer = AdamW(vae.parameters(), lr=LEARNING_RATE)
    
    # MỚI: Khởi tạo các biến để resume
    start_epoch = 0
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # MỚI: Logic tải checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Phát hiện checkpoint. Đang tải từ {CHECKPOINT_PATH}...")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            
            vae.load_state_dict(checkpoint['vae_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] # Đây là epoch *tiếp theo*
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            best_val_loss = checkpoint['best_val_loss']
            
            print(f"Resume training từ epoch {start_epoch}")
        except Exception as e:
            print(f"Lỗi khi tải checkpoint: {e}. Bắt đầu lại từ đầu.")
            start_epoch = 0
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
    else:
        print("Không tìm thấy checkpoint. Bắt đầu training từ đầu.")
    
    start_time = time.time()
    
    # MỚI: Bắt đầu vòng lặp từ start_epoch
    for epoch in range(start_epoch, EPOCHS):
        vae.train()
        epoch_train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch in pbar:
            images = batch["pixel_values"].to(DEVICE)
            
            # Forward pass
            posterior = vae.encode(images).latent_dist
            latents = posterior.mean * vae_scale_factor
            
            # Giải nén
            reconstructions = vae.decode(latents / vae_scale_factor).sample
            
            # Tính loss
            loss = F.mse_loss(reconstructions, images)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})
            
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # --- Validation ---
        vae.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
            for batch in pbar_val:
                images = batch["pixel_values"].to(DEVICE)
                posterior = vae.encode(images).latent_dist
                latents = posterior.mean * vae_scale_factor
                reconstructions = vae.decode(latents / vae_scale_factor).sample
                loss = F.mse_loss(reconstructions, images)
                epoch_val_loss += loss.item()
                
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")
        
        # Lưu VAE tốt nhất (logic cũ)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(vae.state_dict(), SAVE_PATH)
            print(f"Đã lưu VAE *tốt nhất* mới tại {SAVE_PATH} (Val Loss: {best_val_loss:.6f})")

        # MỚI: Lưu checkpoint mới nhất để resume
        print(f"Đang lưu checkpoint cuối cùng tại {CHECKPOINT_PATH}...")
        checkpoint = {
            'epoch': epoch + 1, # Lưu epoch *tiếp theo*
            'vae_state_dict': vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, CHECKPOINT_PATH)


    end_time = time.time()
    # MỚI: Tính toán thời gian đã chạy
    total_time_min = (end_time - start_time) / 60
    print(f"\n--- Hoàn thành Giai đoạn 1 ---")
    print(f"Tổng thời gian chạy (phiên này): {total_time_min:.2f} phút")
    print(f"VAE đã fine-tune (tốt nhất) được lưu tại: {SAVE_PATH}")
    
    # Chỉ vẽ biểu đồ nếu có dữ liệu
    if train_losses and val_losses:
        plot_losses(train_losses, val_losses, "vae_loss_plot.png")

if __name__ == "__main__":
    main()