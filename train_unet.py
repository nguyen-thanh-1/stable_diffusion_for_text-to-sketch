import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
# KHÔNG CẦN IMPORT SAFETENSORS NỮA
# from safetensors.torch import load_file 

# Giả sử file stable_diffusion.py (chứa MiniDiffusionPipeline)
from stable_diffusion import MiniDiffusionPipeline

# --- 1. Cấu hình ---
PROMPT = "a black and white sketch of a young woman with her hair in a messy bun, wearing a cozy sweater"
SAVE_IMAGE_PATH = "./generated_image.png"

# Đường dẫn đến các file .safetensors ĐÃ CẬP NHẬT
UNET_SAFE_PATH = "./unet-mini.safetensors"
VAE_SAFE_PATH = "./vae-finetuned.safetensors"

BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cấu hình UNet-mini (phải GIỐNG HỆT lúc train)
TINY_UNET_CONFIG = {
    "unet_block_out_channels": (128, 256, 512),
}

@torch.no_grad()
def main():
    print("--- Bắt đầu quá trình Inference (từ Safetensors) ---")
    
    # --- 2. Khởi tạo "Thùng chứa" (MiniDiffusionPipeline) ---
    print(f"Đang tải pipeline gốc từ {BASE_MODEL_ID}...")
    container = MiniDiffusionPipeline(
        base_model_id=BASE_MODEL_ID,
        device=DEVICE,
        config_overrides=TINY_UNET_CONFIG 
    )

    # --- 3. Tải trọng số đã huấn luyện (TỪ SAFETENSORS) ---
    
    # Tải UNet
    print(f"Đang tải trọng số UNet từ {UNET_SAFE_PATH}...")
    try:
        # THAY ĐỔI: Dùng torch.load() vì file của bạn là file .pth
        unet_weights = torch.load(UNET_SAFE_PATH, map_location=DEVICE)
        container.unet.load_state_dict(unet_weights)
    except Exception as e:
        print(f"LỖI: Không thể tải UNet state dict: {e}")
        return

    # Tải VAE
    print(f"Đang tải trọng số VAE từ {VAE_SAFE_PATH}...")
    try:
        # THAY ĐỔI: Dùng torch.load() vì file của bạn là file .pth
        vae_weights = torch.load(VAE_SAFE_PATH, map_location=DEVICE)
        container.vae.load_state_dict(vae_weights)
    except Exception as e:
        print(f"LỖI: Không thể tải VAE state dict: {e}")
        return

    # --- 4. Khởi tạo "Người thực thi" (StableDiffusionPipeline) ---
    torch_dtype = torch.float16 if DEVICE.startswith("cuda") else torch.float32
    
    print("Đang tạo pipeline inference...")
    inference_pipeline = StableDiffusionPipeline(
        unet=container.unet,
        vae=container.vae,
        text_encoder=container.text_encoder,
        tokenizer=container.tokenizer,
        scheduler=container.noise_scheduler,
        safety_checker=None,
        feature_extractor=None,
    ).to(DEVICE)

    if DEVICE.startswith("cuda"):
         inference_pipeline.to(dtype=torch_dtype)

    inference_pipeline.set_progress_bar_config(disable=False)

    # --- 5. Tạo ảnh ---
    print(f"\nĐang tạo ảnh cho prompt: '{PROMPT}'")
    generator = torch.Generator(device=DEVICE).manual_seed(42)
    
    image = inference_pipeline(
        prompt=PROMPT,
        num_inference_steps=50,
        generator=generator,
        guidance_scale=7.5 
    ).images[0]

    # --- 6. Lưu ảnh ---
    image.save(SAVE_IMAGE_PATH)
    print(f"\n--- Hoàn thành! ---")
    print(f"Đã lưu ảnh tại: {SAVE_IMAGE_PATH}")
    
    try:
        image.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()