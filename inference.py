import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from safetensors.torch import load_file 

from stable_diffusion import MiniDiffusionPipeline

# --- Cấu hình ---
PROMPT = "a woman smiling, pointy nose, wavy hair"
SAVE_IMAGE_PATH = "./generated_image.png"

UNET_SAFE_PATH = "./unet-mini.safetensors"
VAE_SAFE_PATH = "./vae-finetuned.safetensors"

BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


TINY_UNET_CONFIG = {
    "unet_block_out_channels": (128, 256, 512),
}

@torch.no_grad()
def main():
    print("--- Bắt đầu quá trình Inference (từ Safetensors) ---")
    
    # --- Khởi tạo MiniDiffusionPipeline ---
    print(f"Đang tải pipeline gốc từ {BASE_MODEL_ID}...")
    container = MiniDiffusionPipeline(
        base_model_id=BASE_MODEL_ID,
        device=DEVICE,
        config_overrides=TINY_UNET_CONFIG 
    )

    # --- Tải trọng số đã huấn luyện ---
    
    # Tải UNet
    print(f"Đang tải trọng số UNet từ {UNET_SAFE_PATH}...")
    try:
        unet_weights = load_file(UNET_SAFE_PATH, device=DEVICE)
        container.unet.load_state_dict(unet_weights)
    except Exception as e:
        print(f"LỖI: Không thể tải UNet state dict: {e}")
        print("Kiểm tra xem bạn đã bỏ chú thích 'config_overrides=TINY_UNET_CONFIG' chưa?")
        return

    # Tải VAE
    print(f"Đang tải trọng số VAE từ {VAE_SAFE_PATH}...")
    try:
        vae_weights = load_file(VAE_SAFE_PATH, device=DEVICE)
        container.vae.load_state_dict(vae_weights)
    except Exception as e:
        print(f"LỖI: Không thể tải VAE state dict: {e}")
        return

    # --- Khởi tạo StableDiffusionPipeline ---
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

    # --- Tạo ảnh ---
    print(f"\nĐang tạo ảnh cho prompt: '{PROMPT}'")
    generator = torch.Generator(device=DEVICE).manual_seed(42)
    
    image = inference_pipeline(
        prompt=PROMPT,
        num_inference_steps=50,
        generator=generator,
        guidance_scale=7.5 
    ).images[0]

    # --- Lưu ảnh ---
    image.save(SAVE_IMAGE_PATH)
    print(f"\n--- Hoàn thành! ---")
    print(f"Đã lưu ảnh tại: {SAVE_IMAGE_PATH}")
    
    try:
        image.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()