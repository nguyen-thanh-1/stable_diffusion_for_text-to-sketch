import torch
from diffusers.models import UNet2DConditionModel, AutoencoderKL
from diffusers.schedulers import DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Dict, Any, Optional

class MiniDiffusionPipeline:
    """
    Một class "pipeline" tùy chỉnh để gom UNet "mini", VAE, CLIP
    và các cấu hình training/diffusion.
    """
    
    # Lấy config mặc định bạn cung cấp
    DEFAULT_CONFIG: Dict[str, Any] = {
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "beta_end": 0.0120,
        "num_train_timesteps": 1000,
        "prediction_type": "epsilon",
        "variance_type": "fixed_small",
        "clip_sample": False,
        "rescale_betas_zero_snr": False,
        "timestep_spacing": "leading",
        "lr": 1e-4,
        "optimizer": "AdamW",
        "scheduler": "cosine",
        "ema_decay": 0.9999,
        "latent_scale": 0.18215,
        "text_embed_dim": 768,
        "latent_channels": 4,
        "latent_downscale_factor": 8,
        
        # --- Bổ sung: Cấu hình kiến trúc UNet-mini ---
        # (Bạn có thể ghi đè chúng bằng 'config_overrides')
        "image_size": 128, # Kích thước ảnh gốc
        "unet_block_out_channels": (256, 512, 1024),
        "unet_layers_per_block": 1,
        "unet_down_block_types": (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        "unet_up_block_types": (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        "unet_mid_block_type": "UNetMidBlock2DCrossAttn",
        "unet_attention_head_dim": 8,
    }

    def __init__(
        self,
        base_model_id: str = "stabilityai/stable-diffusion-v1-5",
        vae_model_id: Optional[str] = None,
        device: str = "cpu",
        config_overrides: Optional[Dict[str, Any]] = None
    ):
        """
        Khởi tạo pipeline.

        Args:
            base_model_id: ID model Hugging Face để tải CLIP và Tokenizer
                           (và VAE nếu 'vae_model_id' không được cung cấp).
            vae_model_id: (Tùy chọn) ID model để tải VAE.
                          Dùng "stabilityai/sd-vae-ft-mse" cho VAE sắc nét hơn.
            device: "cpu" hoặc "cuda".
            config_overrides: Một dict để ghi đè bất kỳ giá trị nào trong
                              DEFAULT_CONFIG.
        """
        self.device = torch.device(device)
        
        # 1. Hợp nhất cấu hình (Config)
        # Bắt đầu với config mặc định, sau đó ghi đè bằng config_overrides
        self.config = {**self.DEFAULT_CONFIG, **(config_overrides or {})}

        # 2. Tải các thành phần
        print(f"Đang tải Tokenizer và Text Encoder (đã đóng băng) từ {base_model_id}...")
        self.tokenizer = self._load_tokenizer(base_model_id)
        self.text_encoder = self._load_text_encoder(base_model_id)

        # Quyết định tải VAE từ đâu
        _vae_id = vae_model_id or base_model_id
        _vae_subfolder = "vae" if vae_model_id is None else None
        print(f"Đang tải VAE (để fine-tune) từ {_vae_id}...")
        self.vae = self._load_vae(_vae_id, _vae_subfolder)
        
        print("Khởi tạo UNet-mini (với trọng số ngẫu nhiên)...")
        self.unet = self._load_mini_unet()

        print("Khởi tạo Noise Scheduler...")
        self.noise_scheduler = self._load_noise_scheduler()
        
        print("\n--- MiniDiffusionPipeline đã sẵn sàng! ---")
        self.print_model_stats()

    def _load_tokenizer(self, model_id: str) -> CLIPTokenizer:
        return CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    def _load_text_encoder(self, model_id: str) -> CLIPTextModel:
        model = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        model.to(self.device)
        model.requires_grad_(False) # Đóng băng CLIP
        return model

    def _load_vae(self, model_id: str, subfolder: Optional[str]) -> AutoencoderKL:
        if subfolder:
            model = AutoencoderKL.from_pretrained(model_id, subfolder=subfolder)
        else:
            model = AutoencoderKL.from_pretrained(model_id)
        model.to(self.device)
        # Không đóng băng VAE, vì bạn muốn fine-tune nó
        return model

    def _load_mini_unet(self) -> UNet2DConditionModel:
        """Xây dựng config UNet từ self.config và khởi tạo mô hình."""
        
        # Tính toán kích thước latent
        latent_size = self.config["image_size"] // self.config["latent_downscale_factor"]
        
        unet_config = {
            "sample_size": latent_size,
            "in_channels": self.config["latent_channels"],
            "out_channels": self.config["latent_channels"],
            "block_out_channels": self.config["unet_block_out_channels"],
            "layers_per_block": self.config["unet_layers_per_block"],
            "down_block_types": self.config["unet_down_block_types"],
            "up_block_types": self.config["unet_up_block_types"],
            "mid_block_type": self.config["unet_mid_block_type"],
            "cross_attention_dim": self.config["text_embed_dim"],
            "attention_head_dim": self.config["unet_attention_head_dim"],
        }
        
        model = UNet2DConditionModel(**unet_config)
        model.to(self.device)
        return model

    def _load_noise_scheduler(self) -> DDPMScheduler:
        """Khởi tạo scheduler từ self.config."""
        # DDPMScheduler.from_config sẽ tự động chọn các key
        # có liên quan từ dict config
        return DDPMScheduler.from_config(self.config)

    def print_model_stats(self):
        """In thông số của các mô hình có thể huấn luyện."""
        unet_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        vae_params = sum(p.numel() for p in self.vae.parameters() if p.requires_grad)
        print(f"  UNet-mini (để train): {unet_params / 1_000_000:.2f} triệu tham số")
        print(f"  VAE (để fine-tune): {vae_params / 1_000_000:.2f} triệu tham số")
        
    def get_trainable_parameters(self) -> Dict[str, Any]:
        """Tiện ích trả về các tham số có thể huấn luyện."""
        return {
            "unet": self.unet.parameters(),
            "vae": self.vae.parameters()
        }


# --- KHỐI KIỂM THỬ (SMOKE TEST) ---
def _run_smoke_test():
    print("--- Bắt đầu kiểm thử MiniDiffusionPipeline ---")
    
    if not torch.cuda.is_available():
        print("CẢNH BÁO: Không tìm thấy CUDA. Chạy trên CPU (sẽ chậm).")
        device = "cpu"
    else:
        device = "cuda"

    # --- Kịch bản 1: Tải mặc định (dùng VAE của 1.5) ---
    print("\n--- Kịch bản 1: Tải mặc định ---")
    pipeline_1 = MiniDiffusionPipeline(
        base_model_id="runwayml/stable-diffusion-v1-5",
        device=device
    )
    
    # --- Kịch bản 2: Tải VAE-MSE (sắc nét hơn) ---
    print("\n--- Kịch bản 2: Tải VAE-MSE tùy chỉnh ---")
    pipeline_2 = MiniDiffusionPipeline(
        base_model_id="runwayml/stable-diffusion-v1-5", # Vẫn dùng CLIP/Tokenizer của 1.5
        vae_model_id="stabilityai/sd-vae-ft-mse",      # Nhưng dùng VAE riêng
        device=device
    )

    # --- Kịch bản 3: Ghi đè config (ví dụ: mô hình nhỏ hơn) ---
    print("\n--- Kịch bản 3: Ghi đè config (UNet siêu nhỏ) ---")
    tiny_config = {
        "unet_block_out_channels": (128, 256, 512), # Nhỏ hơn
        "lr": 5e-5 # Learning rate khác
    }
    pipeline_3 = MiniDiffusionPipeline(
        base_model_id="runwayml/stable-diffusion-v1-5",
        device=device,
        config_overrides=tiny_config
    )
    
    print("\n--- Kiểm thử thành công! ---")
    print(f"Config LR của Pipeline 1: {pipeline_1.config['lr']}") # 1e-4
    print(f"Config LR của Pipeline 3: {pipeline_3.config['lr']}") # 5e-5


if __name__ == "__main__":
    # Dòng này đảm bảo code kiểm thử chỉ chạy khi bạn
    # chạy file này trực tiếp: python mini_pipeline.py
    _run_smoke_test()