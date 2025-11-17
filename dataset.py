import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random

class SketchDataset(Dataset):
    """
    Dataset tùy chỉnh cho bộ dữ liệu sketch.
    Cấu trúc thư mục mong đợi:
    /data_root
        /train
            /images
                00001.png
                ...
            /texts
                00001.txt
                ...
        /val
            ...
    """
    def __init__(self, data_root, tokenizer, image_size=128):
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, "images")
        self.text_dir = os.path.join(data_root, "texts")
        self.tokenizer = tokenizer
        
        # Lấy danh sách tên file (không có đuôi)
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Định nghĩa các phép biến đổi ảnh
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # Chuẩn hóa về [-1, 1]
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. Tải ảnh
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB") # VAE luôn cần 3 kênh
            image = self.transform(image)
        except Exception as e:
            print(f"Lỗi khi tải ảnh {img_path}: {e}")
            # Trả về ảnh rỗng nếu lỗi
            return self.__getitem__((idx + 1) % len(self)) 

        # 2. Tải text
        text_name = os.path.splitext(img_name)[0] + ".txt"
        text_path = os.path.join(self.text_dir, text_name)
        
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                captions = f.read().splitlines()
            
            # Chọn ngẫu nhiên 1 caption nếu có nhiều
            caption = random.choice(captions).strip()
            if not caption:
                raise ValueError("Caption rỗng")
                
        except Exception as e:
            print(f"Lỗi khi đọc file text {text_path}: {e}. Dùng caption mặc định.")
            caption = "a sketch of a person" # Caption dự phòng

        # 3. Tokenize text
        inputs = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length, # Thường là 77
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image,
            "input_ids": inputs.input_ids.squeeze(0) # (77,)
        }