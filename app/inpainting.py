import gc
import cv2
import numpy as np
import torch
from iopaint.model.mat import MAT
from iopaint.schema import InpaintRequest

device = "cuda" if torch.cuda.is_available() else "cpu"

class INPmodel:
    def __init__(self, device: str = device) -> None:
        self.device = torch.device(device)
        self.model = None

    def make_pipe(self) -> None:
        if self.model is None:
            print("Завантаження моделі MAT (Mask-Aware Transformer)...")
            self.model = MAT(device=self.device)

    def clean_(self, init_img_bgr: np.ndarray, mask_img_bw: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Спочатку викличте make_pipe()")

        # 1. Підготовка зображення (RGB)
        img_rgb = cv2.cvtColor(init_img_bgr, cv2.COLOR_BGR2RGB)

        # 2. Підготовка маски
        if len(mask_img_bw.shape) == 3:
            mask_bw = cv2.cvtColor(mask_img_bw, cv2.COLOR_BGR2GRAY)
        else:
            mask_bw = mask_img_bw.copy()

        config = InpaintRequest(
            hd_strategy="Original"
        )

        with torch.no_grad():
            result_rgb = self.model(img_rgb, mask_bw, config)

        if result_rgb.dtype != np.uint8:
            result_rgb = np.clip(result_rgb, 0, 255).astype(np.uint8)
        # ------------------------------------------
        
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        gc.collect()
        torch.cuda.empty_cache()

        return result_bgr
    
    def unload(self) -> None:
        if self.model is not None:
            # Трюк для Windows: переносимо на CPU перед видаленням
            if hasattr(self.model, 'model'):
                self.model.model.to('cpu')
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()