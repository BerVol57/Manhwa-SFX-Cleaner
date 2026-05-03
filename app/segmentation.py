import cv2
import logging
import numpy as np
import torch
from ultralytics import YOLO

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SEGmodel:

    pth_model: str

    def __init__(self, pth: str) -> None:
        self.pth_model = pth
        self.results = None
        try:
            self.model = YOLO(self.pth_model)
        except Exception as e:
            logger.error("Не вдалось завантажити модель '%s': %s", pth, e)
            raise

    def train(
        self,
        sfx_data: str,
        epochs: int = 100,
        imgsz: int = 1024,
        workers: int = 0,
    ) -> None:
        self.model.train(
            data=sfx_data,
            epochs=epochs,
            imgsz=imgsz,
            workers=workers,
        )

    def run(self, img: np.ndarray, conf: float = 0.25) -> None:
        """Запускає інференс. img — BGR np.ndarray (читається cv2)."""
        try:
            self.img = img
            self.results = self.model.predict(source=self.img, conf=conf)[0]
        except Exception as e:
            logger.error("Помилка під час predict: %s", e)
            raise

    def get_mask_model(self) -> np.ndarray | None:
        """
        Повертає бінарну маску (uint8, значення 0/255) розміром orig_shape,
        або None якщо жодного об'єкту не знайдено.
        """
        if self.results is None:
            raise RuntimeError("Спочатку викличте run()")

        res = self.results[0]
        if res.masks is None or len(res.masks) == 0:
            logger.warning("Маски не знайдено — повертаємо None")
            return None

        # Об'єднуємо всі маски через логічне OR
        combined_mask = torch.any(res.masks.data, dim=0).cpu().numpy()

        orig_h, orig_w = res.orig_shape
        mask_resized = cv2.resize(
            combined_mask.astype(np.uint8),
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )
        self.model_mask = (mask_resized * 255).astype(np.uint8)
    
    def get_mask(self) -> np.ndarray | None:
        num_chars = self.get_numbers_of_char()
        if num_chars == 0:
            return None

        perfect_mask = np.zeros(self.img.shape[:2], dtype=np.uint8)

        for i in range(num_chars):
            yolo_mask = self.results.masks.data[i].cpu().numpy()
            yolo_mask = cv2.resize(yolo_mask, 
                                (self.img.shape[1], self.img.shape[0]))

            box = self.results.boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            
            # Обрізаємо ROI — GrabCut тільки на потрібній ділянці
            roi = self.img[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            gc_mask = np.zeros(self.img.shape[:2], np.uint8)
            gc_mask[yolo_mask > 0.5] = cv2.GC_PR_FGD

            # Масштабований радіус відносно розміру символу
            radius = max(3, int(min(x2 - x1, y2 - y1) * 0.15))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(gc_mask, (cx, cy), radius, cv2.GC_FGD, thickness=-1)

            gc_mask_roi = gc_mask[y1:y2, x1:x2]
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)

            cv2.grabCut(roi, gc_mask_roi, None,
                        bgdModel, fgdModel, 3,
                        cv2.GC_INIT_WITH_MASK)

            temp_mask = np.zeros(self.img.shape[:2], np.uint8)
            temp_mask[y1:y2, x1:x2] = np.where(
                (gc_mask_roi == cv2.GC_BGD) | (gc_mask_roi == cv2.GC_PR_BGD),
                0, 255
            ).astype(np.uint8)

            # Fallback якщо GrabCut не впорався
            if temp_mask.max() == 0:
                logger.warning("GrabCut порожній для символу %d — fallback на YOLO", i)
                temp_mask = (yolo_mask > 0.5).astype(np.uint8) * 255

            perfect_mask = cv2.bitwise_or(perfect_mask, temp_mask)

        # Менш агресивний dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)) 
        perfect_mask = cv2.dilate(perfect_mask, kernel, iterations=1)
        
        perfect_mask = cv2.GaussianBlur(perfect_mask, (21, 21), 0)

        return perfect_mask

    def get_numbers_of_char(self) -> int:
        if self.results is None:
            raise RuntimeError("Спочатку викличте run()")
        masks = self.results.masks
        return len(masks) if masks is not None else 0