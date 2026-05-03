import gc
import logging

import numpy as np
import torch

from inpainting import INPmodel
from segmentation import SEGmodel

logger = logging.getLogger(__name__)


class CLEANER:
    # Розмір керується всередині INPmodel (_MODEL_SIZE = 512×512 для SD1.5)

    def __init__(
        self,
        seg_model: SEGmodel,
        inp_model: INPmodel,
    ) -> None:
        self.seg_model = seg_model
        self.inp_model = inp_model

    def clear_gpu_memory(self) -> None:
        gc.collect()
        torch.cuda.empty_cache()

    def clean_img(self, img: np.ndarray) -> tuple:
        """
        Приймає BGR np.ndarray (як після cv2.imread).
        Повертає tuple: (очищений_img, маска, кількість_символів).
        """
        # --- Сегментація ---
        self.seg_model.run(img)
        bw_mask = self.seg_model.get_mask()
        num_of_char = self.seg_model.get_numbers_of_char()
        logger.info("Знайдено символів: %d", num_of_char)

        # Якщо маски немає — повертаємо однакову структуру!
        if bw_mask is None:
            logger.warning("Маску не отримано, повертаємо оригінал")
            # ВИПРАВЛЕННЯ 3: Повертаємо 3 елементи, щоб не зламати розпакування
            return img, None, 0

        self.clear_gpu_memory()

        # --- Інпейнтинг ---
        self.inp_model.make_pipe()
        cleaned_img = self.inp_model.clean_(img, bw_mask)

        return cleaned_img, bw_mask, num_of_char