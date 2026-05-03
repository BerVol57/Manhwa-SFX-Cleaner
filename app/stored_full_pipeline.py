"""
full_pipeline.py
================
End-to-end pipeline: Korean manhwa SFX → segmentation → inpainting →
VLM translation → stylised Cyrillic typography → final composition.

Dependency graph
----------------
  original image
       │
       ▼
  SEGmodel.run()          ──► binary mask  ──► num_of_char
       │                                          │
       ▼                                          │
  INPmodel.clean_()       ──► clean_bg            │
       │                                          │
       ├──────────────────────────────────────────┘
       ▼
  VLMTranslator                ──► ukrainian_text
       │
       ▼
  TTRenderer                   ──► stylised_rgba (BGRA, 512×512)
       │
       ▼
  Compositor.place_on_background()  ──► final_image (BGR)
"""

import gc
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

# ── Project modules ────────────────────────────────────────────────────────────
from clean_pipeline import CLEANER
from inpainting import INPmodel
from segmentation import SEGmodel

logger = logging.getLogger(__name__)


ROOT_DIR = Path(__file__).resolve().parent.parent

class Compositor:
    """
    Places a BGRA stylised text image back onto the cleaned background,
    respecting the spatial bounding box of the original mask.
    """

    @staticmethod
    def place_on_background(
        clean_bg: np.ndarray,        # BGR, full-size
        stylised_bgra: np.ndarray,   # BGRA, 512×512
        binary_mask: np.ndarray,     # uint8 0/255, full-size (from segmentation)
    ) -> np.ndarray:
        """
        Warps the 512×512 stylised text into the bounding box of binary_mask
        and alpha-blends it onto clean_bg.  Returns BGR.
        """
        x, y, w, h = cv2.boundingRect(binary_mask)
        if w == 0 or h == 0:
            logger.warning("[Compositor] Empty bounding rect — returning clean_bg unchanged.")
            return clean_bg.copy()

        # Split stylised image into BGR + alpha
        b, g, r, alpha = cv2.split(stylised_bgra)
        text_bgr = cv2.merge((b, g, r))                           # 512×512 BGR
        text_alpha = alpha.astype(np.float32) / 255.0             # 512×512 float

        # Scale stylised text to fill the original bounding box
        text_bgr_scaled = cv2.resize(text_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
        alpha_scaled = cv2.resize(text_alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        alpha_3c = np.stack([alpha_scaled] * 3, axis=-1)

        # Alpha-blend into clean background
        result = clean_bg.copy()
        roi = result[y:y + h, x:x + w].astype(np.float32)
        blended = alpha_3c * text_bgr_scaled.astype(np.float32) + (1 - alpha_3c) * roi
        result[y:y + h, x:x + w] = blended.astype(np.uint8)

        return result


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Master Pipeline
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    seg_model_path: str                          # Path to YOLO .pt weights
    font_path: str = "../Grinched.ttf"
    dict_path: str = "../manhwa_onomatopoeia.json"
    chroma_db_path: str = ".././chroma_db"
    output_dir: str = "./output"
    seg_conf: float = 0.25


class ManhwaPipeline:
    """
    Orchestrates the full Korean→Ukrainian SFX translation pipeline.

    Usage
    -----
    >>> pipe = ManhwaPipeline(PipelineConfig(seg_model_path="best.pt"))
    >>> result = pipe.run("page.jpg")
    >>> cv2.imwrite("translated.png", result)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)

        # Segmentation + inpainting (always in memory)
        seg = SEGmodel(config.seg_model_path)
        inp = INPmodel()
        self.cleaner = CLEANER(seg_model=seg, inp_model=inp)

        # VLM and typography are lazy-loaded inside their classes
        self.vlm = VLMTranslator()
        self.tt = TTRenderer()
        self.compositor = Compositor()

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _save_temp_image(img: np.ndarray) -> str:
        """Save a BGR numpy array to a temp file; return its path."""
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, img)
        return tmp.name

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        image_path: str,
        save_intermediates: bool = False,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        image_path : str
            Path to input BGR image (manhwa page or panel).
        save_intermediates : bool
            If True, saves mask and clean_bg to config.output_dir.

        Returns
        -------
        np.ndarray
            Final BGR image with translated, stylised text composited back.
        """
        # ── Stage 1: Load ─────────────────────────────────────────────────────
        original = cv2.imread(image_path)
        if original is None:
            raise FileNotFoundError(f"Cannot open image: {image_path}")
        logger.info("Loaded image %s  shape=%s", image_path, original.shape)
        gc.collect()
        # ── Stage 2: Segment + Inpaint ────────────────────────────────────────
        logger.info("=== Stage 2: Clean Pipeline ===")
        clean_bg, mask, num_chars = self.cleaner.clean_img(original)
        gc.collect()
        if num_chars == 0 or mask is None:
            logger.warning("No SFX detected — returning original image unchanged.")
            return original
        
        logger.info("Звільняємо VRAM для перекладача...")
        self.cleaner.inp_model.unload() # викличе новий метод з inpainting.py
        # Якщо в SEGmodel ще немає unload, просто:
        if hasattr(self.cleaner.seg_model, 'model'):
             self.cleaner.seg_model.model.to('cpu')
             del self.cleaner.seg_model.model
        
        gc.collect()
        torch.cuda.empty_cache()
        # ============================
        
        logger.info("Detected %d character cluster(s).", num_chars)
        # Pass the original image (the VLM sees Korean SFX in context)
        tmp_path = self._save_temp_image(original)
        try:
            ukrainian_text = self.vlm.translate(tmp_path)
        finally:
            os.unlink(tmp_path)

        logger.info("Ukrainian translation: %s", ukrainian_text)
        self.vlm.unload()   # free VRAM before running SD pipeline

        # ── Stage 4: Stylised Typography ──────────────────────────────────────
        logger.info("=== Stage 4: Typography Rendering ===")
        stylised_bgra = self.tt.render(
            ukrainian_text=ukrainian_text,
            original_img=original,
            binary_mask=mask,
        )

        if save_intermediates:
            cv2.imwrite(
                f"{self.config.output_dir}/{Path(image_path).stem}_stylised.png",
                stylised_bgra,
            )

        # Вивантажуємо SD pipeline до Stage 5 — він більше не потрібен
        self.tt.unload()

        # ── Stage 5: Composition ──────────────────────────────────────────────
        logger.info("=== Stage 5: Final Composition ===")
        final = self.compositor.place_on_background(
            clean_bg=clean_bg,
            stylised_bgra=stylised_bgra,
            binary_mask=mask,
        )

        return final


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CLI entry point 
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Translate Korean manhwa SFX to Ukrainian and re-render in style."
    )
    parser.add_argument("image", help="Path to input manhwa image")
    parser.add_argument("--seg-model", default="best.pt",
                        help="Path to YOLO segmentation weights")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--save-intermediates", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig(
        seg_model_path=args.seg_model,
        output_dir=args.output_dir,
    )
    pipeline = ManhwaPipeline(cfg)
    result = pipeline.run(args.image, save_intermediates=args.save_intermediates)

    out_path = os.path.join(
        args.output_dir,
        Path(args.image).stem + "_translated.png",
    )
    cv2.imwrite(out_path, result)
    print(f"✅ Saved to {out_path}")