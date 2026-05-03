import gc
import logging
from pathlib import Path

import fix_hub

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

class STRenderer:
    """
    Converts a Ukrainian word + original mask geometry into a stylised
    Cyrillic text image matching the Korean original's rotation and texture.
    """

    _BASE_DIR = Path(__file__).resolve().parent.parent
    
    FONT_PATH = _BASE_DIR / "Grinched.ttf" 
    FALLBACK_FONT = "arialbd.ttf"
    CANVAS_SIZE = (512, 512)

    def __init__(self) -> None:
        # SD-pipeline кешується між викликами — не завантажуємо щоразу
        # ЖОДНОГО завантаження моделей під час __init__ (Lazy Loading)
        self._sd_pipe = None
        self._controlnet = None

    def unload(self) -> None:
        import torch
        logger.info("[TT] Вивантаження SD1.5 + ControlNet з відеопам'яті...")
        if self._sd_pipe is not None:
            self._sd_pipe.to("cpu")
            del self._sd_pipe
        if self._controlnet is not None:
            self._controlnet.to("cpu")
            del self._controlnet
        self._sd_pipe = None
        self._controlnet = None
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()

    # ── Geometry helpers (from TypographyFeatureExtractor) ────────────────────

    @staticmethod
    def _extract_clusters(binary_mask: np.ndarray, mode: str = "scattered") -> list:
        """Extract per-character cluster geometry from binary mask."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        dilated = cv2.dilate(binary_mask, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) > 300]

        if not valid:
            default_corners = np.array(
                [[0, 0], [512, 0], [512, 512], [0, 512]], dtype="float32"
            )
            return [{"corners": default_corners, "center_x": 256, "center_y": 256,
                     "center": (256, 256), "size": (512, 512), "angle": 0.0}]

        clusters = []
        for c in valid:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            s = box.sum(axis=1)
            diff = np.diff(box, axis=1)
            corners = np.zeros((4, 2), dtype="float32")
            corners[0] = box[np.argmin(s)]
            corners[2] = box[np.argmax(s)]
            corners[1] = box[np.argmin(diff)]
            corners[3] = box[np.argmax(diff)]
            clusters.append({
                "corners": corners,
                "center_x": rect[0][0],
                "center_y": rect[0][1],
                "center": rect[0],
                "size": rect[1],
                "angle": rect[2],
            })

        std_x = np.std([cl["center_x"] for cl in clusters]) if len(clusters) > 1 else 1
        std_y = np.std([cl["center_y"] for cl in clusters]) if len(clusters) > 1 else 0
        clusters.sort(key=lambda x: x["center_x"] if std_x >= std_y else x["center_y"])
        return clusters

    @staticmethod
    def _analyze_edge_morphology(binary_mask: np.ndarray) -> str:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "neutral"
        mc = max(contours, key=cv2.contourArea)
        p = cv2.arcLength(mc, True)
        a = cv2.contourArea(mc)
        circularity = 4 * np.pi * (a / (p * p + 1e-6))
        return "rounded" if circularity > 0.18 else "sharp"

    @staticmethod
    def _detect_texture(original_img: np.ndarray, binary_mask: np.ndarray) -> dict:
        masked = cv2.bitwise_and(original_img, original_img, mask=binary_mask)
        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        _, sh = cv2.meanStdDev(hsv[:, :, 0], mask=binary_mask)
        _, sv = cv2.meanStdDev(hsv[:, :, 2], mask=binary_mask)
        color_var = max(float(sh[0][0]), float(sv[0][0]))
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return {
            "has_gradient": color_var > 8.0,
            "has_texture": lap_var > 300.0,
            "color_variance": color_var,
            "texture_complexity": lap_var,
        }

    # ── Text rendering ────────────────────────────────────────────────────────

    def _load_font(self, size: int = 150):
        from PIL import ImageFont
        try:
            return ImageFont.truetype(str(self.FONT_PATH), size)
        except Exception as e:
            logger.warning("Font not found at %s, using fallback. Error: %s", self.FONT_PATH, e)
            return ImageFont.load_default()

    @staticmethod
    def _split_text(text: str, n: int) -> list:
        if n >= len(text):
            return list(text) + [""] * (n - len(text))
        sz = max(1, len(text) // n)
        chunks = [text[i * sz:(i + 1) * sz] for i in range(n - 1)]
        chunks.append(text[(n - 1) * sz:])
        return chunks

    def _render_dynamic(self, text: str, clusters: list) -> np.ndarray:
        from PIL import Image as PILImage, ImageDraw

        font = self._load_font(150)
        chunks = self._split_text(text, len(clusters))
        centers = np.array([cl["center"] for cl in clusters])
        center_of_all = (np.min(centers, axis=0) + np.max(centers, axis=0)) / 2

        temp = np.zeros((2048, 2048), dtype=np.uint8)

        for chunk, cl in zip(chunks, clusters):
            if not chunk:
                continue

            base = PILImage.new("L", (800, 400), color=0)
            draw = ImageDraw.Draw(base)
            bbox = draw.textbbox((0, 0), chunk, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((400 - tw // 2 - bbox[0], 200 - th // 2 - bbox[1]),
                      chunk, fill=255, font=font)

            rendered = np.array(base)
            rx, ry, rw, rh = cv2.boundingRect(rendered)
            if rw == 0 or rh == 0:
                continue
            cropped = rendered[ry:ry + rh, rx:rx + rw]

            cw, ch = cl["size"]
            angle = cl["angle"]
            if cw < ch:
                cw, ch = ch, cw
                angle -= 90

            warped = cv2.resize(cropped, (int(cw), int(ch)), interpolation=cv2.INTER_CUBIC)

            pad = max(int(cw), int(ch)) * 2
            padded = np.zeros((pad, pad), dtype=np.uint8)
            px, py = (pad - int(cw)) // 2, (pad - int(ch)) // 2
            padded[py:py + int(ch), px:px + int(cw)] = warped
            M = cv2.getRotationMatrix2D((pad // 2, pad // 2), angle, 1.0)
            rotated = cv2.warpAffine(padded, M, (pad, pad), flags=cv2.INTER_CUBIC)

            rrx, rry, rrw, rrh = cv2.boundingRect(rotated)
            if rrw <= 0 or rrh <= 0:
                continue
            final_rot = rotated[rry:rry + rrh, rrx:rrx + rrw]

            cx, cy = cl["center"]
            off_x, off_y = cx - center_of_all[0], cy - center_of_all[1]
            sx = int(1024 + off_x - rrw // 2)
            sy = int(1024 + off_y - rrh // 2)
            if sx >= 0 and sy >= 0 and sx + rrw <= 2048 and sy + rrh <= 2048:
                roi = temp[sy:sy + rrh, sx:sx + rrw]
                temp[sy:sy + rrh, sx:sx + rrw] = np.maximum(roi, final_rot)

        fx, fy, fw, fh = cv2.boundingRect(temp)
        if fw > 0 and fh > 0:
            crop = temp[fy:fy + fh, fx:fx + fw]
            scale = min(460 / fw, 460 / fh)
            resized = cv2.resize(crop, (0, 0), fx=scale, fy=scale,
                                 interpolation=cv2.INTER_CUBIC)
            h, w = resized.shape
            canvas = np.zeros(self.CANVAS_SIZE, dtype=np.uint8)
            sx, sy = (512 - w) // 2, (512 - h) // 2
            canvas[sy:sy + h, sx:sx + w] = resized
            _, binary = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)
            return binary

        return np.zeros(self.CANVAS_SIZE, dtype=np.uint8)

    @staticmethod
    def _apply_edge_morphology(mask: np.ndarray, style: str) -> np.ndarray:
        if style == "rounded":
            blurred = cv2.GaussianBlur(mask, (25, 25), 0)
            _, out = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
            return out
        if style == "sharp":
            cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
            rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated = cv2.dilate(mask, cross, iterations=2)
            return cv2.erode(dilated, rect, iterations=1)
        return mask

    def _load_sd_pipe(self) -> None:
        if self._sd_pipe is not None:
            return
        import torch  # <-- ліниво тут
        from diffusers import (
            ControlNetModel,
            EulerAncestralDiscreteScheduler,
            StableDiffusionControlNetPipeline,
        )
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[TT] Завантаження SD1.5 + ControlNet + IP-Adapter …")
        self._controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16
        ).to(_device)
        self._sd_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self._controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None,
            feature_extractor=None,
        ).to(_device)
        self._sd_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self._sd_pipe.scheduler.config
        )
        self._sd_pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter-plus_sd15.safetensors",
        )
        self._sd_pipe.set_ip_adapter_scale(0.85)
        
        # Оптимізація пам'яті для Windows
        self._sd_pipe.enable_model_cpu_offload()
        self._sd_pipe.enable_vae_slicing()
        self._sd_pipe.enable_vae_tiling()
        logger.info("[TT] SD pipeline завантажено.")

    # ── Style transfer via SD1.5 + ControlNet + IP-Adapter ───────────────────

    def _generate_texture(self,
        ukr_mask: np.ndarray,
        style_ref_img: np.ndarray,  # BGR, original-size
        kor_mask: np.ndarray,       # binary, same size as style_ref_img
        texture_params: dict,
    ) -> np.ndarray:
        
        edges = cv2.Canny(ukr_mask.astype(np.uint8), 100, 200)
        cond_img = Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

        style_rgb = cv2.cvtColor(style_ref_img, cv2.COLOR_BGR2RGB)
        _, kor_bin = cv2.threshold(kor_mask, 127, 255, cv2.THRESH_BINARY)
        rx, ry, rw, rh = cv2.boundingRect(kor_bin)
        if rw > 0 and rh > 0:
            crop_s = style_rgb[ry:ry + rh, rx:rx + rw]
            crop_m = kor_bin[ry:ry + rh, rx:rx + rw]
            bg_mask = cv2.bitwise_not(crop_m)
            inpainted = cv2.inpaint(crop_s, bg_mask, 3, cv2.INPAINT_TELEA)
            style_rgb = cv2.resize(inpainted, (512, 512), interpolation=cv2.INTER_CUBIC)
        ref_img = Image.fromarray(style_rgb)

        self._load_sd_pipe()

        pos = ("highly detailed typography, manga comic text effect, stylized font, "
               "manhwa lettering style, perfect rendering")
        neg = ("distorted letters, bad anatomy, messy, artifacts, watermark, "
               "low resolution, standard font, background noise")
        if not texture_params["has_gradient"]:
            pos += ", flat color, solid fill, monochrome design, no gradient"
            neg += ", gradient, 3d render, shadows, highlights"
        pos += (", rich texture, halftone dots, comic screentones, distressed surface"
                if texture_params["has_texture"] else
                ", clean surface, smooth vector art")

        result = self._sd_pipe(
            prompt=pos, negative_prompt=neg,
            image=cond_img, ip_adapter_image=ref_img,
            num_inference_steps=30, guidance_scale=7.5,
            controlnet_conditioning_scale=1.0,
        ).images[0]

        return np.array(result)

    @staticmethod
    def _composite_text(stylised_rgb: np.ndarray,
                        ukr_mask: np.ndarray) -> np.ndarray:
        h, w = ukr_mask.shape
        if stylised_rgb.shape[:2] != (h, w):
            stylised_rgb = cv2.resize(stylised_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        bgr = cv2.cvtColor(stylised_rgb, cv2.COLOR_RGB2BGR)
        b, g, r = cv2.split(bgr)
        alpha = cv2.GaussianBlur(ukr_mask, (3, 3), 0)
        return cv2.merge((b, g, r, alpha))

    # ── Public API ────────────────────────────────────────────────────────────

    def render(
        self,
        ukrainian_text: str,
        original_img: np.ndarray,
        binary_mask: np.ndarray,
    ) -> np.ndarray:
        logger.info("[TT] Extracting geometry …")
        _, bin_mask_small = cv2.threshold(
            cv2.resize(binary_mask, (512, 512)), 127, 255, cv2.THRESH_BINARY
        )
        clusters = self._extract_clusters(bin_mask_small)
        edge_style = self._analyze_edge_morphology(bin_mask_small)
        texture_params = self._detect_texture(original_img, binary_mask)

        logger.info(
            "[TT] Clusters=%d  EdgeStyle=%s  Texture=%s",
            len(clusters), edge_style, texture_params,
        )

        logger.info("[TT] Rendering Cyrillic text …")
        ukr_mask = self._render_dynamic(ukrainian_text, clusters)
        ukr_mask = self._apply_edge_morphology(ukr_mask, edge_style)

        logger.info("[TT] Generating stylised texture (SD1.5+ControlNet+IP-Adapter) …")
        orig_512 = cv2.resize(original_img, (512, 512))
        kor_mask_512 = cv2.resize(binary_mask, (512, 512))
        stylised_rgb = self._generate_texture(ukr_mask, orig_512, kor_mask_512, texture_params)

        logger.info("[TT] Compositing text onto transparent canvas …")
        return self._composite_text(stylised_rgb, ukr_mask)