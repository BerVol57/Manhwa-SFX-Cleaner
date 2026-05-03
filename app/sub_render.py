import sys
import cv2
import numpy as np
from vlm_interpreter import VLMInterpreter
from st_render import STRenderer

import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import fix_hub

def main(orig_img_path):
    # 1. Отримуємо переклад від VLM
    vlm = VLMInterpreter()
    translation = vlm.translate(orig_img_path)
    
    # with open("TEXT.txt", "w", encoding="utf-8") as f:
    #     f.write(translation)
    
    # 2. Завантажуємо результати очищення
    cleaned_bg = cv2.imread("temp_cache/cleaned.png")
    mask = cv2.imread("temp_cache/mask.png", cv2.IMREAD_GRAYSCALE)
    orig_img = cv2.imread(orig_img_path)
    
    h, w = cleaned_bg.shape[:2]

    # 3. Рендеримо стилізований текст
    renderer = STRenderer()
    # render() повертає BGRA шар з українським текстом
    stylized_layer = renderer.render(translation, orig_img, mask)
    full_stylized_layer = np.zeros((h, w, 4), dtype=np.uint8)
    
    x, y, bw, bh = cv2.boundingRect(mask)
    
    text_fit = cv2.resize(stylized_layer, (bw, bh))
    
    full_stylized_layer[y:y+bh, x:x+bw] = text_fit
    stylized_layer = full_stylized_layer
    
    # 4. Композитинг (Накладання тексту на фон)
    text_rgb = stylized_layer[:, :, :3]
    text_alpha = stylized_layer[:, :, 3] / 255.0
    
    for c in range(3):
        cleaned_bg[:, :, c] = (1.0 - text_alpha) * cleaned_bg[:, :, c] + text_alpha * text_rgb[:, :, c]

    cv2.imwrite("output/final_page.png", cleaned_bg)
    
    sys.stdout.reconfigure(encoding='utf-8')
    print(f"Translation: {translation}")

if __name__ == "__main__":
    main(sys.argv[1])