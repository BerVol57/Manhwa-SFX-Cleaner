import sys
import cv2
import numpy as np
from clean_pipeline import CLEANER
from segmentation import SEGmodel
from inpainting import INPmodel

def main(img_path):
    # Модель YOLO з вашого шляху
    seg = SEGmodel("../research/runs/segment/sfx_comparison/train_y11n/weights/best.pt")
    inp = INPmodel()
    inp.make_pipe()
    cleaner = CLEANER(seg, inp)
    
    img = cv2.imread(img_path)
    if img is None: raise ValueError("Image not found")
    
    # Використовуємо ваш існуючий метод clean_img
    cleaned, mask, num = cleaner.clean_img(img)
    
    cleaned = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
    cv2.imwrite("temp_cache/cleaned.png", cleaned)
    if mask is not None:
        cv2.imwrite("temp_cache/mask.png", mask)
    
    print(f"Detected {num} characters.")

if __name__ == "__main__":
    main(sys.argv[1])