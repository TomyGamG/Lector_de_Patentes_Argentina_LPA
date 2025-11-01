import os, random, string
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
import albumentations as A

LAST_SERIES = ("AH", 100, "AA")  # límite estimado enero 2025
FONT = "font/OCR-B.ttf"
OUT = "dataset"

os.makedirs(OUT, exist_ok=True)

def all_new_format():
    letters = string.ascii_uppercase
    for a in letters:
        for b in letters:
            # cortar en AH
            if a + b > LAST_SERIES[0]:
                return
            for n in range(0, 1000):
                if a + b == LAST_SERIES[0] and n > LAST_SERIES[1]:
                    break
                for c in letters:
                    for d in letters:
                        if a + b == LAST_SERIES[0] and n == LAST_SERIES[1] and c + d > LAST_SERIES[2]:
                            break
                        yield f"{a}{b}{n:03d}{c}{d}"

def old_format():
    letters = string.ascii_uppercase
    for a in letters:
        for b in letters:
            for c in letters:
                for n in range(0, 1000):
                    yield f"{a}{b}{c}{n:03d}"

# augmentaciones visuales realistas
aug = A.Compose([
    A.MotionBlur(p=0.3),
    A.GaussNoise(p=0.4),
    A.RandomBrightnessContrast(p=0.4),
    A.Sharpen(p=0.2),
    A.Perspective(scale=(0.02, 0.08), p=0.4)
])

def render_plate(text):
    W, H = 360, 160
    img = Image.new("RGB", (W, H), (255,255,255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 95)

    draw.text((20, 20), text, fill=(0,0,0), font=font)
    return img

def save_augmented(img, name, count=10):
    for i in range(count):
        arr = np.array(img)
        arr = aug(image=arr)["image"]
        Image.fromarray(arr).save(f"{OUT}/{name}_{i}.png")

if __name__ == "__main__":
    all_plates = list(all_new_format()) + list(old_format())
    random.shuffle(all_plates)

    for p in tqdm(all_plates[:5000]):  # cambia 5000 → miles que quieras
        img = render_plate(p)
        save_augmented(img, p)
