import os, random, string
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
import albumentations as A

LAST = ("AA", 900, "AH")  # límite correcto AA900AH
FONT = "font/OCR-B.ttf"
OUT = "dataset"

os.makedirs(OUT, exist_ok=True)

def series_to_num(s):
    return (ord(s[0]) - 65) * 26 + (ord(s[1]) - 65)

def all_new_format():
    letters = string.ascii_uppercase
    limit_prefix = series_to_num(LAST[0])
    limit_number = LAST[1]
    limit_suffix = series_to_num(LAST[2])

    for a in letters:
        for b in letters:
            pref = a + b
            sp = series_to_num(pref)

            if sp > limit_prefix:
                return

            for n in range(0, 1000):

                if sp == limit_prefix and n > limit_number:
                    break

                for c in letters:
                    for d in letters:
                        suf = c + d
                        ss = series_to_num(suf)

                        if sp == limit_prefix and n == limit_number and ss > limit_suffix:
                            break

                        yield f"{pref}{n:03d}{suf}"

def old_format():
    letters = string.ascii_uppercase
    for a in letters:
        for b in letters:
            for c in letters:
                for n in range(0, 1000):
                    yield f"{a}{b}{c}{n:03d}"

# augmentaciones visuales
aug = A.Compose([
    A.MotionBlur(p=0.3),
    A.GaussNoise(p=0.4),
    A.RandomBrightnessContrast(p=0.4),
    A.Sharpen(p=0.2),
    A.Perspective(scale=(0.02, 0.08), p=0.4)
])

# ✔ SOLUCIÓN 1: Fuente autoajustada + texto centrado
def render_plate(text):
    W, H = 360, 160
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # buscar fuente más grande que entre en el área
    size = 120
    while size > 10:
        font = ImageFont.truetype(FONT, size)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        if tw <= W - 40 and th <= H - 40:  # margen de 20 px de cada lado
            break
        size -= 2

    # centrar texto
    x = (W - tw) // 2
    y = (H - th) // 2

    draw.text((x, y), text, font=font, fill=(0, 0, 0))
    return img


def save_augmented(img, name, count=10):
    for i in range(count):
        arr = np.array(img)
        arr = aug(image=arr)["image"]
        Image.fromarray(arr).save(f"{OUT}/{name}_{i}.png")

if __name__ == "__main__":
    all_plates = list(all_new_format()) + list(old_format())
    random.shuffle(all_plates)

    for p in tqdm(all_plates[:5000]):  # cambia 5000 si querés
        img = render_plate(p)
        save_augmented(img, p)
