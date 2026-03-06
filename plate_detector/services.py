"""
services.py
Pipeline robusto para detección y lectura de patentes argentinas.

OBJETIVOS:
- Mejorar precisión real del OCR combinando:
    1) Detección de candidato (YOLO + fallback por contornos)
    2) Prefiltro OCR rápido
    3) Rectificación / warp
    4) Multi-preprocesado
    5) OCR múltiple
    6) Normalización + validación estricta de formato
    7) Ranking final por score

- Mantener API simple:
    ImageAnalyzer.analyze_image(path) -> dict
    process_plate_image(path) -> (plate, confidence)

- Evitar errores JSON:
    NO devolvemos numpy arrays dentro del resultado final.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from logging.handlers import RotatingFileHandler

import os
import re
import time
import logging

import cv2
import numpy as np


# ============================================================
# LOGGER
# ============================================================

def _setup_logger() -> logging.Logger:
    log_name = "plate_detector"
    log = logging.getLogger(log_name)

    if log.handlers:
        return log

    level_str = os.environ.get("PLATE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    log.setLevel(level)
    log.propagate = False

    log_dir = Path(os.environ.get("PLATE_LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / os.environ.get("PLATE_LOG_FILE", "plate_detector.log")

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    fh = RotatingFileHandler(
        str(log_file),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)

    log.addHandler(fh)
    log.addHandler(ch)
    log.info("Logger inicializado | file=%s | level=%s", str(log_file), level_str)
    return log


logger = _setup_logger()


# ============================================================
# CONFIG / PATRONES
# ============================================================

PLATE_PATTERNS = [
    r"^[A-Z]{3}\d{3}$",        # ABC123
    r"^[A-Z]{2}\d{3}[A-Z]{2}$" # AB123CD
]

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Correcciones OCR comunes.
# OJO: no conviene mapear ciegamente en todos los casos.
OCR_DIGIT_LIKE = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
}

OCR_LETTER_LIKE = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}


# ============================================================
# TEXTO / NORMALIZACIÓN / VALIDACIÓN
# ============================================================

def clean_ocr_text(text: str) -> str:
    if not text:
        return ""
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def has_letters(text: str) -> bool:
    return bool(re.search(r"[A-Z]", text or ""))


def has_digits(text: str) -> bool:
    return bool(re.search(r"\d", text or ""))


def is_valid_plate_format(text: str) -> bool:
    t = clean_ocr_text(text)
    if not t:
        return False

    patterns = [
        r"^[A-Z]{3}\d{3}$",        # ABC123
        r"^[A-Z]{2}\d{3}[A-Z]{2}$" # AB123CD
    ]
    return any(re.match(p, t) for p in patterns)


def _apply_pattern_mapping(text: str, pattern_name: str) -> str:
    """
    Corrige ambiguedades según el formato esperado.

    old: ABC123     -> 3 letras + 3 dígitos
    new: AB123CD    -> 2 letras + 3 dígitos + 2 letras
    """
    t = clean_ocr_text(text)
    if not t:
        return ""

    chars = list(t)

    if pattern_name == "old_abc123":
        expected = ["L", "L", "L", "D", "D", "D"]
    elif pattern_name == "new_ab123cd":
        expected = ["L", "L", "D", "D", "D", "L", "L"]
    else:
        return t

    if len(chars) != len(expected):
        return t

    fixed = []
    for ch, kind in zip(chars, expected):
        if kind == "D" and ch.isalpha():
            fixed.append(OCR_DIGIT_LIKE.get(ch, ch))
        elif kind == "L" and ch.isdigit():
            fixed.append(OCR_LETTER_LIKE.get(ch, ch))
        else:
            fixed.append(ch)

    return "".join(fixed)


def normalize_plate_candidates(text: str) -> List[str]:
    """
    Genera variantes corregidas de una lectura OCR.
    """
    t = clean_ocr_text(text)
    if not t:
        return []

    variants = {t}

    # Variantes por formato conocido
    variants.add(_apply_pattern_mapping(t, "old_abc123"))
    variants.add(_apply_pattern_mapping(t, "new_ab123cd"))

    # Si hay espacios / basura, ya fueron limpiados.
    # Filtrar vacíos.
    variants = {v for v in variants if v}

    # Priorizar válidas
    ordered = sorted(
        variants,
        key=lambda x: (
            0 if is_valid_plate_format(x) else 1,
            abs(len(x) - 6),
            abs(len(x) - 7),
        )
    )
    return ordered


def score_plate_text(text: str, ocr_conf: float) -> float:
    """
    Score final de texto basado en:
    - formato válido
    - confianza OCR
    """
    t = clean_ocr_text(text)
    if not t:
        return 0.0

    if not is_valid_plate_format(t):
        return 0.0

    score = float(ocr_conf)

    if re.match(r"^[A-Z]{3}\d{3}$", t):
        score += 0.12

    if re.match(r"^[A-Z]{2}\d{3}[A-Z]{2}$", t):
        score += 0.12

    return min(score, 1.0)

# ============================================================
# GEOMETRÍA
# ============================================================

def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def _resize_keep_ratio(img: np.ndarray, target_width: int = 320) -> np.ndarray:
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]
    if w == 0:
        return img

    scale = target_width / float(w)
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_CUBIC)


def warp_plate(roi_bgr: np.ndarray, out_size: Tuple[int, int] = (320, 100)) -> np.ndarray:
    """
    Busca contorno cuadrilateral dentro del ROI y hace perspective warp.
    Si no puede, devuelve resize.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return roi_bgr

    target_w, target_h = out_size

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return cv2.resize(roi_bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:15]
    quad = None

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        if h <= 0:
            continue

        ar = w / float(h)
        if 2.0 <= ar <= 6.5:
            quad = approx.reshape(4, 2).astype("float32")
            break

    if quad is None:
        return cv2.resize(roi_bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    rect = _order_points(quad)
    dst = np.array(
        [
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(roi_bgr, M, (target_w, target_h))
    return warped


# ============================================================
# PREPROCESADO
# ============================================================

def _gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _sharpen(gray: np.ndarray) -> np.ndarray:
    kernel = np.array([
        [0, -1,  0],
        [-1, 5, -1],
        [0, -1,  0]
    ], dtype=np.float32)
    return cv2.filter2D(gray, -1, kernel)


def _adaptive_thresh(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )


def _otsu_thresh(gray: np.ndarray) -> np.ndarray:
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thr


def build_ocr_variants(roi_bgr: np.ndarray) -> List[np.ndarray]:
    """
    Genera varias versiones de la patente para OCR.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return []

    roi = _resize_keep_ratio(roi_bgr, target_width=360)
    gray = _gray(roi)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    c = _clahe(gray)
    s = _sharpen(c)

    variants = []

    # escala de grises mejorada
    variants.append(c)
    variants.append(s)

    # umbralizaciones
    variants.append(_adaptive_thresh(c))
    variants.append(255 - _adaptive_thresh(c))
    variants.append(_otsu_thresh(c))
    variants.append(255 - _otsu_thresh(c))
    variants.append(_otsu_thresh(s))
    variants.append(255 - _otsu_thresh(s))

    # abrir/cerrar para limpiar
    kernel = np.ones((2, 2), np.uint8)
    for img in variants[:]:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        variants.append(opened)
        variants.append(closed)

    # quitar duplicados por forma + bytes
    unique = []
    seen = set()
    for v in variants:
        key = (v.shape, v.dtype.str, v.tobytes()[:200])
        if key in seen:
            continue
        seen.add(key)
        unique.append(v)

    return unique[:12]


# ============================================================
# OCR
# ============================================================

_EASYOCR_READER = None


def _get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is not None:
        return _EASYOCR_READER

    try:
        import easyocr
    except Exception as e:
        logger.warning("EasyOCR no disponible: %s", e)
        _EASYOCR_READER = None
        return None

    _EASYOCR_READER = easyocr.Reader(["en"], gpu=False)
    return _EASYOCR_READER


def _to_easyocr_input(img: np.ndarray) -> np.ndarray:
    """
    EasyOCR acepta mejor RGB o grayscale bien definidos.
    """
    if img is None or img.size == 0:
        return img

    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _easyocr_read_candidates(img: np.ndarray) -> List[Tuple[str, float]]:
    reader = _get_easyocr_reader()
    if reader is None or img is None or img.size == 0:
        return []

    try:
        results = reader.readtext(
            _to_easyocr_input(img),
            detail=1,
            allowlist=ALLOWLIST,
            paragraph=False,
            decoder="greedy",
            contrast_ths=0.05,
            adjust_contrast=0.7,
            text_threshold=0.5,
            low_text=0.3,
            link_threshold=0.2,
        )
    except Exception as e:
        logger.debug("EasyOCR falló en lectura: %s", e)
        return []

    out = []
    for _, txt, conf in results:
        txt = clean_ocr_text(txt)
        if not txt:
            continue
        out.append((txt, float(conf)))

    return out


def ocr_plate_best(roi_bgr: np.ndarray) -> Tuple[str, float]:
    """
    Hace varias lecturas OCR sobre varias transformaciones
    y elige la mejor por score.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return "", 0.0

    variants = build_ocr_variants(roi_bgr)
    candidates: List[Tuple[str, float, float]] = []

    for img in variants:
        reads = _easyocr_read_candidates(img)
        for txt, conf in reads:
            for norm in normalize_plate_candidates(txt):
                if not is_valid_plate_format(norm):
                    continue
                score = score_plate_text(norm, conf)
                candidates.append((norm, conf, score))

    if not candidates:
        return "", 0.0

    # deduplicar por texto, quedarnos con el mejor score
    best_by_text: Dict[str, Tuple[float, float]] = {}
    for txt, conf, score in candidates:
        prev = best_by_text.get(txt)
        if prev is None or score > prev[1]:
            best_by_text[txt] = (conf, score)

    ranked = sorted(
        best_by_text.items(),
        key=lambda kv: (
            0 if is_valid_plate_format(kv[0]) else 1,
            -kv[1][1],
            -kv[1][0],
        )
    )

    best_text = ranked[0][0]
    best_conf = ranked[0][1][1]
    return best_text, float(best_conf)


def quick_prefilter_ocr(roi_bgr: np.ndarray) -> Tuple[bool, float, str]:
    """
    OCR rápido para descartar candidatos absurdos.
    Solo deja pasar si encuentra una lectura compatible
    con formato argentino estricto:
    - ABC123
    - AB123CD
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return False, 0.0, ""

    small = _resize_keep_ratio(roi_bgr, target_width=240)
    gray = _gray(small)
    gray = _clahe(gray)

    reads = _easyocr_read_candidates(gray)
    if not reads:
        return False, 0.0, ""

    best_text = ""
    best_conf = 0.0

    for txt, conf in reads:
        txt = clean_ocr_text(txt)

        for norm in normalize_plate_candidates(txt):
            if is_valid_plate_format(norm) and float(conf) > best_conf:
                best_text = norm
                best_conf = float(conf)

    if not best_text:
        return False, 0.0, ""

    return True, best_conf, best_text


# ============================================================
# DETECCIÓN DE CANDIDATOS
# ============================================================

@dataclass
class PlateCandidate:
    bbox: Tuple[int, int, int, int]   # x, y, w, h
    confidence: float
    method: str
    region: np.ndarray


def _clip_bbox(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h


def _expand_bbox(x: int, y: int, w: int, h: int, W: int, H: int, pad_x: float = 0.10, pad_y: float = 0.18):
    px = int(w * pad_x)
    py = int(h * pad_y)
    nx = max(0, x - px)
    ny = max(0, y - py)
    nw = min(W - nx, w + 2 * px)
    nh = min(H - ny, h + 2 * py)
    return nx, ny, nw, nh


def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union

def _merge_similar_plate_reads(plates: List[Dict[str, Any]], iou_threshold: float = 0.45) -> List[Dict[str, Any]]:
    """
    Fusiona lecturas que pertenecen a la misma patente física
    usando overlap de bounding boxes.

    Regla:
    - Si dos detecciones se superponen bastante, se consideran la misma chapa.
    - Se conserva la mejor según:
        1) formato válido
        2) mayor confidence total
        3) mayor ocr_confidence
    """
    if not plates:
        return []

    def _plate_bbox(p: Dict[str, Any]) -> Tuple[int, int, int, int]:
        c = p.get("coordinates") or {}
        return (
            int(c.get("x", 0)),
            int(c.get("y", 0)),
            int(c.get("w", 0)),
            int(c.get("h", 0)),
        )

    def _is_better(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        a_valid = bool(a.get("valid_format", False))
        b_valid = bool(b.get("valid_format", False))

        if a_valid != b_valid:
            return a_valid

        a_method = str(a.get("method", "") or "")
        b_method = str(b.get("method", "") or "")

        a_conf = float(a.get("confidence", 0.0) or 0.0)
        b_conf = float(b.get("confidence", 0.0) or 0.0)

        a_ocr = float(a.get("ocr_confidence", 0.0) or 0.0)
        b_ocr = float(b.get("ocr_confidence", 0.0) or 0.0)

        # preferir mejor OCR aunque confidence total sea similar
        if abs(a_conf - b_conf) < 0.08 and a_ocr != b_ocr:
            return a_ocr > b_ocr

        # si están muy parejos, preferir contours sobre yolo
        # porque a veces contours recorta más justo la chapa
        if abs(a_conf - b_conf) < 0.05 and a_method != b_method:
            if a_method == "contours" and b_method == "yolo":
                return True
            if a_method == "yolo" and b_method == "contours":
                return False

        if a_conf != b_conf:
            return a_conf > b_conf

        if a_ocr != b_ocr:
            return a_ocr > b_ocr

        a_text = str(a.get("text", "") or "")
        b_text = str(b.get("text", "") or "")
        return len(a_text) <= len(b_text)

    merged: List[Dict[str, Any]] = []

    for p in plates:
        pbox = _plate_bbox(p)
        matched_idx = None

        for i, existing in enumerate(merged):
            ebox = _plate_bbox(existing)
            if _iou(pbox, ebox) >= iou_threshold:
                matched_idx = i
                break

        if matched_idx is None:
            merged.append(p)
        else:
            if _is_better(p, merged[matched_idx]):
                merged[matched_idx] = p

    return merged

def _dedup_candidates(candidates: List[PlateCandidate], iou_threshold: float = 0.50) -> List[PlateCandidate]:
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda c: c.confidence, reverse=True)
    selected: List[PlateCandidate] = []

    for cand in candidates:
        if all(_iou(cand.bbox, s.bbox) < iou_threshold for s in selected):
            selected.append(cand)

    return selected


def _scan_by_contours(image_bgr: np.ndarray) -> List[PlateCandidate]:
    """
    Fallback cuando no hay YOLO o YOLO no detecta.
    """
    H, W = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_x = cv2.convertScaleAbs(grad_x)

    _, thr = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    morph = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []

    candidates: List[PlateCandidate] = []

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)

        if w < 60 or h < 20:
            continue

        ar = w / float(h)
        area = w * h

        if not (2.0 <= ar <= 6.5):
            continue

        if area < 1400:
            continue

        x, y, w, h = _expand_bbox(x, y, w, h, W, H)
        x, y, w, h = _clip_bbox(x, y, w, h, W, H)

        roi = image_bgr[y:y + h, x:x + w].copy()

        conf = min(0.82, 0.30 + (area / float(W * H)) * 10.0)
        candidates.append(PlateCandidate((x, y, w, h), float(conf), "contours", roi))

    return _dedup_candidates(candidates)[:20]


def _scan_with_yolo(image_bgr: np.ndarray, yolo_model: str) -> List[PlateCandidate]:
    try:
        from ultralytics import YOLO
    except Exception as e:
        logger.info("Ultralytics no disponible: %s", e)
        return []

    H, W = image_bgr.shape[:2]

    try:
        model = YOLO(yolo_model)
    except Exception as e:
        logger.warning("No pude cargar YOLO(%s): %s", yolo_model, e)
        return []

    try:
        results = model.predict(source=image_bgr, verbose=False)
    except Exception as e:
        logger.warning("YOLO predict falló: %s", e)
        return []

    candidates: List[PlateCandidate] = []

    for r in results:
        if r.boxes is None:
            continue

        for b in r.boxes:
            xyxy = b.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy.tolist()

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W - 1, x2)
            y2 = min(H - 1, y2)

            w = max(1, x2 - x1)
            h = max(1, y2 - y1)

            if w < 40 or h < 15:
                continue

            x, y, w, h = _expand_bbox(x1, y1, w, h, W, H, pad_x=0.10, pad_y=0.20)
            roi = image_bgr[y:y + h, x:x + w].copy()

            conf = float(b.conf[0].cpu().numpy()) if getattr(b, "conf", None) is not None else 0.5
            candidates.append(PlateCandidate((x, y, w, h), conf, "yolo", roi))

    return _dedup_candidates(candidates)[:20]


# ============================================================
# PIPELINE
# ============================================================

class PlatePipeline:
    def __init__(
        self,
        *,
        use_yolo: bool = True,
        yolo_model: str = "yolov8n.pt",
        processing_interval_sec: float = 0.0,
        prefilter_min_conf: float = 0.12,
        min_final_score: float = 0.45,
    ):
        self.use_yolo = use_yolo
        self.yolo_model = yolo_model
        self.processing_interval_sec = processing_interval_sec
        self.prefilter_min_conf = prefilter_min_conf
        self.min_final_score = min_final_score
        self._last_ts = 0.0

    def scan_candidates(self, image_bgr: np.ndarray) -> List[PlateCandidate]:
        candidates: List[PlateCandidate] = []

        if self.use_yolo:
            yolo_candidates = _scan_with_yolo(image_bgr, self.yolo_model)
            candidates.extend(yolo_candidates)

        contour_candidates = _scan_by_contours(image_bgr)
        candidates.extend(contour_candidates)

        candidates = _dedup_candidates(candidates)
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates[:25]

    def candidate_has_letters_and_digits(self, roi_bgr: np.ndarray) -> Tuple[bool, float, str]:
        valid, conf, txt = quick_prefilter_ocr(roi_bgr)

        if conf < self.prefilter_min_conf:
            return False, conf, txt

        return valid, conf, txt

    def recognize_plate(self, roi_bgr: np.ndarray) -> Tuple[str, float]:
        """
        OCR final:
        1) ROI directo
        2) ROI warpeado
        Se queda con el mejor score.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return "", 0.0

        warped = warp_plate(roi_bgr)

        t1, c1 = ocr_plate_best(roi_bgr)
        t2, c2 = ocr_plate_best(warped)

        candidates = [(t1, c1), (t2, c2)]
        candidates = [(t, c) for t, c in candidates if t]

        if not candidates:
            return "", 0.0

        candidates.sort(
            key=lambda x: (
                0 if is_valid_plate_format(x[0]) else 1,
                -x[1],
            )
        )

        best_text, best_conf = candidates[0]

        if not best_text or not is_valid_plate_format(best_text):
            return "", 0.0

        if best_conf < self.min_final_score:
            return "", 0.0

        return best_text, float(best_conf)

    def process_image(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        now = time.time()
        if self.processing_interval_sec and (now - self._last_ts) < self.processing_interval_sec:
            return {
                "image_info": {},
                "total_candidates": 0,
                "license_plates": [],
                "skipped": True,
            }

        self._last_ts = now

        H, W = image_bgr.shape[:2]
        candidates = self.scan_candidates(image_bgr)

        logger.info("process_image | candidates=%d", len(candidates))

        plates: List[Dict[str, Any]] = []
        discarded_prefilter = 0
        discarded_invalid = 0

        for cand in candidates:
            ok_pref, pre_conf, pre_txt = self.candidate_has_letters_and_digits(cand.region)
            if not ok_pref:
                discarded_prefilter += 1
                logger.debug(
                    "DISCARD prefilter | method=%s | pre_conf=%.3f | pre_txt=%s",
                    cand.method,
                    pre_conf,
                    pre_txt,
                )
                continue

            plate_text, ocr_score = self.recognize_plate(cand.region)
            if not plate_text or not is_valid_plate_format(plate_text):
                discarded_invalid += 1
                logger.debug(
                    "DISCARD invalid OCR | method=%s | pre_txt=%s",
                    cand.method,
                    pre_txt,
                )
                continue

            # Score combinado detector + OCR final
            detector_weight = 0.35
            ocr_weight = 0.65
            combined = (cand.confidence * detector_weight) + (ocr_score * ocr_weight)

            plates.append({
                "text": plate_text,
                "confidence": round(float(combined), 4),
                "detector_confidence": round(float(cand.confidence), 4),
                "ocr_confidence": round(float(ocr_score), 4),
                "prefilter_confidence": round(float(pre_conf), 4),
                "prefilter_text": pre_txt,
                "coordinates": {
                    "x": int(cand.bbox[0]),
                    "y": int(cand.bbox[1]),
                    "w": int(cand.bbox[2]),
                    "h": int(cand.bbox[3]),
                },
                "method": cand.method,
                "valid_format": is_valid_plate_format(plate_text),
            })

        # 1) dedup por texto exacto, conservando mayor confidence
        best_by_text: Dict[str, Dict[str, Any]] = {}
        for p in plates:
            t = str(p.get("text", "") or "")
            if not t:
                continue
            prev = best_by_text.get(t)
            if prev is None or float(p.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
                best_by_text[t] = p

        plates = list(best_by_text.values())

        # 2) fusionar lecturas distintas sobre la misma patente física
        plates = _merge_similar_plate_reads(plates, iou_threshold=0.45)

        # 3) ordenar resultado final
        plates.sort(
            key=lambda p: (
                0 if p.get("valid_format") else 1,
                -float(p.get("confidence", 0.0) or 0.0),
                -float(p.get("ocr_confidence", 0.0) or 0.0),
            )
        )

        logger.info(
            "process_image SUMMARY | plates=%d | discarded_prefilter=%d | discarded_invalid=%d",
            len(plates),
            discarded_prefilter,
            discarded_invalid,
        )

        return {
            "image_info": {
                "width": int(W),
                "height": int(H),
                "channels": int(image_bgr.shape[2]) if len(image_bgr.shape) > 2 else 1,
            },
            "total_candidates": len(candidates),
            "license_plates": plates,
            "skipped": False,
        }


# ============================================================
# API PRINCIPAL
# ============================================================

class ImageAnalyzer:
    """
    API compatible:
        analyzer = ImageAnalyzer(...)
        result = analyzer.analyze_image("ruta.jpg")
    """

    def __init__(
        self,
        use_yolo: bool = True,
        yolo_model: str = "yolov8n.pt",
    ):
        self.pipeline = PlatePipeline(
            use_yolo=use_yolo,
            yolo_model=yolo_model,
            processing_interval_sec=0.0,
            prefilter_min_conf=0.12,
            min_final_score=0.45,
        )

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        t0 = time.time()
        logger.info("analyze_image START | image=%s", image_path)

        image = cv2.imread(image_path)
        if image is None:
            logger.error("No se pudo cargar la imagen | image=%s", image_path)
            return {
                "error": "No se pudo cargar la imagen",
                "license_plates": [],
            }

        result = self.pipeline.process_image(image)

        logger.info(
            "analyze_image DONE | image=%s | plates=%d | candidates=%s | elapsed=%.2fs",
            image_path,
            len(result.get("license_plates") or []),
            result.get("total_candidates"),
            time.time() - t0,
        )
        return result


# singleton opcional
_analyzer_singleton: Optional[ImageAnalyzer] = None


def process_plate_image(image_path: str) -> Tuple[str, float]:
    """
    Wrapper simple:
        retorna (plate_text, confidence)
    """
    global _analyzer_singleton

    t0 = time.time()
    logger.info("process_plate_image START | image=%s", image_path)

    if _analyzer_singleton is None:
        _analyzer_singleton = ImageAnalyzer(
            use_yolo=True,
            yolo_model="yolov8n.pt",
        )

    result = _analyzer_singleton.analyze_image(image_path) or {}
    plates = result.get("license_plates") or []

    if not plates:
        logger.info(
            "process_plate_image DONE | image=%s | plate= | conf=0.0 | elapsed=%.2fs",
            image_path,
            time.time() - t0,
        )
        return "", 0.0

    best = plates[0]
    plate = str(best.get("text") or "")
    conf = float(best.get("confidence") or 0.0)

    logger.info(
        "process_plate_image DONE | image=%s | plate=%s | conf=%.4f | elapsed=%.2fs",
        image_path,
        plate,
        conf,
        time.time() - t0,
    )
    return plate, conf


# ============================================================
# DJANGO SAVE (OPCIONAL)
# ============================================================

def _django_available() -> bool:
    try:
        import django  # noqa
        return True
    except Exception:
        return False


def save_detection_to_django(
    *,
    image_bgr: np.ndarray,
    plate_data: Dict[str, Any],
    user=None,
    filename_prefix: str = "plate",
):
    """
    Guarda imagen debug + PlateDetection si Django está disponible.
    """
    if not _django_available():
        return None

    try:
        from django.conf import settings
        from .models import PlateDetection, Vehicle  # ajustá si cambia tu app
        try:
            from .models import Client
        except Exception:
            Client = None
    except Exception as e:
        logger.warning("No pude importar settings/models Django: %s", e)
        return None

    try:
        plates_dir = os.path.join(settings.MEDIA_ROOT, "plates")
        os.makedirs(plates_dir, exist_ok=True)

        ts = int(time.time())
        plate_text = str(plate_data.get("text", "")).replace(" ", "")
        filename = f"{filename_prefix}_{plate_text}_{ts}.jpg"
        filepath = os.path.join(plates_dir, filename)

        debug_image = image_bgr.copy()

        coords = plate_data.get("coordinates", {})
        x = int(coords.get("x", 0))
        y = int(coords.get("y", 0))
        w = int(coords.get("w", 0))
        h = int(coords.get("h", 0))

        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            debug_image,
            plate_text,
            (x, max(0, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imwrite(filepath, debug_image)

        vehicle = None
        is_client = False

        if user and getattr(user, "is_authenticated", False) and Client is not None:
            try:
                client = Client.objects.get(user=user)
                vehicle = Vehicle.objects.filter(
                    client=client,
                    plate_number=plate_text
                ).first()
                if vehicle:
                    is_client = True
            except Exception:
                pass

        detection = PlateDetection(
            user=user,
            plate_number=plate_text,
            confidence=float(plate_data.get("confidence", 0.0)),
            image=f"plates/{filename}",
            is_client=is_client,
            vehicle=vehicle,
        )
        detection.save()

        logger.info(
            "Saved PlateDetection | id=%s | plate=%s | is_client=%s",
            getattr(detection, "id", None),
            plate_text,
            is_client,
        )
        return detection

    except Exception as e:
        logger.exception("Error guardando detección Django: %s", e)
        return None


def analyze_and_save(
    image_path: str,
    *,
    user=None,
    use_yolo: bool = True,
    yolo_model: str = "yolov8n.pt",
) -> Dict[str, Any]:
    analyzer = ImageAnalyzer(use_yolo=use_yolo, yolo_model=yolo_model)

    logger.info("analyze_and_save START | image=%s", image_path)
    image = cv2.imread(image_path)

    if image is None:
        logger.error("analyze_and_save ERROR | no se pudo cargar imagen | image=%s", image_path)
        return {
            "error": "No se pudo cargar la imagen",
            "license_plates": [],
        }

    result = analyzer.pipeline.process_image(image)
    plates = result.get("license_plates") or []

    if plates:
        best = plates[0]
        saved = save_detection_to_django(
            image_bgr=image,
            plate_data=best,
            user=user,
        )
        result["saved_detection"] = getattr(saved, "id", None) if saved else None
    else:
        result["saved_detection"] = None

    return result