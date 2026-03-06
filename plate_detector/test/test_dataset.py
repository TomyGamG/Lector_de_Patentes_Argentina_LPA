import os
import re
import time
from pathlib import Path

import pytest


PLATE_PATTERNS = [
    r"^[A-Z]{3}\d{3}$",         # ABC123
    r"^[A-Z]{2}\d{3}[A-Z]{2}$", # AB123CD
]


def _dataset_dir() -> Path:
    PLATE_DATASET_DIR = "autos"
    p = os.environ.get("PLATES_DATASET_DIR")
    assert p, "Definí PLATES_DATASET_DIR con la ruta a tu carpeta de imágenes."

    path = Path(p)
    assert path.exists(), f"No existe la carpeta: {path}"
    assert path.is_dir(), f"La ruta no es una carpeta: {path}"
    return path


def _iter_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    yield from sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])


def normalize_plate(s: str) -> str:
    if not s:
        return ""
    s = s.upper().strip()
    return re.sub(r"[^A-Z0-9]", "", s)


def is_valid_plate_format(text: str) -> bool:
    t = normalize_plate(text)
    return any(re.match(p, t) for p in PLATE_PATTERNS)


def expected_is_plate_like(filename_stem: str) -> bool:
    return is_valid_plate_format(filename_stem)


def pick_best_plate(result: dict):
    if not result:
        return "", 0.0, {}

    plates = result.get("license_plates") or []
    if not plates:
        return "", 0.0, {}

    best = plates[0] or {}
    text = normalize_plate(best.get("text", ""))
    conf = float(best.get("confidence") or 0.0)
    return text, conf, best


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


@pytest.mark.slow
def test_plate_dataset_from_folder():
    images_dir = _dataset_dir()
    images = list(_iter_images(images_dir))
    assert images, "No encontré imágenes en la carpeta del dataset."

    from plate_detector.services import ImageAnalyzer

    use_yolo = os.environ.get("TEST_USE_YOLO", "1") == "1"
    yolo_model = os.environ.get("TEST_YOLO_MODEL", "yolov8n.pt")

    analyzer = ImageAnalyzer(
        use_yolo=use_yolo,
        yolo_model=yolo_model,
    )

    plate_like_total = 0
    plate_like_read_ok = 0
    plate_like_hit_ok = 0
    plate_like_valid_format_ok = 0

    non_plate_total = 0
    non_plate_false_reads = 0

    fails = []
    false_positive_examples = []

    total_elapsed = 0.0
    predicted_confidences = []
    successful_confidences = []

    per_image_results = []

    for img in images:
        expected = normalize_plate(img.stem)

        t0 = time.perf_counter()
        result = analyzer.analyze_image(str(img))
        elapsed = time.perf_counter() - t0
        total_elapsed += elapsed

        pred, conf, best = pick_best_plate(result)

        if pred:
            predicted_confidences.append(conf)

        row = {
            "file": img.name,
            "expected": expected,
            "predicted": pred,
            "confidence": round(_safe_float(conf), 4),
            "elapsed_sec": round(elapsed, 4),
            "method": best.get("method"),
            "ocr_confidence": _safe_float(best.get("ocr_confidence")),
            "detector_confidence": _safe_float(best.get("detector_confidence")),
            "prefilter_text": best.get("prefilter_text"),
            "valid_format": bool(best.get("valid_format", False)),
            "type": "plate_like" if expected_is_plate_like(expected) else "non_plate",
            "hit": False,
        }

        if expected_is_plate_like(expected):
            plate_like_total += 1

            if pred:
                plate_like_read_ok += 1
                successful_confidences.append(conf)

            if is_valid_plate_format(pred):
                plate_like_valid_format_ok += 1

            if pred == expected:
                plate_like_hit_ok += 1
                row["hit"] = True
            else:
                fails.append(
                    {
                        "file": img.name,
                        "expected": expected,
                        "predicted": pred,
                        "confidence": round(_safe_float(conf), 4),
                        "elapsed_sec": round(elapsed, 4),
                        "method": best.get("method"),
                        "ocr_confidence": _safe_float(best.get("ocr_confidence")),
                        "detector_confidence": _safe_float(best.get("detector_confidence")),
                        "prefilter_text": best.get("prefilter_text"),
                        "valid_format": best.get("valid_format"),
                    }
                )
        else:
            non_plate_total += 1
            if pred:
                non_plate_false_reads += 1
                false_positive_examples.append(
                    {
                        "file": img.name,
                        "expected": expected,
                        "predicted": pred,
                        "confidence": round(_safe_float(conf), 4),
                        "elapsed_sec": round(elapsed, 4),
                        "method": best.get("method"),
                        "prefilter_text": best.get("prefilter_text"),
                    }
                )

        per_image_results.append(row)

    assert plate_like_total > 0, (
        "No encontré imágenes con nombre de patente válido. "
        "Renombrá los archivos del dataset con el formato esperado."
    )

    total_images = len(images)
    read_rate = plate_like_read_ok / plate_like_total
    hit_rate = plate_like_hit_ok / plate_like_total
    valid_format_rate = plate_like_valid_format_ok / plate_like_total
    false_positive_rate = (
        non_plate_false_reads / non_plate_total if non_plate_total else 0.0
    )

    avg_conf_all_predictions = (
        sum(predicted_confidences) / len(predicted_confidences)
        if predicted_confidences else 0.0
    )
    avg_conf_successful_reads = (
        sum(successful_confidences) / len(successful_confidences)
        if successful_confidences else 0.0
    )
    avg_time_per_image = total_elapsed / total_images if total_images else 0.0

    min_read_rate = float(os.environ.get("MIN_READ_RATE", "0.55"))
    min_hit_rate = float(os.environ.get("MIN_HIT_RATE", "0.35"))
    min_valid_format_rate = float(os.environ.get("MIN_VALID_FORMAT_RATE", "0.50"))
    max_false_positive_rate = float(os.environ.get("MAX_FALSE_POSITIVE_RATE", "0.60"))

    debug_examples = fails[:10]
    fp_examples = false_positive_examples[:10]

    print("\n" + "=" * 72)
    print("BENCHMARK DATASET - LECTURA DE PATENTES")
    print("=" * 72)

    print(f"Total imágenes:               {total_images}")
    print(f"Patentes esperadas:          {plate_like_total}")
    print(f"No-patentes:                 {non_plate_total}")
    print("-" * 72)
    print(f"Read rate:                   {read_rate:.2%} ({plate_like_read_ok}/{plate_like_total})")
    print(f"Hit rate:                    {hit_rate:.2%} ({plate_like_hit_ok}/{plate_like_total})")
    print(f"Valid format rate:           {valid_format_rate:.2%} ({plate_like_valid_format_ok}/{plate_like_total})")
    print(f"False positive rate:         {false_positive_rate:.2%} ({non_plate_false_reads}/{non_plate_total if non_plate_total else 0})")
    print("-" * 72)
    print(f"Confianza promedio (pred):   {avg_conf_all_predictions:.4f}")
    print(f"Confianza promedio (reads):  {avg_conf_successful_reads:.4f}")
    print(f"Tiempo total:                {total_elapsed:.2f}s")
    print(f"Tiempo promedio/imagen:      {avg_time_per_image:.4f}s")
    print("-" * 72)
    print(f"Umbral read_rate:            {min_read_rate:.0%}")
    print(f"Umbral hit_rate:             {min_hit_rate:.0%}")
    print(f"Umbral valid_format_rate:    {min_valid_format_rate:.0%}")
    print(f"Máx false_positive_rate:     {max_false_positive_rate:.0%}")

    if debug_examples:
        print("\nERRORES PRINCIPALES")
        print("-" * 72)
        for e in debug_examples:
            print(
                f"{e['file']:<18} "
                f"exp={e['expected']:<8} "
                f"pred={e['predicted']:<8} "
                f"conf={e['confidence']:<7} "
                f"time={e['elapsed_sec']:<7} "
                f"method={str(e['method']):<10} "
                f"valid={str(e['valid_format'])}"
            )

    if fp_examples:
        print("\nFALSOS POSITIVOS")
        print("-" * 72)
        for e in fp_examples:
            print(
                f"{e['file']:<18} "
                f"pred={e['predicted']:<8} "
                f"conf={e['confidence']:<7} "
                f"time={e['elapsed_sec']:<7} "
                f"method={str(e['method']):<10}"
            )

    print("\nTOP 10 RESULTADOS POR IMAGEN")
    print("-" * 72)
    for r in per_image_results[:10]:
        print(
            f"{r['file']:<18} "
            f"type={r['type']:<10} "
            f"exp={r['expected']:<8} "
            f"pred={r['predicted']:<8} "
            f"hit={str(r['hit']):<5} "
            f"conf={r['confidence']:<7} "
            f"time={r['elapsed_sec']:<7}"
        )

    print("=" * 72 + "\n")

    assert read_rate >= min_read_rate, (
        f"Read rate bajo: {read_rate:.2%} (min {min_read_rate:.0%}). "
        f"Leídos: {plate_like_read_ok}/{plate_like_total}. "
        f"Errores ejemplo: {debug_examples}"
    )

    assert valid_format_rate >= min_valid_format_rate, (
        f"Valid format rate bajo: {valid_format_rate:.2%} "
        f"(min {min_valid_format_rate:.0%}). "
        f"Formato válido: {plate_like_valid_format_ok}/{plate_like_total}. "
        f"Errores ejemplo: {debug_examples}"
    )

    assert hit_rate >= min_hit_rate, (
        f"Hit rate bajo: {hit_rate:.2%} (min {min_hit_rate:.0%}). "
        f"Aciertos: {plate_like_hit_ok}/{plate_like_total}. "
        f"Errores ejemplo: {debug_examples}"
    )

    assert false_positive_rate <= max_false_positive_rate, (
        f"False positive rate alto: {false_positive_rate:.2%} "
        f"(max {max_false_positive_rate:.0%}). "
        f"Falsos positivos: {non_plate_false_reads}/{non_plate_total}. "
        f"Ejemplos: {fp_examples}"
    )