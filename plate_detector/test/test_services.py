import pytest
import numpy as np
import cv2
from unittest.mock import patch
from plate_detector.services import (
    process_plate_image,
    preprocess_plate_image,
    run_ocr,
    detect_plate_yolo,
    run_yolo
)


def test_process_plate_image_success(tmp_path):
    # Crear imagen dummy temporal
    img_path = tmp_path / "test.jpg"
    dummy = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), dummy)

    result = process_plate_image(str(img_path))

    assert result["success"] is True
    assert result["plate"] == "TEST123"
    assert "confidence" in result


def test_process_plate_image_not_found():
    result = process_plate_image("archivo_que_no_existe.jpg")
    assert result["success"] is False
    assert "error" in result


def test_preprocess_plate_image():
    dummy = np.zeros((50, 100, 3), dtype=np.uint8)
    output = preprocess_plate_image(dummy)
    assert output is dummy


def test_run_ocr():
    dummy = np.zeros((50, 100, 3), dtype=np.uint8)
    text = run_ocr(dummy)
    assert text == "TEST123"


def test_detect_plate_yolo():
    dummy = np.zeros((200, 200, 3), dtype=np.uint8)
    success, crop = detect_plate_yolo(dummy)
    assert success is True
    assert isinstance(crop, np.ndarray)


def test_run_yolo():
    dummy = np.zeros((50, 100, 3), dtype=np.uint8)
    success, output = run_yolo(dummy)
    assert success is True
    assert isinstance(output, np.ndarray)


@patch("plate_detector.services.run_ocr", return_value="MOCKED123")
@patch("plate_detector.services.run_yolo", return_value=(True, np.zeros((50,100,3))))
def test_process_plate_image_mocked(mock_yolo, mock_ocr, tmp_path):
    img_path = tmp_path / "test2.jpg"
    cv2.imwrite(str(img_path), np.zeros((100,200,3)))

    result = process_plate_image(str(img_path))

    assert result["success"] is True
    assert result["plate"] == "MOCKED123"
    mock_yolo.assert_called_once()
    mock_ocr.assert_called_once()
