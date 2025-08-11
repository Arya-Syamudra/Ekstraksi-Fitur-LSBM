# tests/test_compare_with_python_ref.py
import sys
from pathlib import Path
# ensure repo root is in sys.path so test can import local modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import cv2
import pytest
import lsbm_features as lf
from lsbm_features.lsbm_features_python_ref import features_54_python, features_all_python

@pytest.mark.parametrize("img_path", [
    "tests/sample1.bmp",
    "tests/sample2.png"
])
def test_features_54_match(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    assert img is not None, f"Cannot load {img_path}"
    py_feats = features_54_python(img)
    assert py_feats.shape == (54,)

@pytest.mark.parametrize("img_path", [
    "tests/sample1.bmp",
    "tests/sample2.png"
])
def test_features_all_match(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    assert img is not None, f"Cannot load {img_path}"
    py_feats = features_all_python(img)
    assert py_feats.shape == (135,)
