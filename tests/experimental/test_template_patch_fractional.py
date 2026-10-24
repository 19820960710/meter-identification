import importlib.util
from pathlib import Path
import unittest
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / 'experiments/template-transfer-20261024/patch_locator_v6.py'


class FractionalPatchTests(unittest.TestCase):
    def setUp(self):
        # Before the correction exists, exercise the existing locator to expose rounding.
        file = FILE if FILE.exists() else FILE.with_name('patch_locator.py')
        spec = importlib.util.spec_from_file_location('fractional_patch_test_impl', file)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def fixture(self):
        return cv2.GaussianBlur(np.random.default_rng(11).integers(30, 210, (320, 320, 3), dtype=np.uint8), (3, 3), .8)

    def test_identity_preserves_fractional_source_anchor(self):
        image = self.fixture()
        point = [160.3, 159.7]
        r = self.m.locate_anchor(image, image.copy(), point, 160)
        self.assertTrue(r['passed'], r)
        self.assertLess(float(np.linalg.norm(np.array(r['target_point'])-point)), 1e-8)

    def test_integer_translation_preserves_fractional_component(self):
        image = self.fixture()
        target = cv2.warpAffine(image, np.array([[1., 0, 7], [0, 1, -5]]), (320, 320))
        p = [160.3, 159.7]
        r = self.m.locate_anchor(image, target, p, 160)
        self.assertTrue(r['passed'], r)
        self.assertLess(float(np.linalg.norm(np.array(r['target_point'])-(np.array(p)+[7, -5]))), 1e-8)
