import importlib.util
from pathlib import Path
import unittest
import cv2
import numpy as np

FILE = Path(__file__).resolve().parents[2] / 'experiments/template-transfer-20261024/patch_locator_v6.py'


class TemplatePatchTests(unittest.TestCase):
    def setUp(self):
        if not FILE.exists():
            self.fail('Image patch correspondence locator is not implemented.')
        spec = importlib.util.spec_from_file_location('patch_locator_test_impl', FILE)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def fixture(self):
        image = np.random.default_rng(2026).integers(30, 210, (320, 320, 3), dtype=np.uint8)
        return cv2.GaussianBlur(image, (3, 3), .8)

    def test_translation_with_brightness_change(self):
        image = self.fixture()
        h = np.array([[1., 0, 7], [0, 1, -5]])
        target = cv2.warpAffine(image, h, (320, 320))
        target = np.clip(target.astype(float)*.8+25, 0, 255).astype(np.uint8)
        result = self.m.locate_anchor(image, target, [160, 160], 160)
        self.assertTrue(result['passed'], result)
        self.assertLess(np.linalg.norm(np.array(result['target_point'])-[167, 155]), 1.5)

    def test_flat_patch_rejected(self):
        image = np.full((320, 320, 3), 200, np.uint8)
        self.assertFalse(self.m.locate_anchor(image, image.copy(), [160, 160], 160)['passed'])

    def test_repeated_stripes_are_ambiguous(self):
        image = np.full((320, 320, 3), 230, np.uint8)
        for x in range(0, 320, 12):
            cv2.line(image, (x, 0), (x, 319), (20, 20, 20), 3)
        self.assertFalse(self.m.locate_anchor(image, image.copy(), [160, 160], 160)['passed'])

    def test_current_pointer_occlusion_rejected(self):
        image = self.fixture()
        valid = np.full((320, 320), 255, np.uint8)
        valid[125:195, 125:195] = 0
        self.assertFalse(self.m.locate_anchor(image, image.copy(), [160, 160], 160,
                                            target_valid=valid)['passed'])

    def test_displacement_outside_search_is_not_forced(self):
        image = self.fixture()
        target = cv2.warpAffine(image, np.array([[1., 0, 60], [0, 1, 0]]), (320, 320))
        self.assertFalse(self.m.locate_anchor(image, target, [160, 160], 160)['passed'])
