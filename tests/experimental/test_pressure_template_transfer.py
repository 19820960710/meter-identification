import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / 'experiments/template-transfer-20261024/template_registration.py'


class PressureTemplateTests(unittest.TestCase):
    def setUp(self):
        if not MODULE.exists():
            self.fail('Pressure template transfer module is not implemented yet.')
        spec = importlib.util.spec_from_file_location('pressure_transfer_test_impl', MODULE)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def inputs(self):
        rng = np.random.default_rng(42)
        image = cv2.GaussianBlur(rng.integers(0, 256, (400, 400, 3), dtype=np.uint8), (3, 3), 0)
        raw = cv2.imencode('.png', image)[1].tobytes()
        ticks = [{'value': value, 'point': [200 + 140 * math.cos(math.radians(225 - value / 25 * 270)),
                                          200 - 140 * math.sin(math.radians(225 - value / 25 * 270))]}
                 for value in range(0, 26, 5)]
        profile = {'image': 'reference.png', 'image_sha256': hashlib.sha256(raw).hexdigest(),
                   'image_size': [400, 400], 'ticks': ticks, 'unit': 'MPa',
                   'direction': 'clockwise', 'scale_mode': 'partial', 'minimum_division': None}
        h = np.array([[1.1, .03, 50], [-.02, 1., 25], [.00006, -.00004, 1.]])
        target = cv2.warpPerspective(image, h, (640, 480))
        panel = [20, 20, 380, 380]
        corners = self.m.project([[20, 20], [380, 20], [380, 380], [20, 380]], h)
        proposal = [float(corners[:, 0].min()), float(corners[:, 1].min()),
                    float(corners[:, 0].max()), float(corners[:, 1].max())]
        return raw, target, profile, panel, [proposal], h

    def test_known_partial_scale_warp(self):
        raw, target, profile, panel, proposals, h = self.inputs()
        result = self.m.register_partial(raw, target, profile, panel, proposals)
        self.assertEqual(result['status'], 'registration_candidate', result['reason'])
        actual = self.m.project([t['point'] for t in profile['ticks']], np.array(result['homography']))
        expected = self.m.project([t['point'] for t in profile['ticks']], h)
        self.assertLess(float(np.linalg.norm(actual - expected, axis=1).max()), 1.5)
        self.assertIsNone(result['numeric_value'])

    def test_wrong_reference_sha_rejected(self):
        args = list(self.inputs())
        args[2] = {**args[2], 'image_sha256': '0' * 64}
        self.assertEqual(self.m.register_partial(*args[:5])['status'], 'needs_review')

    def test_unrelated_target_rejected(self):
        raw, target, profile, panel, proposals, _ = self.inputs()
        unrelated = np.random.default_rng(2026).integers(0, 256, target.shape, dtype=np.uint8)
        self.assertEqual(self.m.register_partial(raw, unrelated, profile, panel, proposals)['status'], 'needs_review')

    def test_pointer_roi_is_not_used_for_matching(self):
        baseline = self.m.static_mask([400, 400], [200, 200], 140, [20, 20, 380, 380], [])
        masked = self.m.static_mask([400, 400], [200, 200], 140, [20, 20, 380, 380], [[60, 180, 100, 220]])
        self.assertEqual(int(baseline[200, 70]), 255)
        self.assertEqual(int(masked[200, 70]), 0)
        self.assertEqual(int(masked[60, 200]), 255)

    def test_failed_registration_cannot_make_runtime_profile(self):
        raw, target, profile, *_ = self.inputs()
        with self.assertRaises(ValueError):
            self.m.rectify_for_reading(raw, target, profile, {'status': 'needs_review'}, 'warped.png')

    def test_runtime_profile_uses_current_warped_image_and_keeps_source_ticks(self):
        raw, target, profile, panel, proposals, h = self.inputs()
        original = copy.deepcopy(profile)
        registration = self.m.register_partial(raw, target, profile, panel, proposals)
        warped_raw, runtime = self.m.rectify_for_reading(raw, target, profile, registration, 'warped.png')
        self.assertEqual(runtime['image_sha256'], hashlib.sha256(warped_raw).hexdigest())
        self.assertEqual(runtime['template_image_sha256'], profile['image_sha256'])
        self.assertEqual(runtime['ticks'], profile['ticks'])
        self.assertEqual(profile, original)
        warped = cv2.imdecode(np.frombuffer(warped_raw, np.uint8), 1)
        self.assertEqual(warped.shape[:2], (400, 400))

    def test_rectification_keeps_changed_current_target_content(self):
        raw, target, profile, panel, proposals, h = self.inputs()
        xy = self.m.project([[200, 200]], h)[0]
        cv2.rectangle(target, tuple(np.rint(xy - 16).astype(int)),
                      tuple(np.rint(xy + 16).astype(int)), (0, 0, 255), -1)
        registration = self.m.register_partial(raw, target, profile, panel, proposals)
        self.assertEqual(registration['status'], 'registration_candidate', registration['reason'])
        warped_raw, runtime = self.m.rectify_for_reading(raw, target, profile, registration, 'current.png')
        warped = cv2.imdecode(np.frombuffer(warped_raw, np.uint8), 1)
        self.assertGreater(int(warped[200, 200, 2]), 240)
        self.assertLess(int(warped[200, 200, 0]), 15)

    def test_registration_cannot_be_reused_for_a_different_current_frame(self):
        raw, target, profile, panel, proposals, h = self.inputs()
        registration = self.m.register_partial(raw, target, profile, panel, proposals)
        target[0, 0] = [1, 2, 3]
        with self.assertRaises(ValueError):
            self.m.rectify_for_reading(raw, target, profile, registration, 'wrong-current.png')


if __name__ == '__main__':
    unittest.main()
