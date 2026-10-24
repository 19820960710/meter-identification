import importlib.util
from pathlib import Path
import unittest
import math
import copy

FILE = Path(__file__).resolve().parents[2] / 'experiments/template-transfer-20261024/full_anchor_validation.py'


class FullAnchorTests(unittest.TestCase):
    def setUp(self):
        if not FILE.exists():
            self.fail('Complete automatic anchor validation has not been implemented.')
        spec = importlib.util.spec_from_file_location('full_anchor_test_impl', FILE)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def fixture(self):
        ticks = [{'value': v, 'point': [200+100*math.cos(math.radians(225-v/25*270)),
                                      200-100*math.sin(math.radians(225-v/25*270))]} for v in range(0, 26, 5)]
        located = [{'value': t['value'], 'target_point': t['point'][:], 'passed': True,
                    'gray_ncc': .99, 'gradient_ncc': .99, 'peak_margin': .2,
                    'roundtrip_px': 0., 'static_fraction': 1., 'angular_adjustment_degrees': 0.,
                    'reverse': {'passed': True}} for t in ticks]
        return ticks, located

    def test_six_ordered_supported_anchors_pass(self):
        ticks, located = self.fixture()
        result = self.m.validate_full_anchors(ticks, located, [200, 200], [400, 400])
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['ticks'], ticks)

    def test_one_unverified_anchor_rejects_full_scale(self):
        ticks, located = self.fixture(); located[5]['passed'] = False
        self.assertFalse(self.m.validate_full_anchors(ticks, located, [200, 200], [400, 400])['passed'])

    def test_duplicate_points_rejected(self):
        ticks, located = self.fixture();located[2]['target_point'] = located[1]['target_point'][:]
        self.assertFalse(self.m.validate_full_anchors(ticks, located, [200, 200], [400, 400])['passed'])

    def test_mirrored_tick_order_rejected(self):
        ticks, located = self.fixture()
        for r in located:r['target_point'][0] = 400-r['target_point'][0]
        self.assertFalse(self.m.validate_full_anchors(ticks, located, [200, 200], [400, 400])['passed'])

    def test_inconsistent_score_cannot_be_hidden_by_passed_flag(self):
        ticks, located = self.fixture();located[1]['gradient_ncc'] = .2
        self.assertFalse(self.m.validate_full_anchors(ticks, located, [200, 200], [400, 400])['passed'])

    def test_non_image_axis_rejected(self):
        ticks, located = self.fixture()
        self.assertFalse(self.m.validate_full_anchors(ticks, located, [240, 200], [400, 400])['passed'])
