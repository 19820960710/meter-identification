import importlib.util
from pathlib import Path
import unittest
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / 'experiments/template-transfer-20261024/template_registration_v2.py'


class PressureTemplateV2Tests(unittest.TestCase):
    def setUp(self):
        if not FILE.exists():
            self.fail('Second-version static-mask and ring-geometry implementation is missing.')
        spec = importlib.util.spec_from_file_location('pressure_transfer_v2_tests', FILE)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def motion(self):
        return {'center': [200, 200], 'endpoint': [70, 140], 'cap_radius': 12,
                'panel_radius': 180}

    def test_static_scale_survives_wide_pointer_search_rectangle(self):
        mask = self.m.motion_mask([400, 400], self.motion())
        self.assertEqual(int(mask[200, 200]), 255)
        self.assertEqual(int(mask[140, 70]), 255)
        self.assertEqual(int(mask[200, 60]), 0)

    def test_counterweight_region_is_excluded(self):
        mask = self.m.motion_mask([400, 400], self.motion())
        self.assertEqual(int(mask[235, 275]), 255)

    def test_invalid_motion_cannot_be_used(self):
        motion = self.motion()
        motion['endpoint'] = motion['center'][:]
        with self.assertRaises(ValueError):
            self.m.motion_mask([400, 400], motion)

    def test_ring_can_be_visible_when_rectangle_corners_are_outside(self):
        a = math.radians(30)
        h = np.array([[math.cos(a), -math.sin(a), 200-200*math.cos(a)+200*math.sin(a)],
                      [math.sin(a), math.cos(a), 200-200*math.sin(a)-200*math.cos(a)], [0, 0, 1.]])
        q = self.m.ring_geometry(h, [200, 200], 140, [400, 400], [[40, 40, 360, 360]])
        self.assertTrue(q['passed'], q)

    def test_ring_outside_image_rejected(self):
        h = np.array([[1., 0, 200], [0, 1, 0], [0, 0, 1]])
        self.assertFalse(self.m.ring_geometry(h, [200, 200], 140, [400, 400], [[0, 0, 400, 400]])['passed'])

    def test_mirrored_ring_rejected(self):
        h = np.array([[-1., 0, 400], [0, 1, 0], [0, 0, 1.]])
        self.assertFalse(self.m.ring_geometry(h, [200, 200], 140, [400, 400], [[40, 40, 360, 360]])['passed'])

    def test_ambiguous_panel_binding_rejected(self):
        self.assertFalse(self.m.ring_geometry(np.eye(3), [200, 200], 140, [400, 400],
            [[40, 40, 360, 360], [40, 40, 360, 360]])['passed'])

    def test_feature_pose_consensus_rejects_orientation_alias(self):
        angles = [12, 14, 11, 13, 15, 100, 105]
        scales = [.8, .81, .79, .8, .82, .8, .81]
        keep = self.m.pose_consensus(angles, scales)
        self.assertEqual(keep, [0, 1, 2, 3, 4])


if __name__ == '__main__':
    unittest.main()
