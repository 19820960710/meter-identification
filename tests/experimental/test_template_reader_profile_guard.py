import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / 'experiments/template-transfer-20261024/read_template_photo.py'


class TemplateProfileGuardTests(unittest.TestCase):
    def setUp(self):
        if not FILE.exists():
            self.fail('Standalone single-template reader is not implemented.')
        spec = importlib.util.spec_from_file_location('template_reader_guard_test', FILE)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_only_selected_active_template_profile_is_allowed(self):
        active = ROOT / 'configs/gauges'
        reference = active / 'photo_13e7265b20639756.json'
        self.assertTrue(self.m.calibration_path_allowed(reference, reference, active))
        self.assertFalse(self.m.calibration_path_allowed(active/'demo_832.json', reference, active))

    def test_derived_runtime_profile_is_allowed_outside_active_directory(self):
        active = ROOT / 'configs/gauges'
        reference = active / 'photo_13e7265b20639756.json'
        self.assertTrue(self.m.calibration_path_allowed(ROOT/'artifacts/run/runtime.json', reference, active))
