import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('portable_cli',ROOT/'tools/run_onnx.py')
CLI=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(CLI)


class PortableBindingTests(unittest.TestCase):
    def setUp(self):
        self.raw=(ROOT/'samples/images/0.jpg').read_bytes()
        self.image=cv2.imdecode(np.frombuffer(self.raw,np.uint8),1)
        self.profile=json.loads((ROOT/'configs/gauges/photo_fae0688c1fa78bd7.json').read_text(encoding='utf-8'))

    def test_portable_profile_keeps_original_byte_binding(self):
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(),self.profile['image_sha256'])
        CLI.validate_profile(self.raw,self.image,self.profile)

    def test_wrong_picture_rejected_before_missing_model_is_opened(self):
        with self.assertRaisesRegex(ValueError,'another image'):
            CLI.run(ROOT/'samples/images/10.jpg',ROOT/'does-not-exist.onnx',ROOT/'configs/gauges/photo_fae0688c1fa78bd7.json')

    def test_partial_entry_rejects_full_circle_instead_of_misreading_units(self):
        self.profile['scale_mode']='full_circle_uniform'
        with self.assertRaisesRegex(ValueError,'partial scales'):
            CLI.validate_profile(self.raw,self.image,self.profile)


if __name__=='__main__':
    unittest.main()
