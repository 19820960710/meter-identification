import json
import unittest
from pathlib import Path
from unittest.mock import patch
from validation import validate_model,assemble
from yolo_adapter import matching_profile
from common import REPO,ROOT

class TestX5Agent(unittest.TestCase):
    def test_existing_sha_profile(self):
        p=matching_profile(REPO/'samples/images/0.jpg')
        self.assertIsNotNone(p)
    def test_vlm_cannot_override_yolo(self):
        y={'raw':{'value':0.3776,'unit':'MPa'},'reading_status':'confirmed_candidate','binding':'single_panel'}
        m=validate_model(json.dumps({'仪表类型':'指针压力表','类型依据':'圆形表盘及指针','可核对性':'可见','表盘状态':'未见异常'}))
        self.assertEqual(assemble(y,m)['YOLO读数和单位']['原始读数'],0.3776)
        with self.assertRaises(ValueError):validate_model(json.dumps({**m,'读数':9}))
    def test_unbound_multi_is_pending(self):
        y={'raw':{'value':1,'unit':'MPa'},'reading_status':'待复核','binding':'unresolved_multiple_or_missing'}
        self.assertEqual(assemble(y,{})['YOLO读数和单位']['状态'],'待复核')

if __name__=='__main__':unittest.main()
