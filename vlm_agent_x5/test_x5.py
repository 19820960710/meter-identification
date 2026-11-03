import json
import unittest
from pathlib import Path
from unittest.mock import patch
from validation import validate_model,assemble
from yolo_adapter import matching_profile
from common import REPO,ROOT
from routing import choose
from validation import validate_scene

class TestX5Agent(unittest.TestCase):
    def test_existing_sha_profile(self):
        p=matching_profile(REPO/'samples/images/0.jpg')
        self.assertIsNotNone(p)
    def test_vlm_cannot_override_yolo(self):
        y={'raw':{'value':0.3776,'unit':'MPa'},'reading_status':'confirmed_candidate','binding':'single_panel'}
        m=validate_model(json.dumps({'仪表类型':'指针压力表','类型依据':'圆形表盘及指针','可核对性':'可见','表盘状态':'未见异常'}))
        self.assertEqual(assemble(y,m)['YOLO读数和单位']['原始读数'],0.3776)
        with self.assertRaises(ValueError):validate_model(json.dumps({**m,'读数':9}))
    def test_route_requires_actual_detection(self):
        self.assertEqual(choose({'raw':{'status':'detection_only','detections':[]}}),'scene')
        self.assertEqual(choose({'raw':{'status':'detection_only','detections':[{'label':'panel','score':.8}]}}),'meter')
        self.assertEqual(choose({'raw':{'status':'failed','detections':[]}}),'unknown')
    def test_scene_format_and_unsafe_claim(self):
        good={k:'未见可确认的可见隐患' for k in ('现场观察','安全隐患','场景合理性','建议','判断边界')}
        self.assertEqual(validate_scene(json.dumps(good))['安全隐患'],good['安全隐患'])
        good['安全隐患']='绝对安全'
        with self.assertRaises(ValueError):validate_scene(json.dumps(good))
    def test_unbound_multi_is_pending(self):
        y={'raw':{'value':1,'unit':'MPa'},'reading_status':'待复核','binding':'unresolved_multiple_or_missing'}
        self.assertEqual(assemble(y,{})['YOLO读数和单位']['状态'],'待复核')

if __name__=='__main__':unittest.main()
