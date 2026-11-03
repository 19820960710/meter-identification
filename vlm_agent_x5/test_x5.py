import json
import unittest
import tempfile
import io
import threading
from pathlib import Path
from unittest.mock import patch
from validation import validate_model,assemble
from yolo_adapter import matching_profile
from common import REPO,ROOT
from routing import choose
from validation import validate_scene
from watch import StableImages,run_watch,listen_for_zero

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

    def test_watch_waits_for_complete_file_and_reprocesses_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            image=folder/'image.jpg'
            image.write_bytes(b'first')
            tracker=StableImages(1.0)
            self.assertEqual(tracker.ready(folder,now=0),[])
            self.assertEqual(tracker.ready(folder,now=.5),[])
            first=tracker.ready(folder,now=1.1)
            self.assertEqual(len(first),1)
            tracker.done[image]=first[0][1]
            self.assertEqual(tracker.ready(folder,now=2),[])
            image.write_bytes(b'second')
            self.assertEqual(tracker.ready(folder,now=2.1),[])
            self.assertEqual(len(tracker.ready(folder,now=3.2)),1)

    def test_watch_session_stops_and_records_results(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            incoming=folder/'incoming';incoming.mkdir()
            (incoming/'image.png').write_bytes(b'image')
            calls=[]
            session=run_watch(incoming,folder/'output',
                              lambda path,sha:calls.append(path) or {'status':'completed','result':'test.json'},
                              poll_seconds=.01,stable_seconds=0,max_images=1)
            self.assertEqual(len(calls),1)
            self.assertEqual(session['status'],'stopped')
            self.assertEqual(len(session['items']),1)
            self.assertTrue((folder/'output'/'watch_session.json').is_file())
            self.assertTrue((folder/'output'/'watch.log').is_file())

    def test_zero_stops_watch(self):
        stop=threading.Event()
        with patch('sys.stdin',io.StringIO('0\n')):
            listen_for_zero(stop)
        self.assertTrue(stop.is_set())

if __name__=='__main__':unittest.main()
