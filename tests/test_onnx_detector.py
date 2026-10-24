from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class OnnxProtocolTests(unittest.TestCase):
    def setUp(self):
        if not (ROOT/'src/onnx_detector.py').exists():
            self.fail('Portable ONNX detector is not implemented.')
        sys.path.insert(0,str(ROOT/'src'))
        import onnx_detector
        self.m=onnx_detector

    def test_rgb_normalization_and_square_letterbox(self):
        image=np.full((240,320,3),[0,127,255],np.uint8)
        tensor, mapping=self.m.preprocess(image)
        self.assertEqual(tensor.shape,(1,3,640,640))
        np.testing.assert_allclose(tensor[0,:,100,100],[1,127/255,0],atol=1e-7)
        self.assertEqual(mapping,(2.,2.,0.,80.))

    def test_inverse_mapping_uses_actual_integer_resize_per_axis(self):
        for h,w in [(530,504),(430,438)]:
            _,mapping=self.m.preprocess(np.zeros((h,w,3),np.uint8))
            scale=min(640/w,640/h);nw,nh=round(w*scale),round(h*scale)
            left,top=round((640-nw)/2-.1),round((640-nh)/2-.1)
            a=np.array([10*nw/w+left,20*nh/h+top])
            b=np.array([50*nw/w+left,100*nh/h+top])
            output=np.zeros((1,7,8400),np.float32)
            output[0,:4,0]=np.r_[(a+b)/2,b-a];output[0,4,0]=.9
            result=self.m.decode(output,(h,w,3),mapping)
            np.testing.assert_allclose(result[0]['box'],[10,20,50,100],atol=1e-5)

    def test_box_mapping_and_classwise_nms(self):
        output=np.zeros((1,7,8400),np.float32)
        for i,cls,score in [(0,0,.9),(1,0,.8),(2,1,.85)]:
            output[0,:4,i]=[60,190,80,160]
            output[0,4+cls,i]=score
        detections=self.m.decode(output,(240,320,3),(2,0,80))
        self.assertEqual([d['label'] for d in detections],['panel','needle'])
        np.testing.assert_allclose(detections[0]['box'],[10,15,50,95])

    def test_wrong_model_protocol_is_rejected(self):
        for output in [np.zeros((1,84,8400)),np.zeros((1,7,100)),np.zeros((1,8400,7))]:
            with self.subTest(shape=output.shape),self.assertRaises(ValueError):
                self.m.decode(output,(240,320,3),(2,0,80))

    def test_empty_and_nonfinite_output(self):
        output=np.zeros((1,7,8400),np.float32)
        self.assertEqual(self.m.decode(output,(240,320,3),(2,0,80)),[])
        output[0,4,0]=float('nan')
        with self.assertRaises(ValueError):
            self.m.decode(output,(240,320,3),(2,0,80))

    def test_raw_head_dfl_decode_has_known_uniform_mean(self):
        heads=[]
        for size in [80,40,20]:
            heads.extend([np.zeros((1,size,size,3),np.float32),np.zeros((1,size,size,64),np.float32)])
        output=self.m.raw_heads_to_packed(heads)
        self.assertEqual(output.shape,(1,7,8400))
        np.testing.assert_allclose(output[0,:,0],[4,4,120,120,.5,.5,.5],atol=1e-6)


if __name__=='__main__':
    unittest.main()
