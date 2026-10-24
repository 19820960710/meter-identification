"""Portable single-image ONNX CPU detection and optional image-bound partial reading."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import meter_reader as meter
from calibration_binding import bind_detections
from onnx_detector import preprocess,decode,raw_heads_to_packed


def validate_profile(raw,image,profile):
    meter.validate_calibration_image(raw,profile)
    if profile.get('image_size')!=list(image.shape[1::-1]):
        raise ValueError('Profile dimensions do not match the image.')
    if profile.get('scale_mode','partial')!='partial':
        raise ValueError('This lightweight entry accepts partial scales only.')
    if not str(profile.get('unit','')).strip() or len(profile.get('ticks',[]))<3:
        raise ValueError('Need a unit and at least three image-bound anchors.')
    w,h=profile['image_size']
    for tick in profile['ticks']:
        x,y=tick['point']
        if not np.isfinite([x,y,tick['value']]).all() or not 0<=x<w or not 0<=y<h:
            raise ValueError('Invalid calibration anchor.')
    division=profile.get('minimum_division')
    if division is not None and (not np.isfinite(division) or division<=0):
        raise ValueError('Minimum division must be positive and finite.')


def read_geometry(image,profile,detections):
    panel,needle,axle,association=bind_detections(profile,detections)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    center,detail=meter.locate_center(gray,axle)
    if not (needle[0]<=center[0]<=needle[2] and needle[1]<=center[1]<=needle[3]):
        raise ValueError('Refined image axle is outside the needle ROI.')
    radius=min(panel[2]-panel[0],panel[3]-panel[1])/2
    tip,diagnostic=meter.locate_pointer(gray,needle,center,radius,color_image=image)
    value=meter.angle_reading(center,tip,profile['ticks'],profile['direction'])
    return {'status':'readable','value':value,'display_value':f'{value:.4f} {profile["unit"]}',
        'association_diagnostic':association,'geometry':{'center':center,'pointer_endpoint':tip,
        'center_diagnostic':detail,'pointer_diagnostic':diagnostic}}


def run(image_path,model_path,profile_path=None,protocol='packed'):
    start=time.perf_counter()
    result={'status':'needs_review','value':None,'unit':None,'reason':None,'detections':[],
        'geometry':None,'calibration':None,'backend':'onnxruntime_cpu',
        'operating_status':'not_evaluated','physical_accuracy_verified':False}
    raw=Path(image_path).read_bytes()
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),1)
    if image is None:
        raise ValueError('Cannot decode input image.')
    result['image_sha256']=hashlib.sha256(raw).hexdigest()
    profile=json.loads(Path(profile_path).read_text(encoding='utf-8')) if profile_path else None
    if profile is not None:
        validate_profile(raw,image,profile)  # Reject wrong-image profiles before inference.
        result.update(calibration=profile,unit=profile['unit'])
    import onnxruntime as ort
    session=ort.InferenceSession(str(model_path),providers=['CPUExecutionProvider'])
    tensor,mapping=preprocess(image)
    t=time.perf_counter()
    outputs=session.run(None,{session.get_inputs()[0].name:tensor})
    result['inference_ms']=(time.perf_counter()-t)*1000
    if protocol=='packed':
        if len(outputs)!=1:
            raise ValueError('Packed protocol needs one output; select raw-heads for the X5 export.')
        output=outputs[0]
    else:
        output=raw_heads_to_packed(outputs)
    result.update(detections=decode(output,image.shape,mapping),
                  model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),protocol=protocol)
    if profile is None:
        result.update(status='detection_only',display_value='DETECTION ONLY')
    else:
        try:
            result.update(read_geometry(image,profile,result['detections']))
        except (ValueError,KeyError,TypeError,cv2.error) as error:
            result['reason']=str(error)
    result['processing_ms_including_session_load']=(time.perf_counter()-start)*1000
    return result,image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=ROOT/'models/annotation/meter_labeler_v3/best.onnx')
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--protocol',choices=['packed','raw-heads'],default='packed')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/onnx')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        result,image=run(args.image,args.model,args.profile,args.protocol)
        annotated=meter.draw_result(image,result)
        (args.output/'annotated.jpg').write_bytes(cv2.imencode('.jpg',annotated)[1].tobytes())
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,cv2.error) as error:
        result={'status':'needs_review','value':None,'reason':str(error),'backend':'onnxruntime_cpu'}
    text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
    (args.output/'result.json').write_text(text,encoding='utf-8')
    print(text)
    return 0 if result['status'] in ('readable','detection_only') else 2


if __name__=='__main__':
    raise SystemExit(main())
