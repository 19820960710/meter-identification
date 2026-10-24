"""CPU-only protocol/parity check; never reports X5/BPU or physical accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
from onnx_detector import preprocess,decode,raw_heads_to_packed
from run_onnx import validate_profile,read_geometry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/validation')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/'models/annotation/meter_labeler_v3/best.onnx',ROOT/'models/rdk_x5/meter_labeler_v3_raw_heads.onnx']
    sessions=[ort.InferenceSession(str(p),providers=['CPUExecutionProvider']) for p in paths]
    samples=json.loads((ROOT/'samples/manifest.json').read_text(encoding='utf-8'))['images']
    rows=[]
    for sample in samples:
        raw=(ROOT/sample['image']).read_bytes();image=cv2.imdecode(np.frombuffer(raw,np.uint8),1)
        tensor,mapping=preprocess(image)
        outputs=[s.run(None,{s.get_inputs()[0].name:tensor}) for s in sessions]
        packed=outputs[0][0];reconstructed=raw_heads_to_packed(outputs[1])
        predictions=[decode(o,image.shape,mapping) for o in [packed,reconstructed]]
        record={'id':sample['id'],'image_sha256':hashlib.sha256(raw).hexdigest(),
            'detector_counts':[len(d) for d in predictions],
            'labels_match':[d['label'] for d in predictions[0]]==[d['label'] for d in predictions[1]],
            'maximum_box_difference_px':None,'maximum_score_difference':None,
            'status':'detection_only','value':None}
        if record['labels_match']:
            record['maximum_box_difference_px']=max((abs(np.array(a['box'])-b['box']).max() for a,b in zip(*predictions)),default=0.)
            record['maximum_score_difference']=max((abs(a['score']-b['score']) for a,b in zip(*predictions)),default=0.)
        if sample['profile']:
            profile=json.loads((ROOT/sample['profile']).read_text(encoding='utf-8'))
            validate_profile(raw,image,profile)
            readings=[]
            for detections in predictions:
                try:
                    reading=read_geometry(image,profile,detections)
                    readings.append({'status':reading['status'],'value':reading['value']})
                except (ValueError,KeyError,TypeError,cv2.error) as error:
                    readings.append({'status':'needs_review','value':None,'reason':str(error)})
            record.update(status=readings[0]['status'],value=readings[0]['value'],
                          raw_head_status=readings[1]['status'],raw_head_value=readings[1]['value'])
        rows.append(record)
    # Save predictions before loading historical output comparisons.
    prediction_path=args.output/'predictions.json'
    prediction_path.write_text(json.dumps({'rows':rows},ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    expected=json.loads((ROOT/'evidence/expected_pc_onnx.json').read_text(encoding='utf-8'))['rows']
    by_id={r['id']:r for r in rows}
    checks={'all_six_raw_heads_agree_with_packed_detector':all(r['labels_match'] and r['maximum_box_difference_px']<.1 and r['maximum_score_difference']<.001 for r in rows),
        'all_historical_pc_statuses_preserved':all(by_id[r['id']]['status']==r['status'] for r in expected),
        'all_historical_pc_readings_preserved':all(abs(by_id[r['id']]['value']-r['value'])<1e-10 for r in expected if r['value'] is not None)}
    report={'backend':'ONNX Runtime CPU','X5_BPU_verified':False,'physical_accuracy_verified':False,
        'prediction_sha256':hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        'sample_images':len(rows),'raw_head_box_max_difference_px':max(r['maximum_box_difference_px'] or 0 for r in rows),
        'checks':checks,'passed':all(checks.values()),'rows':rows}
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False,indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
