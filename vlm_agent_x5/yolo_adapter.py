"""Call the teammate's unchanged CPU ONNX reader; preserve its complete result."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from common import REPO,save

def matching_profile(image):
    sha=hashlib.sha256(Path(image).read_bytes()).hexdigest()
    found=[]
    for p in (REPO/'configs/gauges').glob('*.json'):
        try:
            if json.loads(p.read_text(encoding='utf-8')).get('image_sha256')==sha:found.append(p)
        except (ValueError,OSError):continue
    if len(found)>1:raise ValueError('同一图片找到多个标定，拒绝自动选择')
    return found[0] if found else None

def run(image,folder,config,timeout):
    profile=matching_profile(image)
    cmd=[sys.executable,str(REPO/'tools/run_onnx.py'),'--image',str(image),
         '--model',str(config['model_path']),'--output',str(folder/'yolo')]
    if profile:cmd+=['--profile',str(profile)]
    try:
        p=subprocess.run(cmd,cwd=REPO,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
        stdout,stderr=p.stdout,p.stderr
        code=p.returncode
    except subprocess.TimeoutExpired as exc:
        stdout=(exc.stdout or b'').decode(errors='replace') if isinstance(exc.stdout,bytes) else exc.stdout or ''
        stderr=(exc.stderr or b'').decode(errors='replace') if isinstance(exc.stderr,bytes) else exc.stderr or ''
        code=None
    (folder/'yolo_stdout.log').write_text(stdout,encoding='utf-8')
    (folder/'yolo_stderr.log').write_text(stderr,encoding='utf-8')
    result_path=folder/'yolo/result.json'
    result=json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {'status':'failed','reason':'YOLO未产生结果'}
    panels=[d for d in result.get('detections',[]) if d.get('label')=='panel' and d.get('score',0)>=.5]
    binding='single_panel' if len(panels)==1 else 'unresolved_multiple_or_missing'
    calibrated=(profile is not None and result.get('status')=='readable' and result.get('value') is not None)
    return {'raw':result,'profile':str(profile) if profile else None,'binding':binding,
            'reading_status':'confirmed_candidate' if calibrated and binding=='single_panel' else '待复核',
            'exit_code':code,'selected_panel':panels[0] if binding=='single_panel' else None}
