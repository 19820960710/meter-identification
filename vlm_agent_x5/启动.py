"""X5中文唯一入口：图片 -> 队友CPU ONNX读表 -> 板端1B观察 -> 留档。"""
import argparse
import json
import shutil
import sys
import time
import uuid
from datetime import datetime,timezone
from pathlib import Path
from common import ROOT,REPO,resolve,save
from yolo_adapter import run as run_yolo
from internvl_x5_adapter import run as run_vlm
from validation import validate_model,assemble

def main():
    p=argparse.ArgumentParser(description='RDK X5 仪表图像观察 Agent')
    p.add_argument('--image',help='图片路径；与--request二选一')
    p.add_argument('--request',help='输入JSON路径；与--image二选一')
    p.add_argument('--config',default=str(ROOT/'config.json'))
    a=p.parse_args()
    if bool(a.image)==bool(a.request):p.error('必须且只能指定--image或--request')
    cfg=json.loads(Path(a.config).read_text(encoding='utf-8-sig'))
    req=json.loads(Path(a.request).read_text(encoding='utf-8-sig')) if a.request else {'image':a.image}
    if not isinstance(req,dict) or set(req)!={'image'} or not isinstance(req['image'],str):
        p.error('输入JSON只能包含字符串image字段')
    image=Path(req['image']).expanduser().resolve()
    if not image.is_file():p.error('图片不存在：'+str(image))
    if any(type(cfg['timeouts'][k]) is not int or cfg['timeouts'][k]<=0 for k in ('yolo_seconds','vlm_seconds')):
        p.error('超时必须是正整数秒')
    output=resolve(cfg['output_dir']);output.mkdir(parents=True,exist_ok=True)
    ident=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    folder=output/ident;folder.mkdir()
    shutil.copy2(image,folder/('input'+image.suffix.lower()))
    save(folder/'input.json',{'image':str(image),'imported_at':datetime.now(timezone.utc).isoformat(),
                              'config':str(Path(a.config).resolve()),'request':req})
    result={'run_id':ident,'status':'running','four_items':None,'yolo':None,'vlm':None,'errors':[]}
    try:
        ycfg={'model_path':resolve(cfg['yolo']['model_path'])}
        yolo=run_yolo(image,folder,ycfg,cfg['timeouts']['yolo_seconds'])
        result['yolo']=yolo
        save(folder/'yolo_summary.json',yolo)
        panel=yolo['selected_panel']
        vlm_image=image
        if panel:
            import cv2
            import numpy as np
            frame=cv2.imdecode(np.fromfile(str(image),dtype=np.uint8),cv2.IMREAD_COLOR)
            if frame is None:raise ValueError('无法解码目标表盘图片')
            h,w=frame.shape[:2]
            x1,y1,x2,y2=panel['box'];pad=.07*max(x2-x1,y2-y1)
            box=(max(0,int(x1-pad)),max(0,int(y1-pad)),min(w,int(x2+pad)),min(h,int(y2+pad)))
            vlm_image=folder/'target_dial.png'
            ok,encoded=cv2.imencode('.png',frame[box[1]:box[3],box[0]:box[2]])
            if not ok:raise ValueError('目标表盘裁剪失败')
            vlm_image.write_bytes(encoded.tobytes())
        prompt=(ROOT/'prompt_zh.txt').read_text(encoding='utf-8')
        prompt+='\nYOLO原始输入：'+json.dumps({'value':yolo['raw'].get('value'),'unit':yolo['raw'].get('unit'),
                 'reading_status':yolo['reading_status'],'target_binding':yolo['binding']},ensure_ascii=False)
        save(folder/'vlm_request.json',{'image':str(vlm_image),'prompt':prompt,'target_binding':yolo['binding']})
        vcfg={k:resolve(cfg['internvl'][k]) for k in ('vit_bin','llm_gguf')}
        try:
            attempt=run_vlm(vlm_image,prompt,vcfg,folder,cfg['timeouts']['vlm_seconds'])
            result['vlm']=attempt
            model=validate_model(attempt['candidate'])
            if yolo['binding']!='single_panel':
                model['可核对性']='目标表盘未唯一绑定；图片观察不能证明该YOLO读数属于某一块表。'+model['可核对性']
        except Exception as exc:
            result['errors'].append('VLM：'+repr(exc));model={}
        result['four_items']=assemble(yolo,model)
        result['status']='completed' if model and yolo['raw'].get('status')=='readable' else 'partial'
    except Exception as exc:
        result['errors'].append('流程：'+repr(exc));result['status']='failed'
    finally:
        save(folder/'result.json',result)
        (folder/'agent.log').write_text('\n'.join(result['errors']) or '流程完成；原始子进程日志见同目录。',encoding='utf-8')
    print('结果：'+str(folder/'result.json'))
    print(json.dumps(result['four_items'],ensure_ascii=False,indent=2) if result['four_items'] else result['status'])
    return 0 if result['status']=='completed' else 2
if __name__=='__main__':sys.exit(main())
