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
from validation import validate_model,assemble,validate_scene
from preflight import check as check_env
from routing import choose as choose_mode
from watch import run_watch, digest

def process_one(image,cfg,config_path,req=None,expected_sha=None):
    image=Path(image).expanduser().resolve()
    if not image.is_file():raise FileNotFoundError('图片不存在：'+str(image))
    output=resolve(cfg['output_dir']);output.mkdir(parents=True,exist_ok=True)
    ident=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    folder=output/ident;folder.mkdir()
    snapshot=folder/('input'+image.suffix.lower())
    original_image=image
    req=req or {'image':str(image)}
    result={'run_id':ident,'status':'running','mode':None,'four_items':None,'scene_items':None,'yolo':None,'vlm':None,'errors':[]}
    try:
        shutil.copy2(image,snapshot)
        snapshot_sha=digest(snapshot)
        if expected_sha and snapshot_sha!=expected_sha:
            raise ValueError('图片复制时内容发生变化，跳过本次处理；文件稳定后可重新检测')
        save(folder/'input.json',{'image':str(original_image),'snapshot':str(snapshot),
                                  'sha256':snapshot_sha,'imported_at':datetime.now(timezone.utc).isoformat(),
                                  'config':str(Path(config_path).resolve()),'request':req})
        save(folder/'env_check.json',check_env(cfg))
        image=snapshot
        ycfg={'model_path':resolve(cfg['yolo']['model_path'])}
        yolo=run_yolo(image,folder,ycfg,cfg['timeouts']['yolo_seconds'])
        result['yolo']=yolo
        save(folder/'yolo_summary.json',yolo)
        result['mode']=choose_mode(yolo)
        if result['mode']=='unknown':raise ValueError('YOLO检测失败，不能认定现场没有仪表')
        panel=yolo['selected_panel'] if result['mode']=='meter' else None
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
        if result['mode']=='meter':
            prompt=resolve(cfg['prompts']['meter']).read_text(encoding='utf-8')
            prompt+='\nYOLO原始输入：'+json.dumps({'value':yolo['raw'].get('value'),'unit':yolo['raw'].get('unit'),
                     'reading_status':yolo['reading_status'],'target_binding':yolo['binding']},ensure_ascii=False)
        else:
            knowledge=json.loads(resolve(cfg['scene_knowledge']).read_text(encoding='utf-8'))
            prompt=resolve(cfg['prompts']['scene']).read_text(encoding='utf-8')
            prompt+='\n可用资料摘要：'+json.dumps([{'id':s['id'],'applies_to':s['applies_to'],'verified_body':s['verified_body']} for s in knowledge['sources']],ensure_ascii=False)
            save(folder/'scene_sources.json',knowledge)
        save(folder/'vlm_request.json',{'image':str(vlm_image),'prompt':prompt,'target_binding':yolo['binding']})
        vcfg={k:resolve(cfg['internvl'][k]) for k in ('vit_bin','llm_gguf')}
        try:
            attempt=run_vlm(vlm_image,prompt,vcfg,folder,cfg['timeouts']['vlm_seconds'])
            result['vlm']=attempt
            model=validate_model(attempt['candidate']) if result['mode']=='meter' else validate_scene(attempt['candidate'])
            if result['mode']=='meter' and yolo['binding']!='single_panel':
                model['可核对性']='目标表盘未唯一绑定；图片观察不能证明该YOLO读数属于某一块表。'+model['可核对性']
        except Exception as exc:
            result['errors'].append('VLM：'+repr(exc));model={}
        if result['mode']=='meter':result['four_items']=assemble(yolo,model)
        else:result['scene_items']=model or None
        result['status']='completed' if model and (result['mode']=='scene' or yolo['raw'].get('status')=='readable') else 'partial'
    except Exception as exc:
        result['errors'].append('流程：'+repr(exc));result['status']='failed'
    except KeyboardInterrupt:
        result['errors'].append('流程：手动中断');result['status']='interrupted'
        raise
    finally:
        save(folder/'result.json',result)
        (folder/'agent.log').write_text('\n'.join(result['errors']) or '流程完成；原始子进程日志见同目录。',encoding='utf-8')
    print('结果：'+str(folder/'result.json'))
    print(json.dumps(result['four_items'] or result['scene_items'],ensure_ascii=False,indent=2) if result['four_items'] or result['scene_items'] else result['status'])
    return {'status':result['status'],'result':str(folder/'result.json'),'run_id':ident}

def main():
    p=argparse.ArgumentParser(description='RDK X5 仪表图像观察 Agent')
    group=p.add_mutually_exclusive_group()
    group.add_argument('--image',help='处理一张图片')
    group.add_argument('--request',help='单图输入JSON路径')
    group.add_argument('--watch-folder',help='持续监听文件夹中新写入或修改的图片')
    p.add_argument('--config',default=str(ROOT/'config.json'))
    p.add_argument('--check-env',action='store_true',help='只检查板端环境，不调用YOLO或VLM')
    p.add_argument('--poll-seconds',type=float,default=1.0,help='监听轮询间隔，默认1秒')
    p.add_argument('--stable-seconds',type=float,default=1.5,help='文件未变化后等待秒数，默认1.5秒')
    a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text(encoding='utf-8-sig'))
    if a.check_env:
        status=check_env(cfg)
        output=resolve(cfg['output_dir']);output.mkdir(parents=True,exist_ok=True)
        save(output/'环境检查.json',status)
        print(json.dumps(status,ensure_ascii=False,indent=2))
        print('环境检查记录：'+str(output/'环境检查.json'))
        return 0 if status['ready'] else 2
    if not (a.image or a.request or a.watch_folder):p.error('请指定--image、--request或--watch-folder')
    if any(type(cfg['timeouts'][k]) is not int or cfg['timeouts'][k]<=0 for k in ('yolo_seconds','vlm_seconds')):
        p.error('超时必须是正整数秒')
    if a.watch_folder:
        if a.poll_seconds<=0 or a.stable_seconds<0:p.error('监听时间参数无效')
        output=resolve(cfg['output_dir'])
        ident='watch_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
        session_folder=output/ident
        import threading
        from watch import listen_for_zero
        stop=threading.Event()
        threading.Thread(target=listen_for_zero,args=(stop,),daemon=True).start()
        run_watch(a.watch_folder,session_folder,
                  lambda path,sha:process_one(path,cfg,a.config,expected_sha=sha),
                  a.poll_seconds,a.stable_seconds,stop)
        return 0
    req=json.loads(Path(a.request).read_text(encoding='utf-8-sig')) if a.request else {'image':a.image}
    if not isinstance(req,dict) or set(req)!={'image'} or not isinstance(req['image'],str):
        p.error('输入JSON只能包含字符串image字段')
    return 0 if process_one(req['image'],cfg,a.config,req)['status']=='completed' else 2
if __name__=='__main__':sys.exit(main())
