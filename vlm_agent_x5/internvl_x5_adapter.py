"""Official ROS CLI adapter for X5 InternVL2.5-1B; no Python Transformers fallback."""
import os
import re
import shlex
import signal
import subprocess
from pathlib import Path

def run(image,prompt,config,folder,timeout):
    vit=Path(config['vit_bin']);gguf=Path(config['llm_gguf'])
    for p in (vit,gguf):
        if not p.is_file():raise FileNotFoundError('缺少X5模型文件：'+str(p))
    if vit.name!='vit_model_int16_v2.bin' or gguf.name!='Qwen2.5-0.5B-Instruct-Q4_0.gguf':
        raise ValueError('X5 InternVL2.5-1B视觉BIN与GGUF必须配套')
    args=['ros2','run','hobot_llamacpp','hobot_llamacpp','--ros-args',
          '-p','feed_type:=0','-p','image:='+str(image),'-p','image_type:=0',
          '-p','user_prompt:='+prompt,'-p','model_file_name:='+str(vit),
          '-p','llm_model_name:='+str(gguf)]
    script='source /opt/tros/humble/setup.bash && exec '+shlex.join(args)
    p=subprocess.Popen(['bash','-lc',script],stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                       text=True,start_new_session=True)
    timed_out=False
    try:stdout,stderr=p.communicate(timeout=timeout)
    except KeyboardInterrupt:
        # A manually stopped watch session must not leave the ROS model running.
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.communicate()
        raise
    except subprocess.TimeoutExpired:
        timed_out=True
        os.killpg(p.pid,signal.SIGTERM)
        try:stdout,stderr=p.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL);stdout,stderr=p.communicate()
    (folder/'vlm_stdout.log').write_text(stdout,encoding='utf-8')
    (folder/'vlm_stderr.log').write_text(stderr,encoding='utf-8')
    # ROS may keep a node alive after generation. Extract only an explicit JSON answer.
    candidates=re.findall(r'\{[^{}]*"(?:仪表类型|现场观察)"[^{}]*\}',stdout+'\n'+stderr,re.S)
    return {'candidate':candidates[-1] if candidates else None,'return_code':p.returncode,
            'timed_out':timed_out,'command_argv':args}
