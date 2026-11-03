"""Read-only preflight for the real X5 board and PC smoke checks."""
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from common import ROOT,REPO,resolve

def check(config):
    checks={}
    checks['platform_linux_arm64']=sys.platform=='linux' and platform.machine().lower() in ('aarch64','arm64')
    checks['tros_setup']=Path('/opt/tros/humble/setup.bash').exists()
    checks['ros2_command']=shutil.which('ros2') is not None or checks['tros_setup']
    checks['cpu_onnx_model']=resolve(config['yolo']['model_path']).is_file()
    checks['team_yolo_entry']=(REPO/'tools/run_onnx.py').is_file()
    checks['meter_prompt']=resolve(config['prompts']['meter']).is_file()
    checks['scene_prompt']=resolve(config['prompts']['scene']).is_file()
    checks['scene_knowledge']=resolve(config['scene_knowledge']).is_file()
    checks['x5_visual_bin']=resolve(config['internvl']['vit_bin']).is_file()
    checks['paired_gguf']=resolve(config['internvl']['llm_gguf']).is_file()
    for name in ('numpy','cv2','onnxruntime'):
        checks['python_'+name]=importlib.util.find_spec(name) is not None
    if checks['tros_setup']:
        cmd=['bash','-lc','source /opt/tros/humble/setup.bash && ros2 pkg executables hobot_llamacpp']
        try:
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
            checks['hobot_llamacpp_ros_package']=p.returncode==0 and 'hobot_llamacpp' in p.stdout
        except (OSError,subprocess.TimeoutExpired):checks['hobot_llamacpp_ros_package']=False
    else:checks['hobot_llamacpp_ros_package']=False
    return {'ready':all(checks.values()),'checks':checks,
            'manual_checks':['RDK系统版本与ROS包版本','srpi-config的ION=1.6GB及CPU频率','模型文件来源及板端推理实际耗时']}
