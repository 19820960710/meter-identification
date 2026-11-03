"""Read-only preflight for the real X5 board and PC smoke checks."""
import importlib
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
    checks['x5_visual_bin']=resolve(config['internvl']['vit_bin']).is_file() and resolve(config['internvl']['vit_bin']).stat().st_size>1024*1024
    checks['paired_gguf']=resolve(config['internvl']['llm_gguf']).is_file() and resolve(config['internvl']['llm_gguf']).stat().st_size>1024*1024
    checks['ros_topic_helper']=(ROOT/'ros_topic_once.py').is_file()
    checks['system_python']=Path('/usr/bin/python3').is_file()
    for name in ('numpy','cv2','onnxruntime'):
        try:
            importlib.import_module(name)
            checks['python_'+name]=True
        except Exception:
            checks['python_'+name]=False
    if checks['tros_setup']:
        cmd=['bash','-lc','source /opt/tros/humble/setup.bash && ros2 pkg executables hobot_llamacpp']
        try:
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
            checks['hobot_llamacpp_ros_package']=p.returncode==0 and 'hobot_llamacpp' in p.stdout
        except (OSError,subprocess.TimeoutExpired):checks['hobot_llamacpp_ros_package']=False
    else:checks['hobot_llamacpp_ros_package']=False
    if checks['tros_setup'] and checks['system_python']:
        cmd=['bash','-lc',"source /opt/tros/humble/setup.bash && /usr/bin/python3 -c 'import rclpy; from ai_msgs.msg import PerceptionTargets'"]
        try:
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
            checks['ros_result_message_import']=p.returncode==0
        except (OSError,subprocess.TimeoutExpired):checks['ros_result_message_import']=False
    else:checks['ros_result_message_import']=False
    return {'ready':all(checks.values()),'checks':checks,
            'manual_checks':['RDK系统版本与ROS包版本','srpi-config的ION=1.6GB及CPU频率','模型文件来源及板端推理实际耗时']}
