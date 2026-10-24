"""Read-only environment check; this does not install packages or start inference."""
import importlib.util
import json
import platform
from pathlib import Path
import sys


def main():
    modules={name:importlib.util.find_spec(name) is not None for name in
             ['numpy','cv2','onnxruntime','hbm_runtime','hobot_dnn']}
    report={'system':platform.system(),'machine':platform.machine(),'python':sys.version.split()[0],
            'os_release':Path('/etc/os-release').read_text() if Path('/etc/os-release').exists() else None,
            'modules':modules,'video_devices':sorted(str(p) for p in Path('/dev').glob('video*')),
            'note':'Presence of an inference package does not establish model compatibility or BPU performance.'}
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
