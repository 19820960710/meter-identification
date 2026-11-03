"""Shared paths and JSON persistence for the isolated X5 agent."""
import json
import os
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent

def resolve(path):
    p=Path(path).expanduser()
    return p if p.is_absolute() else (ROOT/p).resolve()

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,delete=False) as tmp:
        json.dump(obj,tmp,ensure_ascii=False,indent=2,allow_nan=False)
        name=tmp.name
    os.replace(name,path)
