"""Verify tracked handoff hashes and image/profile bindings; standard library only."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'BUNDLE_SHA256.json')
    args=parser.parse_args()
    entries=json.loads(args.manifest.read_text(encoding='utf-8'))['files']
    failures=[]
    for name,expected in entries.items():
        path=ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            failures.append(name)
    samples=json.loads((ROOT/'samples/manifest.json').read_text(encoding='utf-8'))['images']
    for row in samples:
        raw=(ROOT/row['image']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['image_sha256']:
            failures.append(row['image'])
        if row['profile']:
            profile=json.loads((ROOT/row['profile']).read_text(encoding='utf-8'))
            if profile['image_sha256']!=row['image_sha256'] or profile['image']!=row['image']:
                failures.append(row['profile'])
    print(json.dumps({'verified_files':len(entries),'sample_images':len(samples),
                      'failures':failures,'passed':not failures},ensure_ascii=False,indent=2))
    return 1 if failures else 0


if __name__=='__main__':
    raise SystemExit(main())
