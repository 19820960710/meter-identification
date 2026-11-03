"""Fail-closed routing after actual YOLO detections."""
def choose(yolo):
    raw=yolo['raw']
    if any(d.get('label')=='panel' and d.get('score',0)>=.5 for d in raw.get('detections',[])):
        return 'meter'
    if raw.get('status') in ('detection_only','needs_review'):
        return 'scene'
    return 'unknown'
