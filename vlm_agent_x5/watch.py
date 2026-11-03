"""Poll one input directory and process each stable image version once per session."""
import hashlib
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from common import save

SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class StableImages:
    def __init__(self, stable_seconds):
        self.stable_seconds = stable_seconds
        self.observed = {}
        self.done = {}

    def ready(self, directory, now=None):
        now = time.monotonic() if now is None else now
        candidates = []
        current = set()
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUFFIXES:
                continue
            current.add(path)
            try:
                stat = path.stat()
                signature = (stat.st_size, stat.st_mtime_ns)
            except OSError:
                continue
            old = self.observed.get(path)
            if old is None or old[0] != signature:
                self.observed[path] = (signature, now)
                continue
            if stat.st_size == 0 or now - old[1] < self.stable_seconds:
                continue
            try:
                sha = digest(path)
                if path.stat().st_mtime_ns != signature[1] or path.stat().st_size != signature[0]:
                    self.observed.pop(path, None)
                    continue
            except OSError:
                self.observed.pop(path, None)
                continue
            if self.done.get(path) != sha:
                candidates.append((path, sha))
        for path in set(self.observed) - current:
            self.observed.pop(path, None)
        return candidates


def listen_for_zero(stop):
    import sys
    while not stop.is_set():
        line = sys.stdin.readline()
        if not line:
            return
        if line.strip() == '0':
            stop.set()
            return


def run_watch(directory, output, process, poll_seconds=1.0, stable_seconds=1.5,
              stop=None, max_images=None):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError('监听文件夹不存在：' + str(directory))
    if poll_seconds <= 0 or stable_seconds < 0:
        raise ValueError('轮询间隔须大于0，稳定等待须不小于0')
    stop = stop or threading.Event()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    session = {'started_at': datetime.now(timezone.utc).isoformat(),
               'ended_at': None, 'input_folder': str(directory), 'status': 'running',
               'items': []}
    record = output / 'watch_session.json'
    log_path = output / 'watch.log'
    def log(message):
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write(datetime.now(timezone.utc).isoformat() + ' ' + message + '\n')
    save(record, session)
    log('开始监听：' + str(directory))
    tracker = StableImages(stable_seconds)
    print('持续监听：' + str(directory), flush=True)
    print('会话记录：' + str(record), flush=True)
    print('输入 0 后回车退出；也可按 Ctrl+C。正在处理的图片会先处理完。', flush=True)
    interrupted = False
    try:
        while not stop.is_set():
            for path, sha in tracker.ready(directory):
                if stop.is_set():
                    break
                print('发现稳定图片：' + str(path), flush=True)
                log('开始处理：' + str(path) + ' sha256=' + sha)
                try:
                    item = process(path, sha)
                    item = {'image': str(path), 'sha256': sha, **item}
                except KeyboardInterrupt:
                    interrupted = True
                    raise
                except Exception as exc:
                    item = {'image': str(path), 'sha256': sha, 'status': 'failed',
                            'error': repr(exc)}
                    print('处理失败：' + repr(exc), flush=True)
                log('处理结果：' + item['status'] + ' ' + str(item.get('result', item.get('error', ''))))
                tracker.done[path] = sha
                session['items'].append(item)
                save(record, session)
                if max_images and len(session['items']) >= max_images:
                    stop.set()
                    break
            stop.wait(poll_seconds)
    except KeyboardInterrupt:
        interrupted = True
        print('收到 Ctrl+C，正在保存监听记录。', flush=True)
    finally:
        session['status'] = 'interrupted' if interrupted else 'stopped'
        session['ended_at'] = datetime.now(timezone.utc).isoformat()
        save(record, session)
        log('监听结束：' + session['status'])
        print('监听已退出；会话记录：' + str(record), flush=True)
    return session
