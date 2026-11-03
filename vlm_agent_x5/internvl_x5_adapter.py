"""X5 InternVL2.5-1B ROS adapter; read the final answer from a private topic."""
import json
import os
import select
import shlex
import signal
import subprocess
import time
import uuid
from pathlib import Path


def extract_answer_json(answer):
    """Extract one JSON object without treating log lines as model output."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(answer):
        if char != '{':
            continue
        try:
            value, _ = decoder.raw_decode(answer[index:])
        except ValueError:
            continue
        if isinstance(value, dict) and ('仪表类型' in value or '现场观察' in value):
            return json.dumps(value, ensure_ascii=False)
    return None


def stop_group(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run(image, prompt, config, folder, timeout):
    folder = Path(folder)
    vit = Path(config['vit_bin'])
    gguf = Path(config['llm_gguf'])
    for path in (vit, gguf):
        if not path.is_file():
            raise FileNotFoundError('缺少X5模型文件：' + str(path))
    if vit.name != 'vit_model_int16_v2.bin' or gguf.name != 'Qwen2.5-0.5B-Instruct-Q4_0.gguf':
        raise ValueError('X5 InternVL2.5-1B视觉BIN与GGUF必须配套')

    topic = '/vlm_agent_x5/result_' + uuid.uuid4().hex
    subscriber_args = ['/usr/bin/python3', '-u', str(Path(__file__).with_name('ros_topic_once.py')), topic]
    model_args = ['ros2', 'run', 'hobot_llamacpp', 'hobot_llamacpp', '--ros-args',
                  '-p', 'feed_type:=0', '-p', 'image:=' + str(image), '-p', 'image_type:=0',
                  '-p', 'user_prompt:=' + prompt, '-p', 'model_file_name:=' + str(vit),
                  '-p', 'llm_model_name:=' + str(gguf),
                  '-p', 'ai_msg_pub_topic_name:=' + topic]
    setup = 'source /opt/tros/humble/setup.bash && exec '
    subscriber = None
    model = None
    timed_out = False
    topic_payload = None
    start = time.monotonic()
    with (folder / 'vlm_topic_stderr.log').open('w', encoding='utf-8') as topic_err, \
         (folder / 'vlm_stdout.log').open('w', encoding='utf-8') as model_out, \
         (folder / 'vlm_stderr.log').open('w', encoding='utf-8') as model_err:
        try:
            subscriber = subprocess.Popen(['bash', '-lc', setup + shlex.join(subscriber_args)],
                                          cwd=folder, stdout=subprocess.PIPE, stderr=topic_err,
                                          text=True, encoding='utf-8', start_new_session=True)
            if not select.select([subscriber.stdout], [], [], min(15, timeout))[0]:
                raise RuntimeError('ROS结果订阅器启动超时；查看vlm_topic_stderr.log')
            ready = subscriber.stdout.readline().strip()
            if ready != 'READY':
                raise RuntimeError('ROS结果订阅器未就绪；查看vlm_topic_stderr.log：' + ready)

            model = subprocess.Popen(['bash', '-lc', setup + shlex.join(model_args)],
                                     cwd=folder, stdout=model_out, stderr=model_err,
                                     start_new_session=True)
            # The VLM node may remain alive after producing a single answer.
            model_exited_at = None
            while subscriber.poll() is None:
                if time.monotonic() - start >= timeout:
                    timed_out = True
                    break
                if model.poll() is not None:
                    if model_exited_at is None:
                        model_exited_at = time.monotonic()
                    elif time.monotonic() - model_exited_at > 2:
                        break
                time.sleep(0.2)
            if subscriber.poll() is not None:
                remainder = subscriber.stdout.read()
                for line in remainder.splitlines():
                    try:
                        payload = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(payload, dict) and payload.get('topic') == topic and isinstance(payload.get('answer'), str):
                        topic_payload = payload
                        break
        finally:
            stop_group(model)
            stop_group(subscriber)
            if subscriber and subscriber.stdout:
                subscriber.stdout.close()
    if topic_payload:
        (folder / 'vlm_topic.json').write_text(json.dumps(topic_payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'candidate': extract_answer_json(topic_payload['answer']) if topic_payload else None,
            'raw_answer': topic_payload['answer'] if topic_payload else None,
            'result_topic': topic, 'subscriber_exit_code': subscriber.returncode if subscriber else None,
            'model_exit_code': model.returncode if model else None, 'timed_out': timed_out,
            'command_argv': model_args}
