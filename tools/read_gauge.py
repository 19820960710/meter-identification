"""Run the image-bound prototype or click scale anchors for a new photograph."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
from datetime import datetime
import uuid
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from meter_reader import ReadingError, draw_result, read_image


def configure_scale(template):
    import tkinter as tk
    from tkinter import ttk, messagebox
    window = tk.Tk()
    window.title('填写这张照片的实际刻度')
    window.attributes('-topmost', True)
    fields = {}
    defaults = [('ticks', '依次标定的刻度值', ' '.join(f'{t["value"]:g}' for t in template['ticks'])),
                ('unit', '单位', template['unit']),
                ('division', '最小分度（可留空）', str(template.get('minimum_division') or ''))]
    ttk.Label(window, text='按照片上的实际标值填写，再点击对应刻度线。').grid(row=0, columnspan=2, padx=15, pady=12)
    for row, (name, label, value) in enumerate(defaults, 1):
        ttk.Label(window, text=label).grid(row=row, column=0, padx=12, pady=8, sticky='w')
        entry = ttk.Entry(window, width=35)
        entry.insert(0, value)
        entry.grid(row=row, column=1, padx=12, pady=8)
        fields[name] = entry
    direction = tk.StringVar(value='顺时针' if template['direction'] == 'clockwise' else '逆时针')
    ttk.Label(window, text='数值增加方向').grid(row=4, column=0, padx=12, pady=8, sticky='w')
    ttk.Combobox(window, textvariable=direction, values=['顺时针', '逆时针'], state='readonly').grid(row=4, column=1)
    scale_kind = tk.StringVar(value='整圈外圈（格）' if template.get('scale_mode') == 'full_circle_uniform' else '非整圈刻度弧')
    ttk.Label(window, text='刻度类型').grid(row=5, column=0, padx=12, pady=8, sticky='w')
    kind_widget = ttk.Combobox(window, textvariable=scale_kind, values=['非整圈刻度弧', '整圈外圈（格）'], state='readonly')
    kind_widget.grid(row=5, column=1)
    ttk.Label(window, text='整圈模式只读外圈长针格数，不合并计圈针。').grid(row=6, columnspan=2, padx=12, pady=5)

    def update_kind(event=None):
        fields['unit'].configure(state='normal')
        fields['unit'].delete(0, 'end')
        if scale_kind.get() == '整圈外圈（格）':
            fields['unit'].insert(0, 'div')
            fields['unit'].configure(state='readonly')
            if fields['ticks'].get() == defaults[0][2] and template.get('scale_mode') != 'full_circle_uniform':
                fields['ticks'].delete(0, 'end')
                fields['ticks'].insert(0, '0 10 20 30 40 50 60 70 80 90 100')
            fields['division'].delete(0, 'end')
            fields['division'].insert(0, '1')
        else:
            fields['unit'].insert(0, template['unit'])

    kind_widget.bind('<<ComboboxSelected>>', update_kind)
    if template.get('scale_mode') == 'full_circle_uniform':
        update_kind()
    selected = []

    def accept():
        try:
            values = [float(v) for v in fields['ticks'].get().replace(',', ' ').split()]
            if len(values) < 2 or not all(math.isfinite(v) for v in values) or any(b <= a for a, b in zip(values, values[1:])):
                raise ValueError('刻度值至少两个，必须有限且依次增大。')
            unit = fields['unit'].get().strip()
            if not unit:
                raise ValueError('请填写单位。')
            text = fields['division'].get().strip()
            division = float(text) if text else None
            if division is not None and (not math.isfinite(division) or division <= 0):
                raise ValueError('最小分度应为正数。')
            full_circle = scale_kind.get() == '整圈外圈（格）'
            if full_circle and len(values) < 7:
                raise ValueError('整圈模式至少填6个不同方向的刻度和闭合终点，例如0 10 … 100。')
            selected.append({'ticks': [{'value': v} for v in values], 'unit': unit,
                             'minimum_division': division,
                             'scale_mode': 'full_circle_uniform' if full_circle else 'partial',
                             'needle_role': 'outer_long_hand' if full_circle else None,
                             'direction': 'clockwise' if direction.get() == '顺时针' else 'counterclockwise'})
            window.destroy()
        except ValueError as error:
            messagebox.showerror('请检查填写内容', str(error), parent=window)

    ttk.Button(window, text='开始标定', command=accept).grid(row=7, columnspan=2, pady=15)
    window.mainloop()
    if not selected:
        raise ReadingError('Scale configuration cancelled.')
    return selected[0]


def load_profile(path):
    def invalid_constant(value):
        raise ReadingError(f'Non-finite number in calibration: {value}')
    profile = json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=invalid_constant)
    if not isinstance(profile, dict):
        raise ReadingError('Calibration JSON must be an object.')
    if not isinstance(profile.get('image'), str) or not profile['image'].strip():
        raise ReadingError('Calibration requires a valid image path string.')
    if not isinstance(profile.get('ticks'), list) or len(profile['ticks']) < 2:
        raise ReadingError('Calibration requires at least two scale anchors.')
    if not isinstance(profile.get('unit'), str) or not profile['unit'].strip():
        raise ReadingError('Calibration requires a unit.')
    if profile.get('direction') not in ('clockwise', 'counterclockwise'):
        raise ReadingError('Calibration requires a valid reading direction.')
    return profile


def save_record(destination, run_directory, result):
    result['record_file'] = str(run_directory / 'result.json')
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    (run_directory / 'result.json').write_text(text, encoding='utf-8')
    # The root record always refers to this invocation, even if calibration fails.
    (destination / 'result.json').write_text(text, encoding='utf-8')


def choose_image():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    filename = filedialog.askopenfilename(
        title='选择仪表原图（单针或整圈外圈）', initialdir=str(ROOT),
        filetypes=[('仪表照片', '*.jpg *.jpeg *.png *.bmp *.webp'), ('所有文件', '*.*')],
    )
    root.destroy()
    if not filename:
        raise ReadingError('Image selection cancelled.')
    return Path(filename)


def calibrate(image_path, template):
    raw = image_path.read_bytes()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ReadingError('Image cannot be decoded.')
    ratio = min(1., 1000 / image.shape[1], 800 / image.shape[0])
    base = cv2.resize(image, None, fx=ratio, fy=ratio)
    values = [float(t['value']) for t in template['ticks']]
    points = []
    window = 'Scale calibration: click tick lines; U undo; ENTER save; ESC cancel'

    def click(event, x, y, flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(values):
            points.append([float(x / ratio), float(y / ratio)])

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, click)
    try:
        while True:
            canvas = base.copy()
            for index, point in enumerate(points):
                xy = tuple(int(round(v * ratio)) for v in point)
                cv2.circle(canvas, xy, 5, (255, 0, 255), -1)
                cv2.putText(canvas, f'{values[index]:g}', (xy[0] + 8, xy[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 0, 255), 2)
            prompt = (f'Click scale line for {values[len(points)]:g} {template["unit"]}'
                      if len(points) < len(values) else 'Press ENTER to save, U to undo')
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (255, 255, 255), -1)
            cv2.putText(canvas, prompt, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(40) & 255
            if key in (27, ord('q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise ReadingError('Calibration cancelled.')
            if key in (ord('u'), ord('U')) and points:
                points.pop()
            if key in (10, 13) and len(points) == len(values):
                break
    finally:
        cv2.destroyWindow(window)
    return {
        'image': str(image_path.resolve()), 'image_sha256': hashlib.sha256(raw).hexdigest(),
        'image_size': list(image.shape[1::-1]), 'unit': template['unit'],
        'minimum_division': template.get('minimum_division'),
        'direction': template['direction'],
        'scale_mode': template.get('scale_mode', 'partial'),
        'needle_role': template.get('needle_role'),
        'ticks': [{'value': v, 'point': p} for v, p in zip(values, points)],
        'source': 'user_clicked_visible_scale_lines_for_this_image',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='Single-image gauge reading prototype')
    parser.add_argument('--image', type=Path)
    parser.add_argument('--calibration', type=Path, default=ROOT / 'configs/gauges/demo_832.json')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/annotation/meter_labeler_v3/best.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/reading-demo')
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--choose-image', action='store_true')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    destination = args.output.resolve()
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    run_directory = destination / run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    image_path = args.image
    try:
        profile = load_profile(args.calibration)
        image_path = choose_image() if args.choose_image else args.image or Path(profile['image'])
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        calibration_path = args.calibration.resolve()
        if args.choose_image:
            fingerprint = hashlib.sha256(image_path.read_bytes()).hexdigest()[:16]
            calibration_path = ROOT / 'configs/gauges' / f'photo_{fingerprint}.json'
            if calibration_path.exists() and not args.calibrate:
                profile = load_profile(calibration_path)
            else:
                profile = calibrate(image_path, configure_scale(profile))
                calibration_path.parent.mkdir(parents=True, exist_ok=True)
                calibration_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        elif args.calibrate:
            profile = calibrate(image_path, profile)
            fingerprint = profile['image_sha256'][:16]
            calibration_path = ROOT / 'configs/gauges' / f'photo_{fingerprint}.json'
            calibration_path.parent.mkdir(parents=True, exist_ok=True)
            calibration_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        result = read_image(image_path, profile, args.model)
        result['calibration_file'] = str(calibration_path)
        result.update(run_id=run_id, annotated_image=None)
        image = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR) if image_path.is_file() else None
        if image is not None:
            picture_path = run_directory / 'annotated.png'
            ok, encoded = cv2.imencode('.png', draw_result(image, result))
            if not ok:
                raise ReadingError('Could not encode evidence picture.')
            picture_path.write_bytes(encoded.tobytes())
            result['annotated_image'] = str(picture_path)
        save_record(destination, run_directory, result)
        print('状态：' + result['status'])
        if result['value'] is not None:
            print('估计读数：' + result['display_value'])
            outliers = result.get('calibration_consistency', {}).get('outlier_anchor_values', [])
            if outliers:
                print('标定提示：' + ', '.join(f'{v:g}' for v in outliers) + ' 格点击位置有偏差，已标灰供复查；结果是原型估计。')
        else:
            print('需要复核：' + result.get('reason', 'unknown'))
        print('记录：' + result['record_file'])
        if result['annotated_image']:
            print('标记图：' + str(picture_path))
            if args.show:
                try:
                    os.startfile(str(picture_path))
                except OSError as error:
                    print('结果图已保存，请手动打开：' + str(error))
        return 0 if result['value'] is not None else 2
    except (ReadingError, OSError, ValueError, TypeError, KeyError, cv2.error) as error:
        failed = {'status': 'needs_review', 'value': None, 'unit': None, 'reason': str(error),
                  'image': str(image_path) if image_path else None, 'run_id': run_id,
                  'annotated_image': None, 'calibration_file': str(args.calibration.resolve()),
                  'operating_status': 'not_evaluated'}
        save_record(destination, run_directory, failed)
        print('需要复核：' + str(error))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
