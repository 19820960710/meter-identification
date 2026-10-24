"""Standalone experimental reader: one registered 763 template, no target manual calibration."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'tools'))
from template_registration import reference_ring, project, rectify_for_reading
from template_registration_v2 import register_partial_v2, feature_mask
from patch_locator_v6 import locate_anchor, SETTINGS
from full_anchor_validation import validate_full_anchors
from run_patch_candidate_v6 import geometry_interval, verify_outliers
from read_meter import read_meter
from meter_reader import draw_result


def calibration_path_allowed(path, reference_profile, active_directory):
    p = Path(path).resolve()
    directory = Path(active_directory).resolve()
    return not p.is_relative_to(directory) or p == Path(reference_profile).resolve()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(path):
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), 1)
    if image is None:
        raise ValueError('Image cannot be decoded.')
    return image


def detect(model, image):
    pred = model.predict(image, imgsz=640, device='cpu', conf=.25, iou=.45,
                         agnostic_nms=False, max_det=300, verbose=False)[0]
    return [{'label': pred.names[int(b.cls.item())], 'score': float(b.conf.item()),
             'box': b.xyxy[0].tolist()} for b in pred.boxes]


def run(image_path, reference_path, model_path, destination):
    profile = read(reference_path)
    source_path = Path(profile['image'])
    if not source_path.is_absolute():
        source_path = ROOT/source_path
    reference_raw = source_path.read_bytes()
    reference, center, radius = reference_ring(reference_raw, profile)
    target = decode(image_path)
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    sd, td = detect(model, reference), detect(model, target)
    panels = [d['box'] for d in sd if d['label'] == 'panel']
    if len(panels) != 1:
        raise ValueError('Template coarse panel is not unique.')
    proposals = [d['box'] for d in td if d['label'] == 'panel']
    registration = register_partial_v2(reference_raw, target, profile, panels[0], proposals, sd, td)
    trace = {'template_image': str(source_path), 'template_profile': str(reference_path),
        'template_profile_sha256': sha(reference_path), 'target_image': str(image_path),
        'target_image_sha256': sha(image_path), 'source_detections': sd, 'target_detections': td,
        'registration': registration, 'target_manual_profile_loaded': False}
    try:
        if registration.get('homography') is None:
            raise ValueError(registration['reason'])
        geometry_interval(profile, registration, radius)
        inverse = np.linalg.inv(np.array(registration['homography']))
        rectified = cv2.warpPerspective(target, inverse, tuple(profile['image_size']))
        source_valid = feature_mask(reference, panels[0], registration['source_motion'], (center, radius))
        target_valid = cv2.warpPerspective(feature_mask(target, registration['target_motion']['panel'],
            registration['target_motion']), inverse, tuple(profile['image_size']), flags=cv2.INTER_NEAREST)
        located = []
        c = registration['source_motion']['center']
        angle = lambda p: float(np.degrees(np.arctan2(c[1]-p[1], p[0]-c[0]))) % 360
        for tick in profile['ticks']:
            point = locate_anchor(reference, rectified, tick['point'], radius, source_valid, target_valid)
            point['value'] = tick['value']
            if point['passed']:
                change = abs((angle(point['target_point'])-angle(tick['point'])+180) % 360-180)
                point['angular_adjustment_degrees'] = change
                if change > SETTINGS['angular_adjustment_degrees_max']:
                    point.update(passed=False, reason='Angular adjustment limit exceeded.')
            located.append(point)
        alias = verify_outliers(reference, rectified, registration, radius, source_valid, target_valid)
        axis = project([registration['target_motion']['center']], inverse)[0]
        full = validate_full_anchors(profile['ticks'], located, axis, profile['image_size'])
        trace.update(localized_ticks=located, heldout_alias_checks=alias, six_anchor_check=full)
        if not full['passed'] or not all(r['alias_image_check_passed'] for r in alias):
            raise ValueError(full['reason'] or 'Held-out image alias check failed.')
        approved = {**registration, 'status': 'registration_candidate',
                    'acceptance_policy': 'six_direct_image_anchor_correspondences_v7'}
        image_out = destination/'rectified-current-target.png'
        data, runtime = rectify_for_reading(reference_raw, target, profile, approved, image_out)
        runtime.update(ticks=full['ticks'], source='automatic_template_image_anchor_correspondences',
            source_original_ticks=profile['ticks'], source_template_calibration_sha256=sha(reference_path),
            target_original_image_sha256=sha(image_path), coordinate_frame='rectified_current_target_pixel_plane',
            scope='registered template experimental candidate; full-range physical accuracy unverified')
        image_out.write_bytes(data)
        dump(destination/'runtime-template-profile.json', runtime)
        result = read_meter(image_out, destination/'runtime-template-profile.json', model_path)
        if result['value'] is not None and result.get('association_diagnostic', {}).get('center_selection') != 'single_center_candidate':
            result.update(status='needs_review', value=None, reason='Unique current image axis not established.')
        picture = destination/'annotated.png'
        picture.write_bytes(cv2.imencode('.png', draw_result(decode(image_out), result))[1].tobytes())
        result.update(annotated_image=str(picture), target_original_image=str(image_path),
            target_original_sha256=sha(image_path), template_profile=str(reference_path),
            automatic_tick_count=6, target_manual_profile_loaded=False, development_only=True)
    except (ValueError, KeyError, TypeError, RuntimeError, cv2.error) as error:
        result = {'status': 'needs_review', 'value': None, 'unit': 'MPa', 'reason': str(error),
                  'target_original_image': str(image_path), 'target_manual_profile_loaded': False,
                  'operating_status': 'not_evaluated'}
    dump(destination/'trace.json', trace)
    dump(destination/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--reference-profile', type=Path, default=ROOT/'configs/gauges/photo_13e7265b20639756.json')
    parser.add_argument('--model', type=Path, default=ROOT/'models/annotation/meter_labeler_v3/best.pt')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/template-photo-runs')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if sys.stdout is not None and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    opened = set()

    def audit(event, values):
        if event != 'open' or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(values[0])).resolve()
        if p.is_relative_to(ROOT/'configs/gauges'):
            if not calibration_path_allowed(p, args.reference_profile, ROOT/'configs/gauges'):
                raise PermissionError('Only the selected reference calibration may be opened.')
            opened.add(str(p))

    sys.addaudithook(audit)
    if args.check:
        p = read(args.reference_profile)
        source = Path(p['image'])
        if not source.is_absolute():
            source = ROOT/source
        reference_ring(source.read_bytes(), p)
        if args.image:
            decode(args.image)
        print(json.dumps({'reference_valid': True, 'target_profile_required': False,
                          'model_inference': False, 'active_profiles_opened': sorted(opened)}, ensure_ascii=False))
        return 0
    if args.image is None:
        import tkinter as tk
        from tkinter import filedialog
        window = tk.Tk(); window.withdraw(); window.attributes('-topmost', True)
        name = filedialog.askopenfilename(title='选择同表型仪表照片（复用763模板，无需点刻度）',
            initialdir=str(ROOT/'samples/images'),
            filetypes=[('仪表照片', '*.jpg *.jpeg *.png *.bmp *.webp')])
        window.destroy()
        if not name:
            return 2
        args.image = Path(name)
    destination = args.output/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6])
    destination.mkdir(parents=True, exist_ok=False)
    print('正在复用763模板识读，无需点击刻度，请稍候……', flush=True)
    try:
        result = run(args.image.resolve(), args.reference_profile.resolve(), args.model.resolve(), destination)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, cv2.error) as error:
        result = {'status': 'needs_review', 'value': None, 'unit': 'MPa', 'reason': str(error),
                  'target_original_image': str(args.image.resolve()),
                  'target_manual_profile_loaded': False, 'operating_status': 'not_evaluated'}
    result['active_gauge_profiles_opened'] = sorted(opened)
    dump(destination/'result.json', result)
    print(json.dumps({'status': result['status'], 'value': result['value'], 'unit': result['unit'],
                      'reason': result.get('reason'), 'result': str(destination/'result.json'),
                      'active_gauge_profiles_opened': sorted(opened)}, ensure_ascii=False))
    if args.show and result.get('annotated_image'):
        os.startfile(result['annotated_image'])
    return 0 if result['value'] is not None else 2


if __name__ == '__main__':
    raise SystemExit(main())
