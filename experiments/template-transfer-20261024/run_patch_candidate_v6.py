"""Template-assisted current neighboring tick localization, on sealed v2 poses."""
from pathlib import Path
import bisect
import copy
import hashlib
import json
import math
import sys

import cv2
import numpy as np

from template_registration import reference_ring, project, rectify_for_reading
from template_registration_v2 import feature_mask
from patch_locator_v6 import locate_anchor, fixed_similarity, SETTINGS

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/template-transfer-20261024-v2'
OUT = ROOT / 'artifacts/template-transfer-20261024-v6'
sys.path.insert(0, str(ROOT / 'tools'))
from read_meter import read_meter
from meter_reader import draw_result


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(path):
    return cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), 1)


def geometry_interval(profile, r, radius):
    if any(f != 'unsupported_scale_anchors' for f in r['quality']['failures']):
        raise ValueError('A registration gate other than outer support failed.')
    inverse = np.linalg.inv(np.array(r['homography']))
    motion = r['target_motion']
    c, tip = np.array(motion['center']), np.array(motion['endpoint'])
    u = (tip-c)/np.linalg.norm(tip-c)
    points = project([c, c+u*motion['panel_radius']*.35], inverse)
    if np.linalg.norm(points[0]-r['source_motion']['center']) > radius*.04:
        raise ValueError('Current axis mapping is inconsistent.')
    fits = [m['reference_point'] for m in r['matches'] if m['split'] == 'fit']
    hull = cv2.convexHull(np.array(fits, np.float32)[np.array(r['fit_inliers'], bool)])
    if any(cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), True) < 0 for p in points):
        raise ValueError('Current axis or near-shaft is outside spatial fit support.')
    phi = lambda p: math.degrees(math.atan2(points[0][1]-p[1], p[0]-points[0][0])) % 360
    zero = phi(profile['ticks'][0]['point'])
    positions = [(zero-phi(t['point'])) % 360 for t in profile['ticks']]
    theta = (zero-phi(points[1])) % 360
    if any(b <= a for a, b in zip(positions, positions[1:])) or theta > positions[-1]:
        raise ValueError('Current needle is outside source scale arc.')
    i = min(len(positions)-2, max(0, bisect.bisect_right(positions, theta)-1))
    return [i, i+1]


def verify_outliers(reference, rectified, r, radius, source_valid, target_valid):
    holds = [m for m in r['matches'] if m['split'] == 'holdout']
    errors = r['quality']['metrics']['holdout_errors_px']
    inverse = np.linalg.inv(np.array(r['homography']))
    checks = []
    for match, error in zip(holds, errors):
        if error <= 4:
            continue
        p = match['reference_point']
        wrong = project([match['target_point']], inverse)[0]
        predicted = fixed_similarity(reference, rectified, p, p, radius, source_valid, target_valid)
        descriptor = fixed_similarity(reference, rectified, p, wrong, radius, source_valid, target_valid)
        passed = (predicted.get('valid', False) and descriptor.get('valid', False)
            and predicted['gray_ncc'] >= SETTINGS['gray_ncc_min']
            and predicted['gradient_ncc'] >= SETTINGS['gradient_ncc_min']
            and predicted['combined']-descriptor['combined'] >= SETTINGS['peak_margin_min'])
        checks.append({'original_descriptor_error_px': error, 'source_point': p,
            'descriptor_point_in_reference_plane': wrong.tolist(), 'fixed_H_predicted_patch': predicted,
            'original_descriptor_patch': descriptor, 'alias_image_check_passed': bool(passed)})
    return checks


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if (OUT / 'registration/raw-results.json').exists():
        raise FileExistsError('Preserve the first local-patch candidate pass.')
    manifest = read(OUT / 'inputs/inference-manifest.json')
    profile = read(OUT / 'inputs/reference-763.json')
    raw = Path(manifest['reference_image']).read_bytes()
    reference, center, radius = reference_ring(raw, profile)
    cases = read(BASE / 'registration/raw-results.json')['rows']
    source = next(c for c in cases if c['id'] == '763')
    source_valid = feature_mask(reference, source['proposals'][0], source['result']['source_motion'], (center, radius))
    protected = read(OUT / 'protected-before.json')
    assert all(sha(p) == h for p, h in protected.items())
    dump(OUT / 'inputs/predeclared-policy.json', {'patch_settings': SETTINGS,
        'H_and_old_holdout_errors_unchanged': True,
        'current_interval_selected_from_image_direction_before_patch_localization': True,
        'controls_require_current_two_anchors_not_six_full_scale': True,
        'all_outlier_descriptors_above_4px_need_fixed_H_image_evidence': True,
        'target_profiles_and_human_references_not_loaded': True,
        'scope': 'automatic image-localized current neighboring tick candidates; no full-scale claim'})
    registrations = []
    for case in cases:
        r = copy.deepcopy(case['result'])
        r['base_v2_status'] = r['status']
        r.update(status='needs_review', reason=r.get('reason'))
        if r.get('homography') is not None and not any(
                f != 'unsupported_scale_anchors' for f in r['quality']['failures']):
            try:
                indices = geometry_interval(profile, r, radius)
                h = np.array(r['homography'])
                inverse = np.linalg.inv(h)
                target = decode(case['image'])
                rectified = cv2.warpPerspective(target, inverse, tuple(profile['image_size']))
                target_mask = feature_mask(target, r['target_motion']['panel'], r['target_motion'])
                target_valid = cv2.warpPerspective(target_mask, inverse, tuple(profile['image_size']), flags=cv2.INTER_NEAREST)
                ticks = []
                c = r['source_motion']['center']
                angle = lambda p: math.degrees(math.atan2(c[1]-p[1], p[0]-c[0])) % 360
                for tick in profile['ticks']:
                    located = locate_anchor(reference, rectified, tick['point'], radius, source_valid, target_valid)
                    located['value'] = tick['value']
                    if located['passed']:
                        change = abs((angle(located['target_point'])-angle(tick['point'])+180) % 360-180)
                        located['angular_adjustment_degrees'] = change
                        if change > SETTINGS['angular_adjustment_degrees_max']:
                            located.update(passed=False, reason='Local tick adjustment exceeds fixed angular limit.')
                    ticks.append(located)
                outliers = verify_outliers(reference, rectified, r, radius, source_valid, target_valid)
                r.update(current_interval_indices=indices, localized_ticks=ticks, heldout_alias_checks=outliers)
                if all(ticks[i]['passed'] for i in indices) and all(t['alias_image_check_passed'] for t in outliers):
                    r.update(status='registration_candidate', reason=None,
                             acceptance_policy='direct_local_patch_current_interval_v6')
                else:
                    r['reason'] = 'neighbor_tick_patch_or_heldout_alias_check_failed'
            except (ValueError, KeyError, TypeError, cv2.error, np.linalg.LinAlgError) as error:
                r['reason'] = str(error)
        record = {**case, 'result': r}
        registrations.append(record)
        dump(OUT / 'registration' / (case['id']+'.json'), record)
        print(json.dumps({'id': case['id'], 'status': r['status'], 'reason': r['reason'],
            'interval': r.get('current_interval_indices'),
            'ticks': [{'value': t['value'], 'pass': t['passed'], 'reason': t['reason'],
                       'gray': t.get('gray_ncc'), 'grad': t.get('gradient_ncc'), 'gap': t.get('peak_margin')}
                      for t in r.get('localized_ticks', [])], 'outliers': r.get('heldout_alias_checks')}, ensure_ascii=False), flush=True)
    dump(OUT / 'registration/raw-results.json', {'rows': registrations,
        'target_calibrations_loaded': False, 'human_reference_values_loaded': False})
    reg_sha = sha(OUT / 'registration/raw-results.json')
    controls_passed = all(next(c for c in registrations if c['id'] == sid)['result']['status'] == 'registration_candidate'
                          for sid in ['763', 'synthetic_known_H'])
    readings = []
    for case in registrations:
        r = case['result']
        row = {'id': case['id'], 'role': case['role'], 'registration_status': r['status'],
               'status': 'needs_review', 'value': None, 'unit': 'MPa', 'reason': r['reason']}
        if controls_passed and r['status'] == 'registration_candidate' and case['role'] in ('identity_control', 'user_target'):
            folder = OUT / 'readings' / case['id']
            folder.mkdir()
            image_path = folder / 'rectified-current-target.png'
            data, runtime = rectify_for_reading(raw, decode(case['image']), profile, r, image_path)
            indices = r['current_interval_indices']
            runtime['ticks'] = [{'value': profile['ticks'][i]['value'], 'point': r['localized_ticks'][i]['target_point']} for i in indices]
            runtime.update(source='experimental_automatic_local_tick_patch_correspondence_from_763',
                source_template_calibration_sha256=manifest['reference_profile_sha256'],
                target_original_image_sha256=case['image_sha256'],
                supported_current_interval=[t['value'] for t in runtime['ticks']],
                source_anchor_points=[profile['ticks'][i]['point'] for i in indices],
                coordinate_frame='rectified_current_target_pixel_plane',
                scope='automatic current neighboring ticks only; not full-scale or another frame')
            image_path.write_bytes(data)
            dump(folder / 'runtime-template-profile.json', runtime)
            result = read_meter(image_path, folder/'runtime-template-profile.json', manifest['model'])
            if result['value'] is not None and (not runtime['ticks'][0]['value'] <= result['value'] <= runtime['ticks'][-1]['value']
                    or result.get('association_diagnostic', {}).get('center_selection') != 'single_center_candidate'):
                result.update(status='needs_review', value=None, reason='Current interval or unique image axis not verified.')
            result['supported_current_interval'] = runtime['supported_current_interval']
            dump(folder/'result.json', result)
            (folder/'annotated.png').write_bytes(cv2.imencode('.png', draw_result(decode(image_path), result))[1].tobytes())
            row.update(status=result['status'], value=result['value'], reason=result.get('reason'),
                       supported_current_interval=runtime['supported_current_interval'], result_json=str(folder/'result.json'))
        elif not controls_passed and case['role'] in ('identity_control', 'user_target'):
            row['reason'] = 'Positive control failed; local patch policy not accepted.'
        if case['role'] not in ('identity_control', 'user_target'):
            row['status'] = 'control_registration_only'
        readings.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    dump(OUT/'readings/raw-results.json', {'rows': readings, 'positive_controls_passed': controls_passed,
        'target_calibrations_loaded': False, 'human_reference_values_loaded': False})
    assert sha(OUT/'registration/raw-results.json') == reg_sha
    assert all(sha(p) == h for p, h in protected.items())
    dump(OUT/'prediction-seal.json', {'registration_sha256': reg_sha,
        'readings_sha256': sha(OUT/'readings/raw-results.json'), 'sealed_before_evaluation': True})


if __name__ == '__main__':
    main()
