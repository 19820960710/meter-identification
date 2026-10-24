"""Isolated partial-scale template registration; never writes active calibrations."""
import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import meter_reader as meter

_spec = importlib.util.spec_from_file_location('legacy_registration_quality',
    ROOT / 'experiments/calibration-transfer-20261022/registration.py')
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)
SETTINGS = copy.deepcopy(_legacy.SETTINGS)
project = _legacy.project
quality_check = _legacy.quality_check


def reference_ring(raw, profile):
    meter.validate_calibration_image(raw, profile)
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), 1)
    if image is None or profile['image_size'] != list(image.shape[1::-1]):
        raise ValueError('Reference image dimensions do not match calibration.')
    if (profile.get('unit') != 'MPa' or profile.get('scale_mode', 'partial') != 'partial'
            or profile.get('direction') != 'clockwise'
            or [t['value'] for t in profile['ticks']] != [0, 5, 10, 15, 20, 25]):
        raise ValueError('This isolated experiment requires the prepared 0-25 MPa partial scale.')
    pts = np.array([t['point'] for t in profile['ticks']], float)
    width, height = profile['image_size']
    if (not np.isfinite(pts).all() or pts.shape != (6, 2) or
            not ((pts[:, 0] >= 0) & (pts[:, 0] < width) &
                 (pts[:, 1] >= 0) & (pts[:, 1] < height)).all()):
        raise ValueError('Invalid reference anchors.')
    matrix = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
    if np.linalg.matrix_rank(matrix) != 3:
        raise ValueError('Reference ring is degenerate.')
    cx, cy, constant = np.linalg.lstsq(matrix, (pts ** 2).sum(axis=1), rcond=None)[0]
    radius_squared = constant + cx * cx + cy * cy
    if not math.isfinite(radius_squared) or radius_squared <= 25:
        raise ValueError('Reference ring radius is not usable.')
    radius = math.sqrt(radius_squared)
    residual = np.abs(np.linalg.norm(pts - [cx, cy], axis=1) - radius)
    if float(residual.max()) > max(3, radius * .08):
        raise ValueError('Reference anchors do not form a near-circular dial.')
    meter.angle_reading([cx, cy], profile['ticks'][1]['point'], profile['ticks'], 'clockwise')
    return image, [float(cx), float(cy)], radius


def exclude_boxes(mask, boxes):
    height, width = mask.shape
    for box in boxes:
        if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
            raise ValueError('Invalid moving-region exclusion box.')
        x1, y1, x2, y2 = box
        mask[max(0, math.floor(y1)):min(height, math.ceil(y2)),
             max(0, math.floor(x1)):min(width, math.ceil(x2))] = 0
    return mask


def static_mask(size, center, radius, panel, exclusions):
    width, height = size
    yy, xx = np.indices((height, width))
    rr = np.hypot(xx - center[0], yy - center[1])
    mask = (((rr >= radius * .65) & (rr <= radius * 1.20)) &
            (xx >= panel[0]) & (xx <= panel[2]) & (yy >= panel[1]) & (yy <= panel[3])).astype(np.uint8) * 255
    return exclude_boxes(mask, exclusions)


def register_partial(reference_raw, target, profile, panel, proposals,
                     source_exclusions=(), target_exclusions=()):
    out = {'status': 'needs_review', 'reason': None, 'numeric_value': None,
           'settings': copy.deepcopy(SETTINGS), 'homography': None,
           'matches': [], 'quality': None, 'active_profile_written': False,
           'matching_region': 'reference_outer_annulus_with_detected_needle_and_axle_rois_excluded'}
    try:
        reference, center, radius = reference_ring(reference_raw, profile)
        if not isinstance(target, np.ndarray) or target.ndim != 3 or target.shape[2] != 3 or target.dtype != np.uint8:
            raise ValueError('Invalid target image.')
        rh, rw = reference.shape[:2]
        if not (0 <= panel[0] < panel[2] <= rw and 0 <= panel[1] < panel[3] <= rh):
            raise ValueError('Invalid reference panel.')
        mask = static_mask([rw, rh], center, radius, panel, source_exclusions)
        tm = np.zeros(target.shape[:2], np.uint8)
        for box in proposals:
            x1, y1, x2, y2 = box
            tm[max(0, math.floor(y1)):min(target.shape[0], math.ceil(y2)),
               max(0, math.floor(x1)):min(target.shape[1], math.ceil(x2))] = 255
        exclude_boxes(tm, target_exclusions)
        out.update(template_image_sha256=hashlib.sha256(reference_raw).hexdigest(),
                   target_pixel_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
                   reference_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                   target_mask_sha256=hashlib.sha256(tm.tobytes()).hexdigest(),
                   source_exclusions=[list(b) for b in source_exclusions],
                   target_exclusions=[list(b) for b in target_exclusions])
        cv2.setRNGSeed(SETTINGS['seed'])
        cv2.setNumThreads(1)
        sift = cv2.SIFT_create(nfeatures=SETTINGS['sift_features'])
        sk, sd = sift.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), mask)
        tk, td = sift.detectAndCompute(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), tm)
        out.update(reference_features=len(sk), target_features=len(tk))
        if sd is None or td is None or min(len(sd), len(td)) < 2:
            raise ValueError('Insufficient static local features.')
        bf = cv2.BFMatcher(cv2.NORM_L2)

        def filtered(a, b):
            return {p[0].queryIdx: p[0].trainIdx for p in bf.knnMatch(a, b, k=2)
                    if len(p) == 2 and p[0].distance < SETTINGS['ratio'] * p[1].distance}

        forward, backward = filtered(sd, td), filtered(td, sd)
        pairs = [(i, j) for i, j in forward.items() if backward.get(j) == i]
        pairs.sort(key=lambda p: (sk[p[0]].pt[0], sk[p[0]].pt[1], p[0], p[1]))
        out['mutual_matches'] = len(pairs)
        out['matches'] = [{'reference_point': list(sk[i].pt), 'target_point': list(tk[j].pt),
                           'split': 'holdout' if k % 3 == 0 else 'fit'}
                          for k, (i, j) in enumerate(pairs)]
        if len(pairs) < SETTINGS['minimum_matches']:
            raise ValueError('Insufficient mutual ratio-filtered static correspondences.')
        src = np.array([sk[i].pt for i, j in pairs], float)
        dst = np.array([tk[j].pt for i, j in pairs], float)
        hold = np.arange(len(pairs)) % 3 == 0
        h, inliers = cv2.findHomography(src[~hold], dst[~hold], cv2.RANSAC,
            SETTINGS['ransac_px'], maxIters=SETTINGS['ransac_iterations'], confidence=SETTINGS['ransac_confidence'])
        if h is None or not np.isfinite(h).all() or abs(h[2, 2]) < 1e-9 or abs(np.linalg.det(h)) < 1e-10:
            raise ValueError('No nonsingular homography.')
        h /= h[2, 2]
        out.update(homography=h.tolist(), fit_inliers=inliers.reshape(-1).astype(bool).tolist())
        quality = quality_check(h, src[~hold], dst[~hold], inliers, src[hold], dst[hold],
            panel, list(target.shape[1::-1]), proposals, [t['point'] for t in profile['ticks']])
        out['quality'] = quality
        out['projected_ticks_for_diagnosis'] = [{'value': t['value'], 'point': p.tolist()}
            for t, p in zip(profile['ticks'], project([t['point'] for t in profile['ticks']], h))]
        if quality['failures']:
            out['reason'] = 'registration_quality_rejected'
        else:
            out.update(status='registration_candidate', reason=None)
    except (ValueError, KeyError, TypeError, RuntimeError, np.linalg.LinAlgError, cv2.error) as error:
        out['reason'] = str(error)
    return out


def rectify_for_reading(reference_raw, target, profile, registration, destination):
    if registration.get('status') != 'registration_candidate':
        raise ValueError('Registration quality must pass before rectification.')
    if (registration['template_image_sha256'] != hashlib.sha256(reference_raw).hexdigest()
            or registration['target_pixel_sha256'] != hashlib.sha256(target.tobytes()).hexdigest()):
        raise ValueError('Registration belongs to another source or target.')
    reference_ring(reference_raw, profile)
    h = np.array(registration['homography'], float)
    warped = cv2.warpPerspective(target, np.linalg.inv(h), tuple(profile['image_size']))
    ok, encoded = cv2.imencode('.png', warped)
    if not ok:
        raise ValueError('Unable to encode rectified target.')
    raw = encoded.tobytes()
    runtime = {k: copy.deepcopy(profile.get(k)) for k in
               ('ticks', 'image_size', 'unit', 'minimum_division', 'direction', 'scale_mode', 'needle_role')}
    runtime.update(image=str(Path(destination).resolve()), image_sha256=hashlib.sha256(raw).hexdigest(),
        source='experimental_quality_checked_template_transfer_not_new_user_clicks',
        template_image_sha256=profile['image_sha256'], target_original_pixel_sha256=registration['target_pixel_sha256'],
        template_to_target_homography=registration['homography'],
        coordinate_frame='reference_template_pixel_plane',
        calibration_status='runtime_candidate_not_active_profile',
        note='Ticks are from reference only; current pointer must be detected in rectified target.')
    return raw, runtime
