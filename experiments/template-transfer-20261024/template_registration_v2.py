"""Static face matching with narrow motion exclusion and calibrated ring bounds."""
import copy
import hashlib
import math
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import template_registration as v1

SETTINGS = copy.deepcopy(v1.SETTINGS)
SETTINGS.update(ratio=.65, feature_region='static_face', source_annulus_radius=[0., 1.20],
                descriptor_support_keypoint_size=1.5, pose_angle_degrees=20., pose_scale_ratio=1.35,
                boundary='64_point_reference_calibrated_ring', motion_mask='image_shaft_and_counterweight_v2')
project = v1.project
rectify_for_reading = v1.rectify_for_reading


def motion_from_detections(image, detections):
    panel, needle, axle, _ = v1.meter.select_gauge_detections(detections)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    center, detail = v1.meter.locate_center(gray, axle)
    radius = min(panel[2] - panel[0], panel[3] - panel[1]) / 2
    endpoint, diagnostic = v1.meter.locate_pointer(gray, needle, center, radius, color_image=image)
    return {'center': center, 'endpoint': endpoint, 'cap_radius': detail['radius_px'],
            'panel_radius': radius, 'direction_diagnostic': diagnostic, 'panel': panel}


def motion_mask(size, motion):
    width, height = size
    c = np.array(motion['center'], float)
    t = np.array(motion['endpoint'], float)
    cap, radius = float(motion['cap_radius']), float(motion['panel_radius'])
    length = float(np.linalg.norm(t - c))
    if (not np.isfinite([*c, *t, cap, radius]).all() or min(cap, radius) <= 0
            or length < max(10, radius * .2)):
        raise ValueError('Insufficient image evidence for a moving-pointer exclusion mask.')
    u = (t - c) / length
    p = np.array([-u[1], u[0]])
    mask = np.zeros((height, width), np.uint8)
    front = np.array([c-u*6+p*cap*1.1, c-u*6-p*cap*1.1,
                      t+u*max(4, radius*.03)-p*3, t+u*max(4, radius*.03)+p*3])
    back = c - u * length * .65
    tail = np.array([c+p*cap*1.2, c-p*cap*1.2,
                     back-p*radius*.17, back+p*radius*.17])
    cv2.fillConvexPoly(mask, np.rint(front).astype(np.int32), 255)
    cv2.fillConvexPoly(mask, np.rint(tail).astype(np.int32), 255)
    cv2.circle(mask, tuple(np.rint(c).astype(int)), math.ceil(cap*1.4+3), 255, -1)
    return cv2.dilate(mask, np.ones((5, 5), np.uint8))


def pose_consensus(angles, scales):
    if not angles:
        return []
    a, s = np.array(angles, float), np.array(scales, float)
    if len(a) != len(s) or not np.isfinite(a).all() or not np.isfinite(s).all() or (s <= 0).any():
        raise ValueError('Invalid descriptor orientation/scale metadata.')
    angle_delta = np.abs((a[:, None]-a[None, :]+180) % 360-180)
    scale_delta = np.abs(np.log(s[:, None]/s[None, :]))
    compatible = ((angle_delta <= SETTINGS['pose_angle_degrees']) &
                  (scale_delta <= math.log(SETTINGS['pose_scale_ratio'])))
    seed = int(np.argmax(compatible.sum(axis=1)))
    return np.flatnonzero(compatible[seed]).tolist()


def ring_geometry(h, center, radius, target_size, proposals):
    angles = np.linspace(0, 2*math.pi, 64, endpoint=False)
    circle = np.column_stack([center[0]+radius*np.cos(angles), center[1]+radius*np.sin(angles)])
    projected = project(circle, h)
    denominator = np.column_stack([circle, np.ones(len(circle))]) @ np.array(h)[2]
    area = float(np.sum(projected[:, 0]*np.roll(projected[:, 1], -1)-
                        projected[:, 1]*np.roll(projected[:, 0], -1))/2)
    w, height = target_size
    eps = np.finfo(float).eps * max(w, height) * 1000
    visible = (np.isfinite(projected).all() and (denominator > 1e-6).all() and area > 1
               and cv2.isContourConvex(projected.astype(np.float32))
               and (projected[:, 0] >= -eps).all() and (projected[:, 0] <= w+eps).all()
               and (projected[:, 1] >= -eps).all() and (projected[:, 1] <= height+eps).all())
    box = [float(projected[:, 0].min()), float(projected[:, 1].min()),
           float(projected[:, 0].max()), float(projected[:, 1].max())]
    overlaps = [v1._legacy.iou(box, p) for p in proposals]
    matched = [i for i, value in enumerate(overlaps) if value >= SETTINGS['panel_iou']]
    return {'passed': bool(visible and len(matched) == 1), 'visible_oriented_convex': bool(visible),
            'projected_ring': projected.tolist(), 'projected_ring_box': box,
            'panel_ious': overlaps, 'matched_panel_indices': matched}


def feature_mask(image, panel, motion, circle=None):
    height, width = image.shape[:2]
    yy, xx = np.indices((height, width))
    mask = ((xx >= panel[0]) & (xx <= panel[2]) &
            (yy >= panel[1]) & (yy <= panel[3])).astype(np.uint8)*255
    if circle is not None:
        center, radius = circle
        mask[np.hypot(xx-center[0], yy-center[1]) > radius*1.2] = 0
    else:
        ellipse = np.zeros_like(mask)
        cv2.ellipse(ellipse, (round((panel[0]+panel[2])/2), round((panel[1]+panel[3])/2)),
                    (max(1, round((panel[2]-panel[0])/2)), max(1, round((panel[3]-panel[1])/2))),
                    0, 0, 360, 255, -1)
        mask &= ellipse
    mask[motion_mask([width, height], motion) != 0] = 0
    return mask


def static_features(sift, image, mask):
    keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask)
    if descriptors is None:
        return [], None
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    keep = [i for i, k in enumerate(keypoints) if
            distance[min(mask.shape[0]-1, round(k.pt[1])), min(mask.shape[1]-1, round(k.pt[0]))]
            >= k.size*SETTINGS['descriptor_support_keypoint_size']]
    return [keypoints[i] for i in keep], descriptors[keep] if keep else None


def register_partial_v2(reference_raw, target, profile, panel, proposals,
                        source_detections, target_detections):
    out = {'status': 'needs_review', 'reason': None, 'numeric_value': None,
           'settings': copy.deepcopy(SETTINGS), 'homography': None, 'matches': [],
           'quality': None, 'active_profile_written': False}
    try:
        reference, center, radius = v1.reference_ring(reference_raw, profile)
        sm = motion_from_detections(reference, source_detections)
        tm = motion_from_detections(target, target_detections)
        source_mask = feature_mask(reference, panel, sm, (center, radius))
        target_mask = feature_mask(target, tm['panel'], tm)
        out.update(template_image_sha256=hashlib.sha256(reference_raw).hexdigest(),
                   target_pixel_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
                   source_motion=sm, target_motion=tm,
                   reference_mask_sha256=hashlib.sha256(source_mask.tobytes()).hexdigest(),
                   target_mask_sha256=hashlib.sha256(target_mask.tobytes()).hexdigest())
        cv2.setRNGSeed(SETTINGS['seed'])
        cv2.setNumThreads(1)
        sift = cv2.SIFT_create(nfeatures=SETTINGS['sift_features'])
        sk, sd = static_features(sift, reference, source_mask)
        tk, td = static_features(sift, target, target_mask)
        out.update(reference_features=len(sk), target_features=len(tk))
        if sd is None or td is None or min(len(sd), len(td)) < 2:
            raise ValueError('Insufficient static descriptors.')
        bf = cv2.BFMatcher(cv2.NORM_L2)

        def filter_pairs(a, b):
            return {p[0].queryIdx: p[0].trainIdx for p in bf.knnMatch(a, b, k=2)
                    if len(p) == 2 and p[0].distance < SETTINGS['ratio']*p[1].distance}

        forward, backward = filter_pairs(sd, td), filter_pairs(td, sd)
        pairs = [(i, j) for i, j in forward.items() if backward.get(j) == i]
        pairs.sort(key=lambda p: (sk[p[0]].pt[0], sk[p[0]].pt[1], p[0], p[1]))
        angles = [(tk[j].angle-sk[i].angle) % 360 for i, j in pairs]
        scales = [tk[j].size/sk[i].size for i, j in pairs]
        kept = pose_consensus(angles, scales)
        out['tentative_matches_before_pose_consensus'] = [
            {'reference_point': list(sk[i].pt), 'target_point': list(tk[j].pt),
             'orientation_delta': angles[k], 'scale_ratio': scales[k], 'retained': k in kept}
            for k, (i, j) in enumerate(pairs)]
        out['mutual_matches_before_pose'] = len(pairs)
        pairs = [pairs[k] for k in kept]
        out['mutual_matches'] = len(pairs)
        out['matches'] = [{'reference_point': list(sk[i].pt), 'target_point': list(tk[j].pt),
                           'split': 'holdout' if k % 3 == 0 else 'fit'}
                          for k, (i, j) in enumerate(pairs)]
        if len(pairs) < SETTINGS['minimum_matches']:
            raise ValueError('Insufficient pose-consistent static matches.')
        src, dst = np.array([sk[i].pt for i, j in pairs]), np.array([tk[j].pt for i, j in pairs])
        hold = np.arange(len(pairs)) % 3 == 0
        h, inliers = cv2.findHomography(src[~hold], dst[~hold], cv2.RANSAC,
            SETTINGS['ransac_px'], maxIters=SETTINGS['ransac_iterations'], confidence=SETTINGS['ransac_confidence'])
        if h is None or not np.isfinite(h).all() or abs(h[2, 2]) < 1e-9 or abs(np.linalg.det(h)) < 1e-10:
            raise ValueError('No nonsingular homography.')
        h /= h[2, 2]
        out.update(homography=h.tolist(), fit_inliers=inliers.reshape(-1).astype(bool).tolist())
        old = v1.quality_check(h, src[~hold], dst[~hold], inliers, src[hold], dst[hold],
            panel, list(target.shape[1::-1]), proposals, [t['point'] for t in profile['ticks']])
        ring = ring_geometry(h, center, radius, list(target.shape[1::-1]), proposals)
        failures = [f for f in old['failures'] if f not in ('projected_panel_geometry', 'unique_coarse_panel_binding')]
        if not ring['visible_oriented_convex']:
            failures.append('projected_calibrated_ring_geometry')
        if len(ring['matched_panel_indices']) != 1:
            failures.append('unique_coarse_ring_binding')
        out['quality'] = {'failures': failures, 'metrics': old['metrics'],
                          'legacy_rectangle_diagnostic': old, 'ring_geometry': ring}
        out['projected_ticks_for_diagnosis'] = [{'value': t['value'], 'point': p.tolist()}
            for t, p in zip(profile['ticks'], project([t['point'] for t in profile['ticks']], h))]
        if failures:
            out['reason'] = 'registration_quality_rejected'
        else:
            out.update(status='registration_candidate', reason=None)
    except (ValueError, KeyError, TypeError, RuntimeError, np.linalg.LinAlgError, cv2.error) as error:
        out['reason'] = str(error)
    return out
