"""Single-image gauge prototype; image-bound calibration, no operating verdict."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import cv2
import numpy as np


class ReadingError(ValueError):
    pass


def _angle(center, point):
    if not all(math.isfinite(float(v)) for v in [*center, *point]):
        raise ReadingError('Geometry coordinates must be finite.')
    dx, dy = float(point[0]) - center[0], center[1] - float(point[1])
    if math.hypot(dx, dy) < 1:
        raise ReadingError('A scale point or pointer coincides with the center.')
    return math.degrees(math.atan2(dy, dx)) % 360


def full_circle_calibration_check(center, ticks, direction):
    if direction not in ('clockwise', 'counterclockwise') or len(ticks) < 7:
        raise ReadingError('Uniform full-circle calibration needs six distinct anchors and closure.')
    values = np.asarray([float(t['value']) for t in ticks])
    if not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ReadingError('Full-circle scale values must increase and be finite.')
    angles = np.asarray([_angle(center, t['point']) for t in ticks])
    sign = -1 if direction == 'clockwise' else 1
    positions = sign * (angles - angles[0]) % 360
    closure = abs((angles[-1] - angles[0] + 180) % 360 - 180)
    if closure > 3 or np.any(np.diff(positions[:-1]) <= .1):
        raise ReadingError('Full-circle anchors must increase around a closing zero ray.')
    expected = (values[:-1] - values[0]) / (values[-1] - values[0]) * 360
    residuals = np.abs(positions[:-1] - expected)
    accepted = residuals <= 3
    if np.count_nonzero(accepted) < 6 or np.mean(accepted) < .80:
        raise ReadingError('Scale values and anchor angles are inconsistent with a uniform full circle.')
    return {'method': 'uniform_circle_anchor_consensus', 'click_tolerance_degrees': 3,
            'accepted_anchor_count': int(np.count_nonzero(accepted)),
            'outlier_anchor_values': values[:-1][~accepted].tolist(),
            'maximum_angular_deviation_degrees': float(residuals.max()),
            'raw_user_points_preserved': True,
            'note': 'Nonzero angular outliers are flagged; uniform mapping uses zero ray and period, not piecewise outlier values.'}


def angle_reading(center, tip, ticks, direction, full_circle=False):
    if direction not in ('clockwise', 'counterclockwise') or len(ticks) < 2:
        raise ReadingError('Invalid scale direction or fewer than two scale anchors.')
    values = np.asarray([float(t['value']) for t in ticks])
    if not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ReadingError('Scale values must be finite and strictly increasing.')
    angles = [_angle(center, t['point']) for t in ticks]
    sign = -1 if direction == 'clockwise' else 1
    positions = np.asarray([sign * (a - angles[0]) % 360 for a in angles])
    if full_circle:
        full_circle_calibration_check(center, ticks, direction)
        theta = sign * (_angle(center, tip) - angles[0]) % 360
        return float(values[0] + (values[-1] - values[0]) * theta / 360)
    if np.any(np.diff(positions) <= 0.1) or positions[-1] >= 359.5:
        raise ReadingError('Scale anchors do not form a monotonic usable arc.')
    theta = sign * (_angle(center, tip) - angles[0]) % 360
    if theta > positions[-1]:
        raise ReadingError('Pointer lies outside the calibrated scale arc.')
    return float(np.interp(theta, positions, values))


def validate_calibration_image(image_bytes, calibration):
    if hashlib.sha256(image_bytes).hexdigest() != calibration.get('image_sha256'):
        raise ReadingError('Calibration belongs to another image; calibrate this image first.')


def _roi(gray, box, padding=0):
    height, width = gray.shape[:2]
    x1 = max(0, int(math.floor(box[0])) - padding)
    y1 = max(0, int(math.floor(box[1])) - padding)
    x2 = min(width, int(math.ceil(box[2])) + padding)
    y2 = min(height, int(math.ceil(box[3])) + padding)
    if x2 - x1 < 8 or y2 - y1 < 8:
        raise ReadingError('Detection region is too small for geometry.')
    return gray[y1:y2, x1:x2], x1, y1


def locate_center(gray, center_box, expected_center=None, expected_tolerance=None, allow_small_cap_recovery=False):
    roi, x, y = _roi(gray, center_box, padding=3)
    size = min(roi.shape)
    if allow_small_cap_recovery and (expected_center is None or expected_tolerance is None):
        raise ReadingError('Small-cap recovery requires a calibrated ring constraint.')
    desired = np.asarray([(center_box[0] + center_box[2]) / 2,
                          (center_box[1] + center_box[3]) / 2])
    levels = [15, 10] if allow_small_cap_recovery and size <= 40 else [15]
    for votes in levels:
        circles = cv2.HoughCircles(
            cv2.GaussianBlur(roi, (5, 5), 0), cv2.HOUGH_GRADIENT,
            dp=1, minDist=max(6, size / 5), param1=60, param2=votes,
            minRadius=max(4, int(size * .12)), maxRadius=max(5, int(size * .47)),
        )
        if circles is None:
            continue
        candidates = []
        for cx, cy, radius in np.asarray(circles).reshape(-1, 3):
            center = np.asarray([float(cx) + x, float(cy) + y])
            distance = float(np.linalg.norm(center - desired))
            if distance > size * .30:
                continue
            if expected_center is not None and np.linalg.norm(center - expected_center) > expected_tolerance:
                continue
            candidates.append((distance, center.tolist(), float(radius)))
        if not candidates:
            continue
        distance, center, radius = min(candidates, key=lambda item: item[0])
        return center, {'method': 'axle_cap_hough_circle' if votes == 15 else 'axle_cap_hough_circle_small_reflective',
                        'radius_px': radius, 'hough_vote_threshold': votes,
                        'distance_from_roi_center_px': distance,
                        'calibration_ring_constraint_used': expected_center is not None}
    raise ReadingError('No reliable axle-cap circle inside center_region after geometry checks.')


def _colored_pointer(color_image, needle_box, center, panel_radius):
    roi, ox, oy = _roi(color_image, needle_box, padding=max(8, int(panel_radius * .055)))
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    c = np.asarray(center, dtype=float)
    candidates = []
    for hue in range(0, 180, 15):
        difference = np.abs(hsv[:, :, 0].astype(float) - hue)
        difference = np.minimum(difference, 180 - difference)
        mask = ((difference <= 10) & (hsv[:, :, 1] >= 110) & (hsv[:, :, 2] >= 60)).astype(np.uint8)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < max(30, (panel_radius * .01) ** 2):
                continue
            yy, xx = np.where(labels == label)
            vectors = np.column_stack([xx + ox, yy + oy]).astype(float) - c
            radii = np.linalg.norm(vectors, axis=1)
            extent = float(radii.max())
            # A colored axle cap alone is not a pointer. A component must extend
            # away from the cap while remaining connected close to the axle.
            if radii.min() > panel_radius * .16 or extent < panel_radius * .40:
                continue
            tip_vector = vectors[radii >= np.quantile(radii, .98)].mean(axis=0)
            if np.linalg.norm(tip_vector) < 1:
                raise ReadingError('Colored pointer directions are ambiguous.')
            direction = tip_vector / np.linalg.norm(tip_vector)
            outer = vectors[radii >= extent * .65]
            separation = np.degrees(np.arctan2(
                np.abs(direction[0] * outer[:, 1] - direction[1] * outer[:, 0]),
                outer @ direction,
            ))
            if np.count_nonzero(separation > 12) >= max(5, len(outer) * .025):
                raise ReadingError('Multiple long colored pointer directions need human review.')
            candidates.append({'direction': direction, 'tip': c + tip_vector,
                               'extent': extent, 'hue': hue, 'area': int(len(xx))})
    if not candidates:
        return None
    seed = max(candidates, key=lambda item: item['extent'])
    if any(float(np.dot(item['direction'], seed['direction'])) < math.cos(math.radians(8))
           and item['extent'] >= seed['extent'] * .65 for item in candidates):
        raise ReadingError('Multiple colored pointer components need human review.')
    # Saturation alone cannot assign measuring vs reference needle roles.
    # Check continuous long dark components as well, without treating text
    # stitched by Hough gaps as a second black shaft.
    dark_limit = min(150., float(np.percentile(hsv[:, :, 2], 80)) * .70)
    dark_mask = ((hsv[:, :, 1] <= 65) & (hsv[:, :, 2] <= dark_limit)).astype(np.uint8)
    dark_count, dark_labels, dark_stats, dark_centroids = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    for label in range(1, dark_count):
        if dark_stats[label, cv2.CC_STAT_AREA] < max(30, (panel_radius * .01) ** 2):
            continue
        yy, xx = np.where(dark_labels == label)
        vectors = np.column_stack([xx + ox, yy + oy]).astype(float) - c
        radii = np.linalg.norm(vectors, axis=1)
        if radii.min() > panel_radius * .16 or radii.max() < seed['extent'] * .65:
            continue
        outer = vectors[radii >= seed['extent'] * .65]
        separation = np.degrees(np.arctan2(
            np.abs(seed['direction'][0] * outer[:, 1] - seed['direction'][1] * outer[:, 0]),
            outer @ seed['direction'],
        ))
        if np.count_nonzero(separation > 12) >= max(5, len(outer) * .025):
            raise ReadingError('Colored and dark long needles require explicit role review.')
    tip = seed['tip'].tolist()
    return tip, {'method': 'colored_component_radial_direction',
                 'hue_bin': seed['hue'], 'component_area_px': seed['area'],
                 'extent_px': seed['extent'], 'direction_degrees': _angle(center, tip),
                 'achromatic_competition_checked': True,
                 'point_kind': 'outer_component_direction_endpoint_not_segmented_tip'}


def locate_pointer(gray, needle_box, center, panel_radius, color_image=None):
    if color_image is not None:
        colored = _colored_pointer(color_image, needle_box, center, panel_radius)
        if colored is not None:
            return colored
    roi, ox, oy = _roi(gray, needle_box, padding=max(8, int(panel_radius * .055)))
    edges = cv2.Canny(cv2.GaussianBlur(roi, (3, 3), 0), 40, 110)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 720, threshold=30,
        minLineLength=max(30, int(panel_radius * .22)),
        maxLineGap=max(8, int(panel_radius * .055)),
    )
    if lines is None:
        raise ReadingError('No sufficiently long pointer edges found.')
    dark, bright = np.percentile(roi, [2, 80])
    if bright - dark < 12:
        raise ReadingError('Insufficient contrast for a reliable pointer.')
    threshold = float((dark + bright) / 2)
    candidates = []
    c = np.asarray(center, dtype=float)
    for raw in np.asarray(lines).reshape(-1, 4):
        a = raw[:2].astype(float) + [ox, oy]
        b = raw[2:].astype(float) + [ox, oy]
        vector = b - a
        length = float(np.linalg.norm(vector))
        if length <= 0:
            continue
        distance = abs(float(vector[0] * (a[1] - c[1]) - vector[1] * (a[0] - c[0]))) / length
        far = a if np.linalg.norm(a - c) >= np.linalg.norm(b - c) else b
        extent = float(np.linalg.norm(far - c))
        if distance > max(8, panel_radius * .07) or extent < panel_radius * .30:
            continue
        direction = vector / length
        if float(np.dot(direction, far - c)) < 0:
            direction = -direction
        # A detached mark whose infinite extension crosses the axle is insufficient.
        # Require visible shaft evidence between the axle cap and the outward ray.
        radii = np.linspace(max(12, panel_radius * .085), min(extent * .85, panel_radius * .55), 80)
        perpendicular = np.asarray([-direction[1], direction[0]])
        samples = c + radii[:, None, None] * direction + np.arange(-3, 4)[None, :, None] * perpendicular
        ix = np.clip(np.rint(samples[:, :, 0]).astype(int), 0, gray.shape[1] - 1)
        iy = np.clip(np.rint(samples[:, :, 1]).astype(int), 0, gray.shape[0] - 1)
        support = float(np.mean(np.min(gray[iy, ix], axis=1) < threshold))
        if support < .65:
            continue
        candidates.append({'vector': direction, 'radial_vector': (far - c) / extent, 'length': length,
                           'extent': extent, 'distance': distance, 'shaft_support': support})
    if not candidates:
        raise ReadingError('Pointer edges do not pass near the axle.')
    # Total Hough length can span both sides of the axle and stitch printed text.
    # Compare the visible outward reach, which is the relevant pointer evidence.
    seed = max(candidates, key=lambda item: item['extent'])
    # Tapered needle edges have different slopes. Group by the axle-to-outward
    # endpoint ray, retaining edge directions for the final centerline estimate.
    cluster = [item for item in candidates
               if float(np.dot(item['radial_vector'], seed['radial_vector'])) >= math.cos(math.radians(7))]
    competing = [item for item in candidates
                 if float(np.dot(item['radial_vector'], seed['radial_vector'])) < math.cos(math.radians(7))
                 and item['extent'] >= seed['extent'] * .65]
    if competing:
        raise ReadingError('Multiple long pointer directions need human review.')
    weights = np.asarray([item['length'] for item in cluster])
    direction = np.average([item['vector'] for item in cluster], axis=0, weights=weights)
    direction /= np.linalg.norm(direction)
    extent = max(item['extent'] for item in cluster)
    tip = c + direction * extent
    return tip.tolist(), {'method': 'near_axis_long_edge_cluster',
                         'supporting_lines': len(cluster),
                         'candidate_lines': len(candidates), 'extent_px': extent,
                         'direction_selection': 'outward_extent_from_axle',
                         'direction_grouping': 'axle_to_outward_endpoint_ray',
                         'ignored_short_directions': sum(
                             float(np.dot(item['radial_vector'], seed['radial_vector'])) < math.cos(math.radians(7))
                             and item['extent'] < seed['extent'] * .65 for item in candidates),
                         'minimum_shaft_support': min(item['shaft_support'] for item in cluster),
                         'direction_degrees': _angle(center, tip),
                         'point_kind': 'axis_direction_endpoint_not_segmented_tip'}


def _inside(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def select_gauge_detections(detections):
    def middle(box):
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

    panels = [d for d in detections if d['label'] == 'panel']
    if len(panels) != 1:
        raise ReadingError('Prototype requires exactly one detected panel.')
    panel = panels[0]['box']
    related = [d for d in detections if _inside(middle(d['box']), panel)]
    needles = [d for d in related if d['label'] == 'needle']
    if not needles:
        raise ReadingError('Missing or ambiguous needle detections.')
    def area(box):
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    if any(area(d['box']) <= 0 for d in needles):
        raise ReadingError('Invalid needle detection region.')
    needle = max(needles, key=lambda d: area(d['box']))['box']
    # A counterweight subproposal can survive IoU NMS inside the complete
    # needle ROI. Use containment, never confidence, to resolve this case.
    # Actual multi-needle directions are still checked in the image afterward.
    for detection in needles:
        box = detection['box']
        intersection = (max(0, min(needle[2], box[2]) - max(needle[0], box[0])) *
                        max(0, min(needle[3], box[3]) - max(needle[1], box[1])))
        if intersection / area(box) < .95:
            raise ReadingError('Missing or ambiguous needle detections.')
    all_centers = [d for d in related if d['label'] == 'center_region']
    centers = [d for d in all_centers if _inside(middle(d['box']), needle)]
    if len(centers) != 1:
        raise ReadingError('Missing or ambiguous axle center inside the needle region.')
    return panel, needle, centers[0]['box'], {
        'method': 'axle_midpoint_inside_panel_and_needle',
        'needle_proposal_count': len(needles),
        'needle_roi_selection': 'complete_roi_with_contained_subproposals' if len(needles) > 1 else 'single_detection',
        'ignored_needle_boxes': [d['box'] for d in needles if d['box'] != needle],
        'ignored_center_boxes': [d['box'] for d in all_centers
                                 if not _inside(middle(d['box']), needle)],
    }


def fit_calibration_ring(ticks):
    if len(ticks) < 7:
        raise ReadingError('Full-circle recovery needs six distinct anchors plus closure.')
    points = np.asarray([t['point'] for t in ticks[:-1]], dtype=float)
    if points.shape[1:] != (2,) or not np.isfinite(points).all():
        raise ReadingError('Invalid full-circle calibration points.')
    matrix = np.column_stack([2 * points[:, 0], 2 * points[:, 1], np.ones(len(points))])
    if np.linalg.matrix_rank(matrix) < 3:
        raise ReadingError('Degenerate calibration ring.')
    cx, cy, constant = np.linalg.lstsq(matrix, (points ** 2).sum(axis=1), rcond=None)[0]
    radius_squared = constant + cx * cx + cy * cy
    if not np.isfinite(radius_squared) or radius_squared <= 25:
        raise ReadingError('Calibration ring radius is not usable.')
    radius = float(np.sqrt(radius_squared))
    residual = np.abs(np.linalg.norm(points - [cx, cy], axis=1) - radius)
    if float(residual.max()) > max(3, radius * .08):
        raise ReadingError('Calibration anchors do not form a near-circular dial.')
    return [float(cx), float(cy)], radius, {
        'method': 'manual_calibration_ring_fit', 'radius_px': radius,
        'maximum_radial_residual_px': float(residual.max()),
    }


def validate_ring_for_panel(center, radius, panel):
    width, height = panel[2] - panel[0], panel[3] - panel[1]
    if not _inside(center, panel) or not (min(width, height) * .30 <= radius <= max(width, height) * .60):
        raise ReadingError('Calibration ring size/position is inconsistent with the outer panel.')
    margin = radius * .20
    if (center[0] - radius < panel[0] - margin or center[0] + radius > panel[2] + margin
            or center[1] - radius < panel[1] - margin or center[1] + radius > panel[3] + margin):
        raise ReadingError('Calibration ring extends outside the detected panel.')


def select_full_circle_regions(detections, ring_center=None, ring_radius=None):
    all_panels = [d for d in detections if d['label'] == 'panel']
    panels = all_panels
    if (ring_center is None) != (ring_radius is None):
        raise ReadingError('Both calibration ring center and radius are required for panel binding.')
    if ring_center is not None:
        panels = []
        for detection in all_panels:
            try:
                validate_ring_for_panel(ring_center, ring_radius, detection['box'])
                panels.append(detection)
            except ReadingError:
                continue
    if len(panels) != 1:
        raise ReadingError('Missing or ambiguous panel associated with the calibration ring.')
    panel = panels[0]['box']
    needles = [d['box'] for d in detections if d['label'] == 'needle' and _inside(
        [(d['box'][0] + d['box'][2]) / 2, (d['box'][1] + d['box'][3]) / 2], panel)]
    if not needles:
        raise ReadingError('No outer needle region detected.')

    def overlaps(a, b):
        return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])

    connected = {0}
    changed = True
    while changed:
        changed = False
        for index in range(len(needles)):
            if index not in connected and any(overlaps(needles[index], needles[j]) for j in connected):
                connected.add(index)
                changed = True
    if len(connected) != len(needles):
        raise ReadingError('Separated needle regions cannot be merged safely.')
    needle = [min(b[0] for b in needles), min(b[1] for b in needles),
              max(b[2] for b in needles), max(b[3] for b in needles)]
    all_centers = [d['box'] for d in detections if d['label'] == 'center_region']
    centers = [b for b in all_centers if _inside([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], needle)]
    if len(centers) > 1:
        raise ReadingError('Multiple axle regions remain inside the outer needle ROI.')
    return panel, needle, centers[0] if centers else None, {
        'method': 'explicit_outer_hand_overlapping_roi_union',
        'panel_selection': 'calibration_ring_geometry' if ring_center is not None else 'single_detection',
        'ignored_panel_boxes': [d['box'] for d in all_panels if d['box'] != panel],
        'needle_proposal_count': len(needles),
        'ignored_center_boxes': [b for b in all_centers if b not in centers],
    }


def locate_outer_pointer(gray, needle_box, center, ring_radius, color_image=None):
    if color_image is not None:
        colored = _colored_pointer(color_image, needle_box, center, ring_radius)
        if colored is not None:
            if colored[1]['extent_px'] < ring_radius * .70:
                raise ReadingError('Colored hand does not reach the outer scale.')
            return colored
    roi, ox, oy = _roi(gray, needle_box, padding=max(8, int(ring_radius * .07)))
    dark, bright = np.percentile(roi, [2, 80])
    if bright - dark < 12:
        raise ReadingError('Insufficient outer hand contrast.')
    mask = (roi < (dark + bright) / 2).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    c = np.asarray(center, dtype=float)
    candidates = []
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < 20:
            continue
        yy, xx = np.where(labels == label)
        vectors = np.column_stack([xx + ox, yy + oy]).astype(float) - c
        radii = np.linalg.norm(vectors, axis=1)
        extent = float(radii.max())
        if radii.min() > ring_radius * .16 or extent < ring_radius * .70:
            continue
        tip_vector = vectors[radii >= np.quantile(radii, .98)].mean(axis=0)
        norm = float(np.linalg.norm(tip_vector))
        if norm < 1:
            raise ReadingError('Outer long hand directions are ambiguous.')
        direction = tip_vector / norm
        outer = vectors[radii >= extent * .65]
        separation = np.degrees(np.arctan2(
            np.abs(direction[0] * outer[:, 1] - direction[1] * outer[:, 0]), outer @ direction,
        ))
        if np.count_nonzero(separation > 12) >= max(5, len(outer) * .025):
            raise ReadingError('Multiple long outer hands need role review.')
        candidates.append({'direction': direction, 'tip': c + tip_vector,
                           'extent': extent, 'area': int(len(xx))})
    if not candidates:
        raise ReadingError('No connected long hand reaches the outer scale.')
    seed = max(candidates, key=lambda item: item['extent'])
    if any(float(np.dot(item['direction'], seed['direction'])) < math.cos(math.radians(8))
           and item['extent'] >= seed['extent'] * .65 for item in candidates):
        raise ReadingError('More than one long outer hand is visible.')
    tip = seed['tip'].tolist()
    return tip, {'method': 'outer_long_hand_foreground_component',
                 'extent_px': seed['extent'], 'component_area_px': seed['area'],
                 'direction_degrees': _angle(center, tip), 'counter_hand_read': False,
                 'point_kind': 'outer_component_direction_endpoint_not_segmented_tip'}


def read_image(image_path, calibration, model_path):
    image_path, model_path = Path(image_path), Path(model_path)
    valid_profile_type = isinstance(calibration, dict)
    # Match the exported X-AnyLabeling detector configuration; do not rely on changing library defaults.
    detection_settings = {'imgsz': 640, 'device': 'cpu', 'conf': .25, 'iou': .45,
                          'agnostic_nms': False, 'max_det': 300}
    result = {'status': 'needs_review', 'value': None, 'unit': calibration.get('unit') if valid_profile_type else None,
              'image': str(image_path.resolve()), 'model': str(model_path.resolve()),
              'calibration': None, 'detections': [], 'geometry': {},
              'detection_settings': detection_settings.copy(),
              'operating_status': 'not_evaluated',
              'method': 'image_bound_piecewise_angle_calibration'}
    try:
        if not valid_profile_type:
            raise ReadingError('Calibration must be an object.')
        image_bytes = image_path.read_bytes()
        validate_calibration_image(image_bytes, calibration)
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ReadingError('Image cannot be decoded.')
        if list(image.shape[1::-1]) != calibration.get('image_size'):
            raise ReadingError('Image size does not match calibration.')
        for tick in calibration['ticks']:
            coordinates = tick['point']
            if len(coordinates) != 2 or not all(math.isfinite(float(v)) for v in [tick['value'], *coordinates]):
                raise ReadingError('Calibration values and coordinates must be finite.')
            if not (0 <= coordinates[0] < image.shape[1] and 0 <= coordinates[1] < image.shape[0]):
                raise ReadingError('Calibration anchor lies outside the image.')
        division = calibration.get('minimum_division')
        if division is not None and (not math.isfinite(float(division)) or float(division) <= 0):
            raise ReadingError('Minimum division must be a positive finite number.')
        result['calibration'] = calibration
        mode = calibration.get('scale_mode', 'partial')
        if mode not in ('partial', 'full_circle_uniform'):
            raise ReadingError('Unknown calibration scale mode.')
        full_circle = mode == 'full_circle_uniform'
        if full_circle and (calibration.get('unit') != 'div' or calibration.get('needle_role') != 'outer_long_hand'):
            raise ReadingError('Full-circle mode only returns explicit outer long-hand divisions (div).')
        if not model_path.is_file():
            raise ReadingError('Model file is missing.')
        from ultralytics import YOLO
        prediction = YOLO(str(model_path)).predict(image, **detection_settings, verbose=False)[0]
        for box in prediction.boxes:
            result['detections'].append({'label': prediction.names[int(box.cls.item())],
                                         'score': float(box.conf.item()),
                                         'box': box.xyxy[0].tolist()})
        if full_circle:
            ring_center, ring_radius, ring_info = fit_calibration_ring(calibration['ticks'])
            panel, needle_box, center_box, association = select_full_circle_regions(
                result['detections'], ring_center=ring_center, ring_radius=ring_radius)
            validate_ring_for_panel(ring_center, ring_radius, panel)
            if center_box is None:
                padding = ring_radius * .15
                center_box = [ring_center[0] - padding, ring_center[1] - padding,
                              ring_center[0] + padding, ring_center[1] + padding]
                association['axle_search_source'] = 'manual_calibration_ring_seed'
        else:
            panel, needle_box, center_box, association = select_gauge_detections(result['detections'])
        result['association_diagnostic'] = association
        for detection in result['detections']:
            if detection['label'] == 'panel' and detection['box'] != panel:
                detection['ignored_for_geometry'] = True
            if detection['label'] == 'needle' and detection['box'] in association.get('ignored_needle_boxes', []):
                detection['ignored_for_geometry'] = True
            if detection['label'] == 'center_region' and detection['box'] != center_box:
                detection['ignored_for_geometry'] = True
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if full_circle:
            center, center_info = locate_center(gray, center_box, expected_center=ring_center,
                expected_tolerance=ring_radius * .08, allow_small_cap_recovery=True)
        else:
            center, center_info = locate_center(gray, center_box)
        if full_circle:
            if np.linalg.norm(np.asarray(center) - ring_center) > ring_radius * .15:
                raise ReadingError('Axle circle disagrees with the calibration ring.')
            center_info['calibration_ring'] = ring_info
        if not _inside(center, needle_box):
            raise ReadingError('Refined axle center lies outside the needle region.')
        radius = min(panel[2] - panel[0], panel[3] - panel[1]) / 2
        if full_circle:
            tip, pointer_info = locate_outer_pointer(gray, needle_box, center, ring_radius, color_image=image)
        else:
            tip, pointer_info = locate_pointer(gray, needle_box, center, radius, color_image=image)
        result['geometry'] = {'center': center, 'pointer_endpoint': tip,
                              'center_diagnostic': center_info, 'pointer_diagnostic': pointer_info}
        if full_circle:
            result['calibration_consistency'] = full_circle_calibration_check(center, calibration['ticks'], calibration['direction'])
        value = angle_reading(center, tip, calibration['ticks'], calibration['direction'], full_circle=full_circle)
        if not math.isfinite(value):
            raise ReadingError('Computed reading is not finite.')
        result.update(status='readable', value=value,
                      display_value=f'{value:.2f} {calibration["unit"]}',
                      note='Prototype estimate; decimal digits are not an accuracy guarantee.')
        if full_circle:
            result.update(method='uniform_full_circle_outer_divisions', quantity='outer_scale_divisions',
                          counter_read=False, total_displacement_mm=None,
                          note='Outer scale divisions only; no revolution count or total millimeter displacement.')
    except (ReadingError, OSError, KeyError, TypeError, ValueError, RuntimeError, cv2.error) as error:
        result['reason'] = str(error)
    return result


def draw_result(image, result):
    canvas = cv2.copyMakeBorder(image, 0, 52, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    colors = {'panel': (255, 100, 30), 'needle': (0, 170, 0), 'center_region': (0, 100, 255)}
    for detection in result['detections']:
        x1, y1, x2, y2 = map(int, detection['box'])
        color = colors.get(detection['label'], (180, 180, 180))
        label = detection['label']
        if detection.get('ignored_for_geometry'):
            color = (140, 140, 140)
            label = 'ignored ' + label
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f'{label} {detection["score"]:.2f}',
                    (max(0, x1), max(18, y1)), cv2.FONT_HERSHEY_SIMPLEX, .50, color, 1)
    geometry = result['geometry']
    if geometry:
        center = tuple(int(round(v)) for v in geometry['center'])
        tip = tuple(int(round(v)) for v in geometry['pointer_endpoint'])
        for tick in result['calibration']['ticks']:
            point = tuple(int(round(v)) for v in tick['point'])
            outlier = tick['value'] in result.get('calibration_consistency', {}).get('outlier_anchor_values', [])
            color = (140, 140, 140) if outlier else (255, 0, 255)
            label = str(tick['value']) + ('*' if outlier else '')
            cv2.circle(canvas, point, 5, color, -1)
            cv2.putText(canvas, label, (point[0] + 6, point[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
        cv2.arrowedLine(canvas, center, tip, (0, 0, 255), 2, tipLength=.05)
        cv2.circle(canvas, center, 5, (0, 255, 255), -1)
    label = result.get('display_value', 'NEEDS REVIEW')
    # OpenCV's built-in font cannot render Chinese or the degree symbol.
    label = label.replace('摄氏度', 'degC').replace('℃', 'degC').replace('°C', 'degC')
    cv2.rectangle(canvas, (5, canvas.shape[0] - 52), (canvas.shape[1] - 5, canvas.shape[0] - 5), (255, 255, 255), -1)
    cv2.putText(canvas, label, (18, canvas.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, .85, (0, 0, 0), 2)
    return canvas
