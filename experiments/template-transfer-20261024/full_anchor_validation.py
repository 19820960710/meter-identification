"""Require direct image support for every automatically localized major anchor."""
import math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_locator_v6 import SETTINGS


def validate_full_anchors(source_ticks, located_ticks, current_axis, image_size):
    result = {'passed': False, 'reason': None, 'ticks': None,
              'scope': 'complete six-anchor image transfer candidate, not full-range metrology accuracy'}
    try:
        values = [t['value'] for t in source_ticks]
        if values != [0, 5, 10, 15, 20, 25] or [t['value'] for t in located_ticks] != values:
            raise ValueError('Expected all six original source values in order.')
        for t in located_ticks:
            if (not t['passed'] or not t['reverse']['passed'] or t['gray_ncc'] < SETTINGS['gray_ncc_min']
                or t['gradient_ncc'] < SETTINGS['gradient_ncc_min'] or t['peak_margin'] < SETTINGS['peak_margin_min']
                or t['roundtrip_px'] > SETTINGS['roundtrip_px_max'] or t['static_fraction'] < SETTINGS['static_fraction_min']
                or t['angular_adjustment_degrees'] > SETTINGS['angular_adjustment_degrees_max']):
                raise ValueError('One or more automatic anchors lack the predeclared image evidence.')
        points = np.array([t['target_point'] for t in located_ticks], float)
        c = np.array(current_axis, float)
        w, h = image_size
        if points.shape != (6, 2) or c.shape != (2,) or not np.isfinite(points).all() or not np.isfinite(c).all():
            raise ValueError('Nonfinite anchor or image-axis coordinates.')
        if not ((points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)).all():
            raise ValueError('Localized anchor outside rectified current image.')
        if len({tuple(p) for p in points}) != 6 or np.linalg.norm(points-c, axis=1).min() < 1:
            raise ValueError('Duplicate or axis-coincident anchors.')
        phi = np.degrees(np.arctan2(c[1]-points[:, 1], points[:, 0]-c[0])) % 360
        positions = (phi[0]-phi) % 360
        if (np.diff(positions) <= .1).any() or positions[-1] >= 359.5:
            raise ValueError('Localized anchors do not preserve the clockwise usable arc.')
        matrix = np.column_stack([2*points[:, 0], 2*points[:, 1], np.ones(len(points))])
        if np.linalg.matrix_rank(matrix) != 3:
            raise ValueError('Degenerate anchor geometry.')
        x, y, constant = np.linalg.lstsq(matrix, (points**2).sum(axis=1), rcond=None)[0]
        r2 = constant+x*x+y*y
        if r2 <= 25 or not math.isfinite(r2):
            raise ValueError('Invalid localized ring.')
        radius = math.sqrt(r2)
        residual = np.abs(np.linalg.norm(points-[x, y], axis=1)-radius)
        if residual.max() > max(3, radius*.08) or np.linalg.norm(c-[x, y]) > radius*.15:
            raise ValueError('Localized ring and image axis are inconsistent.')
        result.update(passed=True, ticks=[{'value': v, 'point': p.tolist()} for v, p in zip(values, points)],
            ring_center=[float(x), float(y)], ring_radius=radius,
            maximum_radial_residual_px=float(residual.max()),
            image_axis_to_ring_center_px=float(np.linalg.norm(c-[x, y])),
            ordered_arc_degrees=positions.tolist())
    except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as error:
        result['reason'] = str(error)
    return result
