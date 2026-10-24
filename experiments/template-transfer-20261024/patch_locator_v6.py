"""Fractional-anchor-preserving patch correspondence; frozen quality thresholds."""
import math
import cv2
import numpy as np

SETTINGS = {'patch_radius_fraction': .10, 'search_radius_fraction': .12,
    'gray_ncc_min': .60, 'gradient_ncc_min': .65, 'peak_margin_min': .05,
    'static_fraction_min': .95, 'roundtrip_px_max': 2., 'angular_adjustment_degrees_max': 3.,
    'source_std_min': 3., 'gaussian_sigma': 1.1}


def fields(image):
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 1.1).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return gray, cv2.magnitude(gx, gy)


def crop(field, point, radius):
    x, y = point
    if x-radius < 0 or y-radius < 0 or x+radius >= field.shape[1] or y+radius >= field.shape[0]:
        raise ValueError('Patch extends beyond the image.')
    return cv2.getRectSubPix(field, (2*radius+1, 2*radius+1), (float(x), float(y)))


def _once(source_fields, target_fields, point, patch_radius, search_radius,
          source_valid, target_valid):
    result = {'passed': False, 'reason': None, 'target_point': None}
    try:
        pgray, pgrad = [crop(f, point, patch_radius) for f in source_fields]
        if min(float(pgray.std()), float(pgrad.std())) < SETTINGS['source_std_min']:
            raise ValueError('Source patch lacks distinguishing image structure.')
        if float(crop(source_valid.astype(np.float32)/255, point, patch_radius).mean()) < SETTINGS['static_fraction_min']:
            raise ValueError('Source patch overlaps a moving or invalid region.')
        cx, cy = [float(v) for v in point]
        total = patch_radius+search_radius
        if cx-total < 0 or cy-total < 0 or cx+total >= target_fields[0].shape[1] or cy+total >= target_fields[0].shape[0]:
            raise ValueError('Search window extends beyond the image.')
        search_gray = crop(target_fields[0], point, total)
        search_grad = crop(target_fields[1], point, total)
        search_valid = crop(target_valid.astype(np.float32)/255, point, total)
        score_gray = cv2.matchTemplate(search_gray, pgray, cv2.TM_CCOEFF_NORMED)
        score_grad = cv2.matchTemplate(search_grad, pgrad, cv2.TM_CCOEFF_NORMED)
        coverage = cv2.matchTemplate(search_valid,
                                    np.ones(pgray.shape, np.float32), cv2.TM_CCORR)/pgray.size
        combined = (score_gray+score_grad)/2
        combined[(coverage < SETTINGS['static_fraction_min']) | ~np.isfinite(combined)] = -2
        _, best, _, location = cv2.minMaxLoc(combined)
        x, y = location
        yy, xx = np.indices(combined.shape)
        remaining = combined.copy()
        remaining[(xx-x)**2+(yy-y)**2 <= max(3, round(patch_radius*.25))**2] = -2
        second = float(remaining.max())
        gray_score, gradient_score = float(score_gray[y, x]), float(score_grad[y, x])
        result.update(target_point=[float(cx+x-search_radius), float(cy+y-search_radius)],
            gray_ncc=gray_score, gradient_ncc=gradient_score, best_score=float(best),
            second_score=second, peak_margin=float(best-second), static_fraction=float(coverage[y, x]))
        if (best < -1 or gray_score < SETTINGS['gray_ncc_min'] or gradient_score < SETTINGS['gradient_ncc_min']):
            raise ValueError('Insufficient gray/gradient patch agreement.')
        if best-second < SETTINGS['peak_margin_min']:
            raise ValueError('Repeated image structure leaves ambiguous match peaks.')
        result['passed'] = True
    except (ValueError, cv2.error) as error:
        result['reason'] = str(error)
    return result


def locate_anchor(reference, rectified_target, point, radius, source_valid=None, target_valid=None):
    source_valid = np.full(reference.shape[:2], 255, np.uint8) if source_valid is None else source_valid
    target_valid = np.full(rectified_target.shape[:2], 255, np.uint8) if target_valid is None else target_valid
    sf, tf = fields(reference), fields(rectified_target)
    patch_radius, search_radius = max(12, round(radius*.10)), max(8, round(radius*.12))
    forward = _once(sf, tf, point, patch_radius, search_radius, source_valid, target_valid)
    result = {**forward, 'source_point': list(point), 'patch_radius': patch_radius,
              'search_radius': search_radius, 'settings': dict(SETTINGS)}
    if not forward['passed']:
        return result
    backward = _once(tf, sf, forward['target_point'], patch_radius, search_radius, target_valid, source_valid)
    result['reverse'] = backward
    result['roundtrip_px'] = None if backward['target_point'] is None else float(np.linalg.norm(
        np.array(backward['target_point'])-point))
    if not backward['passed'] or result['roundtrip_px'] > SETTINGS['roundtrip_px_max']:
        result.update(passed=False, reason='Patch correspondence failed reverse consistency.')
    return result


def fixed_similarity(reference, rectified_target, source_point, target_point, radius,
                     source_valid=None, target_valid=None):
    r = max(12, round(radius*.10))
    sf, tf = fields(reference), fields(rectified_target)
    try:
        if source_valid is not None and crop(source_valid.astype(np.float32)/255, source_point, r).mean() < .95:
            raise ValueError('Source patch is not static.')
        if target_valid is not None and crop(target_valid.astype(np.float32)/255, target_point, r).mean() < .95:
            raise ValueError('Target patch is not static.')
        a, b = [crop(f, source_point, r) for f in sf], [crop(f, target_point, r) for f in tf]
        if min(float(v.std()) for v in a+b) < 3:
            raise ValueError('Insufficient fixed-patch variation.')
        scores = [float(cv2.matchTemplate(bb, aa, cv2.TM_CCOEFF_NORMED)[0, 0]) for aa, bb in zip(a, b)]
        return {'valid': True, 'gray_ncc': scores[0], 'gradient_ncc': scores[1], 'combined': sum(scores)/2}
    except (ValueError, cv2.error) as error:
        return {'valid': False, 'reason': str(error)}
