"""Bind model detections to the image-bound calibration target."""
from __future__ import annotations

import math

import numpy as np


class BindingError(ValueError):
    pass


def _middle(box):
    return [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]


def _inside(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def bind_detections(calibration, detections):
    ticks = [t["point"] for t in calibration["ticks"]]
    points = np.asarray(ticks, dtype=float)
    matrix = np.column_stack([2 * points[:, 0], 2 * points[:, 1], np.ones(len(points))])
    fit_x, fit_y, constant = np.linalg.lstsq(matrix, (points ** 2).sum(axis=1), rcond=None)[0]
    fit_radius_sq = constant + fit_x * fit_x + fit_y * fit_y
    fitted_center = np.asarray([fit_x, fit_y], dtype=float)
    fitted_radius = math.sqrt(float(fit_radius_sq)) if fit_radius_sq > 0 else 0.0

    panels = [d for d in detections if d["label"] == "panel"]
    candidates = []
    for detection in panels:
        box = detection["box"]
        margin = max(2.0, min(box[2] - box[0], box[3] - box[1]) * 0.02)
        if all(box[0] - margin <= p[0] <= box[2] + margin and
               box[1] - margin <= p[1] <= box[3] + margin for p in ticks):
            candidates.append(detection)
    if len(candidates) != 1:
        raise BindingError(f"Calibration anchors leave {len(candidates)} panel candidates; human review required.")

    panel = candidates[0]["box"]
    related = [d for d in detections if _inside(_middle(d["box"]), panel)]
    needles = [d for d in related if d["label"] == "needle"]
    if not needles:
        raise BindingError("Missing or ambiguous needle detections inside calibrated panel.")

    def area(box):
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    needle = max(needles, key=lambda d: area(d["box"]))
    for detection in needles:
        box = detection["box"]
        intersection = (max(0, min(needle["box"][2], box[2]) - max(needle["box"][0], box[0])) *
                        max(0, min(needle["box"][3], box[3]) - max(needle["box"][1], box[1])))
        if area(box) <= 0 or intersection / area(box) < .95:
            raise BindingError("Missing or ambiguous needle detections inside calibrated panel.")

    all_centers = [d for d in related if d["label"] == "center_region"]
    centers = [d for d in all_centers if _inside(_middle(d["box"]), needle["box"])]
    center_selection = "single_center_candidate"
    if len(centers) > 1 and fitted_radius > 0:
        ranked = sorted(centers, key=lambda d: float(np.linalg.norm(np.asarray(_middle(d["box"])) - fitted_center)))
        distances = [float(np.linalg.norm(np.asarray(_middle(d["box"])) - fitted_center)) for d in ranked]
        if distances[0] <= fitted_radius * .12 and distances[1] >= max(distances[0] * 3.0, fitted_radius * .30):
            centers = [ranked[0]]
            center_selection = "nearest_to_calibration_circle_center_with_clear_margin"
    if len(centers) != 1:
        raise BindingError("Missing or ambiguous axle center inside calibrated needle region.")

    return panel, needle["box"], centers[0]["box"], {
        "method": "calibration_anchor_panel_binding",
        "panel_selection": "all_user_calibration_anchors_inside_panel",
        "calibration_anchor_count": len(ticks),
        "calibration_circle_center": fitted_center.tolist(),
        "calibration_circle_radius_px": fitted_radius,
        "candidate_panel_count": len(candidates),
        "ignored_panel_boxes": [d["box"] for d in panels if d is not candidates[0]],
        "needle_proposal_count": len(needles),
        "needle_roi_selection": "complete_roi_with_contained_subproposals" if len(needles) > 1 else "single_detection",
        "ignored_needle_boxes": [d["box"] for d in needles if d is not needle],
        "ignored_center_boxes": [d["box"] for d in all_centers if d not in centers],
        "center_selection": center_selection,
    }
