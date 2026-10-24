import math
import sys
import unittest
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from meter_reader import ReadingError, angle_reading, draw_result, locate_center, locate_pointer, validate_calibration_image


def point(angle, radius=100, center=(150, 150)):
    angle = math.radians(angle)
    return [center[0] + radius * math.cos(angle), center[1] - radius * math.sin(angle)]


class GeometryTests(unittest.TestCase):
    def test_clockwise_cross_zero_with_nonzero_minimum(self):
        ticks = [{'value': 10, 'point': point(20)}, {'value': 20, 'point': point(0)}, {'value': 30, 'point': point(340)}]
        self.assertAlmostEqual(angle_reading((150, 150), point(350), ticks, 'clockwise'), 25)

    def test_counterclockwise(self):
        ticks = [{'value': 0, 'point': point(350)}, {'value': 10, 'point': point(10)}]
        self.assertAlmostEqual(angle_reading((150, 150), point(0), ticks, 'counterclockwise'), 5)

    def test_outside_arc_is_not_zero(self):
        ticks = [{'value': 0, 'point': point(225)}, {'value': 25, 'point': point(315)}]
        with self.assertRaises(ReadingError):
            angle_reading((150, 150), point(270), ticks, 'clockwise')

    def test_nonmonotonic_ticks_rejected(self):
        ticks = [{'value': 0, 'point': point(200)}, {'value': 10, 'point': point(100)}, {'value': 20, 'point': point(150)}]
        with self.assertRaises(ReadingError):
            angle_reading((150, 150), point(170), ticks, 'clockwise')

    def test_center_cannot_be_pointer(self):
        with self.assertRaises(ReadingError):
            angle_reading((150, 150), (150, 150), [{'value': 0, 'point': point(200)}, {'value': 1, 'point': point(100)}], 'clockwise')

    def test_wrong_image_calibration_rejected(self):
        with self.assertRaises(ReadingError):
            validate_calibration_image(b'new-photo', {'image_sha256': '0' * 64})

    def test_nonfinite_anchor_needs_review(self):
        ticks = [{'value': 0, 'point': [float('nan'), 10]}, {'value': 25, 'point': point(315)}]
        with self.assertRaises(ReadingError):
            angle_reading((150, 150), point(150), ticks, 'clockwise')

    def test_nonfinite_pointer_needs_review(self):
        with self.assertRaises(ReadingError):
            angle_reading((150, 150), [float('nan'), 10],
                          [{'value': 0, 'point': point(225)}, {'value': 25, 'point': point(315)}], 'clockwise')

    def test_full_circle_outer_count_and_zero_seam(self):
        ticks = [{'value': v, 'point': point(90 - v * 3.6)} for v in range(0, 101, 10)]
        self.assertAlmostEqual(angle_reading((150, 150), point(135), ticks, 'clockwise', full_circle=True), 87.5)
        self.assertAlmostEqual(angle_reading((150, 150), point(90), ticks, 'clockwise', full_circle=True), 0)
        self.assertAlmostEqual(angle_reading((150, 150), point(90.36), ticks, 'clockwise', full_circle=True), 99.9)

    def test_full_circle_requires_closing_scale_ray(self):
        ticks = [{'value': 0, 'point': point(90)}, {'value': 100, 'point': point(45)}]
        with self.assertRaises(ReadingError):
            angle_reading((150, 150), point(135), ticks, 'clockwise', full_circle=True)

    def test_full_circle_inconsistent_value_angles_are_rejected(self):
        ticks = [{'value': value, 'point': point(angle)} for value, angle in
                 [(0, 90), (10, 60), (20, 30), (30, 0), (40, -30), (50, -60), (100, 90)]]
        with self.assertRaises(ReadingError):
            angle_reading((150, 150), point(-60), ticks, 'clockwise', full_circle=True)

    def test_single_misclicked_full_circle_anchor_is_reported(self):
        import meter_reader
        ticks = [{'value': v, 'point': point(90 - v * 3.6)} for v in range(0, 101, 10)]
        ticks[9]['point'] = point(133)
        diagnostic = meter_reader.full_circle_calibration_check((150, 150), ticks, 'clockwise')
        self.assertEqual(diagnostic['outlier_anchor_values'], [90.])
        self.assertAlmostEqual(angle_reading((150, 150), point(135), ticks, 'clockwise', full_circle=True), 87.5)


class PointerTests(unittest.TestCase):
    def test_tapered_pressure_pointer_edges_are_one_direction(self):
        root = Path(__file__).resolve().parents[1]
        path = root / 'tests/fixtures/images/facdf822811a429d82e2bf9ca32ae86c.png'
        color = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        center, _ = locate_center(gray, [637.6559, 239.4907, 676.9973, 277.1274])
        tip, diagnostic = locate_pointer(gray, [579.7672, 105.6890, 717.1594, 344.1353],
                                         center, 219.2625, color_image=color)
        angle = math.degrees(math.atan2(center[1] - tip[1], tip[0] - center[0]))
        self.assertGreater(angle, 110)
        self.assertLess(angle, 120)

    def test_two_nearby_long_black_needles_still_need_review(self):
        gray = np.full((400, 400), 230, np.uint8)
        for angle in [80, 92]:
            endpoint = tuple(round(v) for v in point(angle, radius=170, center=(200, 200)))
            cv2.line(gray, (200, 200), endpoint, 15, 5)
        cv2.circle(gray, (200, 200), 12, 100, -1)
        with self.assertRaises(ReadingError):
            locate_pointer(gray, [150, 20, 260, 225], (200, 200), 185)

    def test_long_thin_pointer_wins_over_counterweight(self):
        gray = np.full((400, 400), 230, np.uint8)
        cv2.line(gray, (200, 200), (65, 105), 15, 5)
        cv2.fillConvexPoly(gray, np.array([[200, 200], [248, 215], [238, 250]], np.int32), 15)
        cv2.circle(gray, (200, 200), 13, 110, -1)
        tip, diagnostic = locate_pointer(gray, [50, 90, 265, 265], (200, 200), 185)
        self.assertLess(tip[0], 120)
        self.assertLess(tip[1], 150)
        estimated = math.degrees(math.atan2(200 - tip[1], tip[0] - 200))
        expected = math.degrees(math.atan2(95, -135))
        self.assertLess(abs(estimated - expected), 4)

    def test_blank_image_needs_review(self):
        with self.assertRaises(ReadingError):
            locate_pointer(np.full((400, 400), 230, np.uint8), [40, 40, 360, 360], (200, 200), 185)

    def test_disconnected_fragment_is_not_a_pointer(self):
        gray = np.full((400, 400), 230, np.uint8)
        cv2.line(gray, (120, 120), (75, 75), 15, 4)
        with self.assertRaises(ReadingError):
            locate_pointer(gray, [50, 50, 250, 250], (200, 200), 185)

    def test_short_cross_axis_mark_is_not_second_long_pointer(self):
        gray = np.full((400, 400), 230, np.uint8)
        cv2.line(gray, (200, 200), (285, 77), 15, 5)
        cv2.line(gray, (202, 145), (198, 255), 15, 3)
        cv2.circle(gray, (200, 200), 13, 110, -1)
        tip, diagnostic = locate_pointer(gray, [140, 60, 310, 270], (200, 200), 185)
        self.assertGreater(tip[0], 250)
        self.assertLess(tip[1], 140)

    def test_two_long_needle_directions_still_need_review(self):
        gray = np.full((400, 400), 230, np.uint8)
        cv2.line(gray, (200, 200), (285, 77), 15, 5)
        cv2.line(gray, (200, 200), (75, 110), 15, 5)
        cv2.circle(gray, (200, 200), 13, 110, -1)
        with self.assertRaises(ReadingError):
            locate_pointer(gray, [50, 60, 310, 270], (200, 200), 185)

    def test_user_low_resolution_gauge_pointer(self):
        root = Path(__file__).resolve().parents[1]
        image = root / 'samples/images/0.jpg'
        color = cv2.imdecode(np.fromfile(image, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        center, diagnostic = locate_center(gray, [152.2883, 120.16, 181.823, 147.5916])
        tip, diagnostic = locate_pointer(gray, [126.9269, 58.6377, 231.6062, 181.2513], center, 118.8626)
        self.assertGreater(tip[0], center[0])
        self.assertLess(tip[1], center[1])
        direction = math.degrees(math.atan2(center[1] - tip[1], tip[0] - center[0]))
        # Broad visual direction bound; this is not a meter accuracy tolerance.
        self.assertLess(abs(direction - 53.13), 5)

    def test_red_pointer_isolated_from_gray_rust_and_text(self):
        root = Path(__file__).resolve().parents[1]
        image = root / 'tests/fixtures/images/256bd59c22077c159c08d52135a72fef.jpeg'
        color = cv2.imdecode(np.fromfile(image, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        center, diagnostic = locate_center(gray, [423.26, 388.7426, 500.4958, 463.0374])
        tip, diagnostic = locate_pointer(gray, [232.3074, 290.0804, 594.0978, 527.7381],
                                         center, 377.6467, color_image=color)
        direction = math.degrees(math.atan2(center[1] - tip[1], tip[0] - center[0]))
        self.assertLess(abs(direction - 149), 5)
        self.assertEqual(diagnostic['method'], 'colored_component_radial_direction')

    def test_two_long_red_needles_remain_ambiguous(self):
        color = np.full((400, 400, 3), 230, np.uint8)
        cv2.line(color, (200, 200), (285, 77), (0, 0, 230), 7)
        cv2.line(color, (200, 200), (75, 110), (0, 0, 230), 7)
        cv2.circle(color, (200, 200), 13, (0, 0, 230), -1)
        with self.assertRaises(ReadingError):
            locate_pointer(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), [50, 60, 310, 270],
                           (200, 200), 185, color_image=color)

    def test_red_reference_and_black_measuring_needles_need_review(self):
        for shade in [15, 110, 150]:
            with self.subTest(achromatic_shaft_value=shade):
                color = np.full((400, 400, 3), 230, np.uint8)
                cv2.line(color, (200, 200), (285, 77), (0, 0, 230), 7)
                cv2.line(color, (200, 200), (75, 110), (shade, shade, shade), 7)
                cv2.circle(color, (200, 200), 13, (0, 0, 230), -1)
                with self.assertRaises(ReadingError):
                    locate_pointer(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), [50, 60, 310, 270],
                                   (200, 200), 185, color_image=color)

    def test_colored_axle_cap_does_not_replace_black_pointer(self):
        color = np.full((400, 400, 3), 230, np.uint8)
        cv2.line(color, (200, 200), (65, 105), (15, 15, 15), 5)
        cv2.fillConvexPoly(color, np.array([[200, 200], [248, 215], [238, 250]], np.int32), (15, 15, 15))
        cv2.circle(color, (200, 200), 13, (40, 140, 190), -1)
        tip, diagnostic = locate_pointer(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), [50, 90, 265, 265],
                                         (200, 200), 185, color_image=color)
        self.assertLess(tip[0], 120)
        self.assertEqual(diagnostic['method'], 'near_axis_long_edge_cluster')


class CliFailureTests(unittest.TestCase):
    def test_failed_configuration_replaces_latest_record(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
        import read_gauge
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            output = folder / 'output'
            output.mkdir()
            (output / 'result.json').write_text('{"status":"readable","value":6.4}', encoding='utf-8')
            code = read_gauge.main(['--calibration', str(folder / 'missing.json'), '--output', str(output)])
            self.assertEqual(code, 2)
            result = json.loads((output / 'result.json').read_text(encoding='utf-8'))
            self.assertEqual(result['status'], 'needs_review')
            self.assertIsNone(result['value'])

    def test_malformed_json_configuration_is_recorded(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
        import read_gauge
        for text in ['null', '[]', '{"image":42}']:
            with self.subTest(configuration=text), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                profile = folder / 'bad.json'
                profile.write_text(text, encoding='utf-8')
                output = folder / 'output'
                output.mkdir()
                (output / 'result.json').write_text('{"status":"readable","value":6.4}', encoding='utf-8')
                self.assertEqual(read_gauge.main(['--calibration', str(profile), '--output', str(output)]), 2)
                result = json.loads((output / 'result.json').read_text(encoding='utf-8'))
                self.assertEqual(result['status'], 'needs_review')
                self.assertIsNone(result['value'])


class EvidenceTests(unittest.TestCase):
    def test_footer_does_not_cover_original_gauge_image(self):
        image = np.full((100, 200, 3), 125, np.uint8)
        result = {'detections': [], 'geometry': {}, 'display_value': '0.38 MPa'}
        marked = draw_result(image, result)
        self.assertGreater(marked.shape[0], image.shape[0])
        self.assertTrue(np.array_equal(marked[:image.shape[0]], image))


class DetectionAssociationTests(unittest.TestCase):
    def test_nested_counterweight_proposal_keeps_complete_needle_roi(self):
        import meter_reader
        detections = [
            {'label': 'panel', 'box': [0, 0, 300, 300], 'score': .9},
            {'label': 'needle', 'box': [60, 120, 240, 180], 'score': .5},
            {'label': 'needle', 'box': [150, 130, 241, 175], 'score': .99},
            {'label': 'center_region', 'box': [140, 140, 160, 160], 'score': .8},
        ]
        _, needle, _, diagnostic = meter_reader.select_gauge_detections(detections)
        self.assertEqual(needle, [60, 120, 240, 180])
        self.assertEqual(diagnostic['needle_proposal_count'], 2)

    def test_partially_overlapping_needle_rois_remain_ambiguous(self):
        import meter_reader
        detections = [
            {'label': 'panel', 'box': [0, 0, 300, 300], 'score': .9},
            {'label': 'needle', 'box': [60, 120, 180, 180], 'score': .8},
            {'label': 'needle', 'box': [150, 120, 240, 180], 'score': .7},
            {'label': 'center_region', 'box': [140, 140, 160, 160], 'score': .8},
        ]
        with self.assertRaises(ReadingError):
            meter_reader.select_gauge_detections(detections)

    def test_nested_rois_do_not_hide_two_actual_long_needles(self):
        import meter_reader
        detections = [
            {'label': 'panel', 'box': [0, 0, 400, 400], 'score': .9},
            {'label': 'needle', 'box': [60, 60, 340, 240], 'score': .8},
            {'label': 'needle', 'box': [190, 60, 340, 230], 'score': .7},
            {'label': 'center_region', 'box': [187, 187, 213, 213], 'score': .8},
        ]
        _, needle, _, _ = meter_reader.select_gauge_detections(detections)
        gray = np.full((400, 400), 230, np.uint8)
        cv2.line(gray, (200, 200), (90, 90), 15, 5)
        cv2.line(gray, (200, 200), (310, 90), 15, 5)
        cv2.circle(gray, (200, 200), 12, 100, -1)
        with self.assertRaises(ReadingError):
            locate_pointer(gray, needle, (200, 200), 185)

    def select(self, centers):
        import meter_reader
        detections = [
            {'label': 'panel', 'box': [0, 0, 300, 300], 'score': .9},
            {'label': 'needle', 'box': [60, 60, 240, 180], 'score': .8},
            *centers,
        ]
        return meter_reader.select_gauge_detections(detections)

    def test_outside_needle_stop_pin_is_not_axle_even_with_high_score(self):
        actual = {'label': 'center_region', 'box': [135, 135, 165, 165], 'score': .5}
        stop_pin = {'label': 'center_region', 'box': [75, 230, 95, 250], 'score': .99}
        panel, needle, center, diagnostic = self.select([actual, stop_pin])
        self.assertEqual(center, actual['box'])
        self.assertEqual(len(diagnostic['ignored_center_boxes']), 1)

    def test_two_centers_inside_needle_remain_ambiguous(self):
        centers = [{'label': 'center_region', 'box': b, 'score': .8}
                   for b in [[130, 130, 155, 155], [170, 130, 195, 155]]]
        with self.assertRaises(ReadingError):
            self.select(centers)

    def test_no_axle_in_needle_needs_review(self):
        with self.assertRaises(ReadingError):
            self.select([{'label': 'center_region', 'box': [75, 230, 95, 250], 'score': .99}])


class DetectorIntegrationTests(unittest.TestCase):
    def test_thermometer_nested_counterweight_proposal_can_be_read(self):
        from meter_reader import read_image
        root = Path(__file__).resolve().parents[1]
        calibration = json.loads((root / 'tests/fixtures/photo_bd22a1e4d6991d28.json').read_text(encoding='utf-8'))
        result = read_image(root / calibration['image'], calibration, root / 'models/annotation/meter_labeler_v3/best.pt')
        self.assertEqual(result['status'], 'readable', result.get('reason'))
        self.assertAlmostEqual(result['value'], 22, delta=2)
        self.assertEqual(sum(d['label'] == 'needle' for d in result['detections']), 2)
        self.assertEqual(sum(d.get('ignored_for_geometry', False) for d in result['detections']), 1)

    def test_pressure_photo_duplicate_proposals_do_not_block_reading(self):
        from meter_reader import read_image
        root = Path(__file__).resolve().parents[1]
        image = root / 'tests/fixtures/images/20210702123915383.jpg'
        calibration = json.loads((root / 'tests/fixtures/photo_7e443834cf784109.json').read_text(encoding='utf-8'))
        result = read_image(image, calibration, root / 'models/annotation/meter_labeler_v3/best.pt')
        self.assertEqual(result['status'], 'readable', result.get('reason'))
        self.assertAlmostEqual(result['value'], 2, delta=.02)
        for label in ('panel', 'needle', 'center_region'):
            self.assertEqual(sum(item['label'] == label for item in result['detections']), 1)


class FullCircleTests(unittest.TestCase):
    def test_small_reflective_axle_can_recover_with_ring_constraint(self):
        root = Path(__file__).resolve().parents[1]
        image = root / 'tests/fixtures/images/38454c244b404646a59ca8ae71503739.jpeg'
        color = cv2.imdecode(np.fromfile(image, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        center, diagnostic = locate_center(gray, [346.9202, 275.5024, 371.7979, 297.0043],
            expected_center=[355.6633, 284.7124], expected_tolerance=10.65, allow_small_cap_recovery=True)
        self.assertLess(np.linalg.norm(np.asarray(center) - [356, 285]), 5)
        self.assertEqual(diagnostic['method'], 'axle_cap_hough_circle_small_reflective')

    def test_small_cap_recovery_does_not_return_box_center_on_blank(self):
        with self.assertRaises(ReadingError):
            locate_center(np.full((100, 100), 230, np.uint8), [35, 35, 65, 65],
                expected_center=[50, 50], expected_tolerance=8, allow_small_cap_recovery=True)

    def test_calibration_ring_selects_panel_instead_of_background_edge(self):
        import meter_reader
        detections = [
            {'label': 'panel', 'box': [0, 0, 300, 300], 'score': .7},
            {'label': 'panel', 'box': [400, 0, 430, 100], 'score': .99},
            {'label': 'needle', 'box': [70, 50, 170, 190], 'score': .8},
        ]
        panel, needle, center, diagnostic = meter_reader.select_full_circle_regions(
            detections, ring_center=[150, 150], ring_radius=100)
        self.assertEqual(panel, [0, 0, 300, 300])
        self.assertEqual(diagnostic['ignored_panel_boxes'], [[400, 0, 430, 100]])

    def test_two_panels_matching_calibration_remain_ambiguous(self):
        import meter_reader
        detections = [
            {'label': 'panel', 'box': [0, 0, 300, 300], 'score': .7},
            {'label': 'panel', 'box': [10, 10, 290, 290], 'score': .99},
            {'label': 'needle', 'box': [70, 50, 170, 190], 'score': .8},
        ]
        with self.assertRaises(ReadingError):
            meter_reader.select_full_circle_regions(detections, ring_center=[150, 150], ring_radius=100)

    def test_inner_counter_ring_is_not_accepted_as_outer_ring(self):
        import meter_reader
        with self.assertRaises(ReadingError):
            meter_reader.validate_ring_for_panel((150, 150), 30, [0, 0, 300, 300])

    def test_outer_ring_cannot_extend_far_outside_panel(self):
        import meter_reader
        with self.assertRaises(ReadingError):
            meter_reader.validate_ring_for_panel((50, 150), 100, [0, 0, 300, 300])

    def test_calibration_ring_center_is_recovered(self):
        import meter_reader
        ticks = [{'value': v, 'point': point(90 - v * 3.6)} for v in range(0, 101, 10)]
        center, radius, diagnostic = meter_reader.fit_calibration_ring(ticks)
        self.assertTrue(np.allclose(center, [150, 150]))
        self.assertAlmostEqual(radius, 100)

    def test_degenerate_ring_is_rejected(self):
        import meter_reader
        with self.assertRaises(ReadingError):
            meter_reader.fit_calibration_ring([{'value': i, 'point': [i, 1]} for i in range(7)])

    def test_overlapping_needle_proposals_are_one_region(self):
        import meter_reader
        detections = [{'label': 'panel', 'box': [0, 0, 300, 300], 'score': .8},
                      {'label': 'needle', 'box': [70, 50, 160, 180], 'score': .7},
                      {'label': 'needle', 'box': [80, 50, 170, 190], 'score': .6}]
        panel, needle, center, diagnostic = meter_reader.select_full_circle_regions(detections)
        self.assertEqual(needle, [70, 50, 170, 190])
        self.assertIsNone(center)

    def test_separate_needle_regions_are_not_blindly_merged(self):
        import meter_reader
        detections = [{'label': 'panel', 'box': [0, 0, 300, 300], 'score': .8},
                      {'label': 'needle', 'box': [10, 10, 50, 50], 'score': .7},
                      {'label': 'needle', 'box': [200, 200, 250, 250], 'score': .6}]
        with self.assertRaises(ReadingError):
            meter_reader.select_full_circle_regions(detections)

    def test_long_outer_hand_and_short_counter(self):
        import meter_reader
        gray = np.full((400, 400), 230, np.uint8)
        cv2.line(gray, (200, 200), (90, 90), 15, 5)
        cv2.line(gray, (200, 200), (200, 160), 15, 5)
        cv2.circle(gray, (200, 200), 12, 100, -1)
        tip, diagnostic = meter_reader.locate_outer_pointer(gray, [70, 70, 230, 230], (200, 200), 170)
        self.assertLess(tip[0], 130)
        self.assertLess(tip[1], 130)

    def test_two_long_outer_hands_need_review(self):
        import meter_reader
        gray = np.full((400, 400), 230, np.uint8)
        cv2.line(gray, (200, 200), (90, 90), 15, 5)
        cv2.line(gray, (200, 200), (310, 90), 15, 5)
        cv2.circle(gray, (200, 200), 12, 100, -1)
        with self.assertRaises(ReadingError):
            meter_reader.locate_outer_pointer(gray, [70, 70, 330, 230], (200, 200), 170)


if __name__ == '__main__':
    unittest.main()
