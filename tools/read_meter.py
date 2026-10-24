"""Run the isolated calibration-bound visual meter reader as a tool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import meter_reader
from calibration_binding import BindingError, bind_detections


def read_meter(image, calibration, model=None):
    """Return a structured visual-only meter result.

    The result deliberately does not evaluate operating limits or equipment
    state. Those decisions belong to the rules layer called by an Agent.
    """
    image = Path(image)
    calibration = Path(calibration)
    profile = json.loads(calibration.read_text(encoding="utf-8"))
    model = Path(model) if model else ROOT / "models" / "annotation" / "meter_labeler_v3" / "best.pt"
    original = meter_reader.select_gauge_detections
    meter_reader.select_gauge_detections = lambda detections: bind_detections(profile, detections)
    try:
        result = meter_reader.read_image(image, profile, model)
    finally:
        meter_reader.select_gauge_detections = original
    result["tool"] = "read_meter"
    result["tool_version"] = "calibration-bound-v1"
    result["operating_state_evaluation"] = "not_performed"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Visual-only meter reading tool")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = read_meter(args.image, args.calibration, args.model)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        result = {"tool": "read_meter", "tool_version": "calibration-bound-v1",
                  "status": "needs_review", "value": None, "reason": str(error),
                  "operating_state_evaluation": "not_performed"}
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if result.get("status") == "readable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
