import json
from datetime import datetime

import cv2
import numpy as np
import pytest

from webcam_app import (AppState, FixtureFrameSource, GaugeReading,
                        ReviewController, ReviewSession, WebcamFrameSource,
                        parse_args)


def controller_for(tmp_path, frame, reader):
    return ReviewController(frame_source=frame.copy,
                            reader=reader,
                            session=ReviewSession(tmp_path))


def read_single_record(controller):
    return json.loads(controller.record_path.read_text().strip())


def test_operator_can_confirm_a_captured_reading(tmp_path):
    frame = np.full((24, 32, 3), (17, 34, 51), dtype=np.uint8)
    visualization = np.full((24, 32, 3), (90, 80, 70), dtype=np.uint8)

    def read_gauge(captured_frame):
        np.testing.assert_array_equal(captured_frame, frame)
        return GaugeReading(value=2.75,
                            unit="bar",
                            visualization=visualization)

    controller = controller_for(tmp_path, frame, read_gauge)

    preview = controller.next_frame()
    reading = controller.capture()
    controller.confirm()

    np.testing.assert_array_equal(preview, frame)
    assert reading.value == 2.75
    assert reading.unit == "bar"
    assert controller.state is AppState.PREVIEW

    records = [
        json.loads(line)
        for line in controller.record_path.read_text().splitlines()
    ]
    assert len(records) == 1
    assert records[0]["automatic_value"] == 2.75
    assert records[0]["unit"] == "bar"
    assert records[0]["decision"] == "confirmed"
    assert records[0]["corrected_value"] is None
    datetime.fromisoformat(records[0]["timestamp"])

    saved_frame = cv2.imread(
        str(controller.session_path / records[0]["frame_path"]))
    np.testing.assert_array_equal(saved_frame, frame)


def test_operator_can_correct_a_captured_reading(tmp_path):
    frame = np.zeros((16, 20, 3), dtype=np.uint8)

    def read_gauge(captured_frame):
        return GaugeReading(value=8.5,
                            unit="psi",
                            visualization=captured_frame)

    controller = controller_for(tmp_path, frame, read_gauge)

    controller.next_frame()
    controller.capture()
    controller.correct(9.25)

    record = read_single_record(controller)
    assert record["automatic_value"] == 8.5
    assert record["unit"] == "psi"
    assert record["decision"] == "corrected"
    assert record["corrected_value"] == 9.25
    assert controller.state is AppState.PREVIEW


def test_operator_can_reject_a_captured_reading(tmp_path):
    frame = np.zeros((12, 18, 3), dtype=np.uint8)

    def read_gauge(captured_frame):
        return GaugeReading(value=120.0,
                            unit="mbar",
                            visualization=captured_frame)

    controller = controller_for(tmp_path, frame, read_gauge)

    controller.next_frame()
    controller.capture()
    controller.reject()

    record = read_single_record(controller)
    assert record["automatic_value"] == 120.0
    assert record["unit"] == "mbar"
    assert record["decision"] == "rejected"
    assert record["corrected_value"] is None
    assert controller.state is AppState.PREVIEW


def test_reading_failure_is_recorded_and_returns_to_preview(tmp_path):
    frame = np.full((10, 14, 3), 128, dtype=np.uint8)

    def fail_to_read(captured_frame):
        raise RuntimeError("OCR could not find two scale labels")

    controller = controller_for(tmp_path, frame, fail_to_read)

    controller.next_frame()
    reading = controller.capture()

    assert reading is None
    assert controller.last_error == "OCR could not find two scale labels"
    assert controller.state is AppState.PREVIEW

    record = read_single_record(controller)
    assert record["automatic_value"] is None
    assert record["unit"] is None
    assert record["decision"] == "failed"
    assert record["corrected_value"] is None
    assert record["error"] == "OCR could not find two scale labels"
    saved_frame = cv2.imread(
        str(controller.session_path / record["frame_path"]))
    np.testing.assert_array_equal(saved_frame, frame)


def test_fixture_image_can_drive_a_complete_review(tmp_path):
    fixture = np.full((18, 26, 3), (12, 120, 240), dtype=np.uint8)
    fixture_path = tmp_path / "gauge.png"
    assert cv2.imwrite(str(fixture_path), fixture)

    def read_gauge(captured_frame):
        return GaugeReading(value=-0.4,
                            unit="MPa",
                            visualization=captured_frame)

    controller = ReviewController(
        frame_source=FixtureFrameSource(fixture_path),
        reader=read_gauge,
        session=ReviewSession(tmp_path / "sessions"))

    np.testing.assert_array_equal(controller.next_frame(), fixture)
    controller.capture()
    controller.confirm()

    record = read_single_record(controller)
    assert record["automatic_value"] == -0.4
    assert record["decision"] == "confirmed"


def test_cli_defaults_match_the_file_pipeline():
    args = parse_args([])

    assert args.camera == 0
    assert args.detection_model == "models/gauge_detection_model.pt"
    assert args.key_point_model == "models/key_point_model.pt"
    assert args.segmentation_model == "models/segmentation_model.pt"
    assert args.base_path == "webcam_sessions"


def test_cli_can_select_camera_models_and_output():
    args = parse_args([
        "--camera", "2", "--detection_model", "detector.pt",
        "--key_point_model", "keypoints.pt", "--segmentation_model",
        "needle.pt", "--base_path", "records"
    ])

    assert args.camera == 2
    assert args.detection_model == "detector.pt"
    assert args.key_point_model == "keypoints.pt"
    assert args.segmentation_model == "needle.pt"
    assert args.base_path == "records"


def test_unavailable_camera_reports_its_device_number():
    with pytest.raises(RuntimeError,
                       match="Could not open camera device 9999"):
        WebcamFrameSource(9999)
