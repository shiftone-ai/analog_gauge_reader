#!/usr/bin/env python3

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

WINDOW_NAME = "Analog gauge reader"
FINAL_VISUALIZATION_FILE = "ellipse_results_final.jpg"


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(
        description="Read an analog gauge from a Web camera and review it")
    parser.add_argument('--camera',
                        type=int,
                        default=0,
                        help="Camera device number (default: 0)")
    parser.add_argument('--detection_model',
                        default="models/gauge_detection_model.pt",
                        help="Path to detection model")
    parser.add_argument('--key_point_model',
                        default="models/key_point_model.pt",
                        help="Path to key point model")
    parser.add_argument('--segmentation_model',
                        default="models/segmentation_model.pt",
                        help="Path to segmentation model")
    parser.add_argument('--base_path',
                        default="webcam_sessions",
                        help="Directory where sessions are stored")
    return parser.parse_args(arguments)


class AppState(Enum):
    PREVIEW = "preview"
    PROCESSING = "processing"
    REVIEW = "review"


@dataclass(frozen=True)
class GaugeReading:
    value: float
    unit: Optional[str]
    visualization: np.ndarray


@dataclass(frozen=True)
class ModelPaths:
    detection: str
    key_point: str
    segmentation: str


class ReviewDecision(Enum):
    CONFIRMED = "confirmed"
    CORRECTED = "corrected"
    REJECTED = "rejected"
    FAILED = "failed"


class ReviewSession:

    def __init__(self, output_base: Path):
        session_name = datetime.now(
            timezone.utc).strftime("session_%Y%m%dT%H%M%S%fZ")
        self.path = Path(output_base) / session_name
        self.path.mkdir(parents=True)
        self._shot_number = 0
        self._current_shot_path = None

    @property
    def current_shot_path(self) -> Path:
        if self._current_shot_path is None:
            raise RuntimeError("No capture is in progress")
        return self._current_shot_path

    def start_shot(self) -> Path:
        self._shot_number += 1
        self._current_shot_path = self.path / f"shot-{self._shot_number:04d}"
        return self._current_shot_path


class FixtureFrameSource:

    def __init__(self, image_path: Path):
        self._frame = cv2.imread(str(image_path))
        if self._frame is None:
            raise RuntimeError(f"Could not read fixture image: {image_path}")

    def __call__(self) -> np.ndarray:
        return self._frame.copy()


class WebcamFrameSource:

    def __init__(self, device_number: int):
        self._device_number = device_number
        self._capture = cv2.VideoCapture(device_number)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"Could not open camera device {device_number}")

    def __call__(self) -> np.ndarray:
        success, frame = self._capture.read()
        if not success or frame is None:
            raise RuntimeError(
                f"Could not read from camera device {self._device_number}")
        return frame

    @property
    def device_number(self) -> int:
        return self._device_number

    def switch_to(self, device_number: int) -> None:
        if device_number == self._device_number:
            return

        next_capture = cv2.VideoCapture(device_number)
        if not next_capture.isOpened():
            next_capture.release()
            raise RuntimeError(f"Could not open camera device {device_number}")

        success, frame = next_capture.read()
        if not success or frame is None:
            next_capture.release()
            raise RuntimeError(
                f"Could not read from camera device {device_number}")

        previous_capture = self._capture
        self._capture = next_capture
        self._device_number = device_number
        previous_capture.release()

    def release(self) -> None:
        self._capture.release()


class PipelineGaugeReader:

    def __init__(self, model_paths: ModelPaths, session: ReviewSession):
        self._model_paths = model_paths
        self._session = session

    def __call__(self, frame: np.ndarray) -> GaugeReading:
        # Importing the pipeline loads the model stack, so keep it outside the
        # lightweight controller and CLI import path.
        from pipeline import process_image  # pylint: disable=import-outside-toplevel
        from matplotlib import pyplot  # pylint: disable=import-outside-toplevel

        shot_path = self._session.current_shot_path
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            result = process_image(
                image=rgb_frame,
                image_is_raw=True,
                detection_model_path=self._model_paths.detection,
                key_point_model_path=self._model_paths.key_point,
                segmentation_model_path=self._model_paths.segmentation,
                run_path=str(shot_path),
                debug=True,
                eval_mode=False)
        finally:
            pyplot.close('all')
        if result["value"] is None:
            raise RuntimeError("The pipeline did not produce a gauge value")

        visualization_path = shot_path / FINAL_VISUALIZATION_FILE
        visualization = cv2.imread(str(visualization_path))
        if visualization is None:
            raise RuntimeError(
                f"The pipeline did not create {FINAL_VISUALIZATION_FILE}")
        return GaugeReading(value=float(result["value"]),
                            unit=result["unit"],
                            visualization=visualization)


class ReviewController:

    def __init__(self, frame_source: Callable[[], np.ndarray],
                 reader: Callable[[np.ndarray],
                                  GaugeReading], session: ReviewSession):
        self._frame_source = frame_source
        self._reader = reader
        self._session = session
        self._current_frame = None
        self._current_reading = None
        self._current_frame_path = None
        self.last_error = None
        self.state = AppState.PREVIEW

        self.session_path = session.path
        self.record_path = self.session_path / "events.jsonl"

    def next_frame(self) -> np.ndarray:
        if self.state is not AppState.PREVIEW:
            raise RuntimeError("Frames are only available during preview")
        frame = self._frame_source()
        if frame is None:
            raise RuntimeError("Camera did not return a frame")
        self.last_error = None
        self._current_frame = frame.copy()
        return self._current_frame.copy()

    def capture(self) -> Optional[GaugeReading]:
        if self.state is not AppState.PREVIEW:
            raise RuntimeError("A reading is already in progress")
        if self._current_frame is None:
            raise RuntimeError("No preview frame is available to capture")

        self.state = AppState.PROCESSING
        shot_path = self._session.start_shot()
        try:
            self._current_reading = self._reader(self._current_frame.copy())
        except Exception as error:  # pylint: disable=broad-except
            self.last_error = str(error) or type(error).__name__
            self._save_current_frame(shot_path)
            self._append_event(ReviewDecision.FAILED, error=self.last_error)
            self._return_to_preview()
            return None

        self._save_current_frame(shot_path)
        self.state = AppState.REVIEW
        return self._current_reading

    def _save_current_frame(self, shot_path: Path) -> None:
        shot_path.mkdir(exist_ok=True)
        self._current_frame_path = shot_path / "frame.png"
        if not cv2.imwrite(str(self._current_frame_path), self._current_frame):
            raise RuntimeError("Could not save captured frame")

    def confirm(self) -> None:
        if self.state is not AppState.REVIEW:
            raise RuntimeError("There is no reading to confirm")
        self._append_event(ReviewDecision.CONFIRMED)
        self._return_to_preview()

    def correct(self, corrected_value: float) -> None:
        if self.state is not AppState.REVIEW:
            raise RuntimeError("There is no reading to correct")
        self._append_event(ReviewDecision.CORRECTED,
                           corrected_value=float(corrected_value))
        self._return_to_preview()

    def reject(self) -> None:
        if self.state is not AppState.REVIEW:
            raise RuntimeError("There is no reading to reject")
        self._append_event(ReviewDecision.REJECTED)
        self._return_to_preview()

    def _append_event(self,
                      decision: ReviewDecision,
                      corrected_value: Optional[float] = None,
                      error: Optional[str] = None) -> None:
        reading = self._current_reading
        record = {
            "timestamp":
            datetime.now(timezone.utc).isoformat(),
            "automatic_value":
            float(reading.value) if reading else None,
            "unit":
            reading.unit if reading else None,
            "decision":
            decision.value,
            "corrected_value":
            corrected_value,
            "frame_path":
            str(self._current_frame_path.relative_to(self.session_path)),
        }
        if error is not None:
            record["error"] = error
        with self.record_path.open("a", encoding="utf-8") as record_file:
            record_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _return_to_preview(self) -> None:
        self._current_frame = None
        self._current_reading = None
        self._current_frame_path = None
        self.state = AppState.PREVIEW


def _draw_footer(image: np.ndarray, text: str) -> np.ndarray:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    cv2.rectangle(canvas, (0, height - 48), (width, height), (20, 20, 20), -1)
    cv2.putText(canvas, text, (18, height - 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _render_review(reading: GaugeReading,
                   correction: Optional[str] = None,
                   correction_error: Optional[str] = None) -> np.ndarray:
    visualization = reading.visualization
    image_height, image_width = visualization.shape[:2]
    canvas_width = max(image_width, 820)
    canvas = np.full((image_height + 180, canvas_width, 3), 24, dtype=np.uint8)
    image_left = (canvas_width - image_width) // 2
    canvas[110:110 + image_height,
           image_left:image_left + image_width] = visualization

    unit = reading.unit or ""
    value_text = f"{reading.value:g} {unit}".strip()
    cv2.putText(canvas, value_text, (24, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.7,
                (255, 255, 255), 3, cv2.LINE_AA)

    footer_y = image_height + 110
    cv2.rectangle(canvas, (0, footer_y), (canvas_width, footer_y + 70),
                  (20, 20, 20), -1)
    if correction is None:
        footer = "[Enter/C] confirm   [E] correct   [R] reject   [Q] quit"
        color = (255, 255, 255)
    else:
        footer = f"Corrected value: {correction}_   [Enter] save   [Esc] cancel"
        color = (90, 220, 255) if correction_error is None else (80, 80, 255)
        if correction_error is not None:
            footer = f"{correction_error}   Value: {correction}_"
    cv2.putText(canvas, footer, (18, footer_y + 43), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, color, 2, cv2.LINE_AA)
    return canvas


def _render_status(frame: np.ndarray, title: str, detail: str) -> np.ndarray:
    canvas = frame.copy()
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], canvas.shape[0]),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.66, canvas, 0.34, 0, canvas)
    cv2.putText(canvas, title, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, detail[:100], (30, 115), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (220, 220, 220), 2, cv2.LINE_AA)
    return canvas


def _read_key(delay: int) -> int:
    key = cv2.waitKey(delay)
    return key & 0xFF if key >= 0 else -1


def _review_loop(controller: ReviewController, reading: GaugeReading) -> bool:
    correction = None
    correction_error = None
    while controller.state is AppState.REVIEW:
        cv2.imshow(WINDOW_NAME,
                   _render_review(reading, correction, correction_error))
        key = _read_key(30)
        if key < 0:
            continue
        if key in (ord('q'), ord('Q')):
            return False

        if correction is not None:
            if key == 27:
                correction = None
                correction_error = None
            elif key in (10, 13):
                try:
                    controller.correct(float(correction))
                except ValueError:
                    correction_error = "Enter a valid number"
            elif key in (8, 127):
                correction = correction[:-1]
                correction_error = None
            elif chr(key) in "0123456789+-." and len(correction) < 24:
                correction += chr(key)
                correction_error = None
            continue

        if key in (10, 13, ord('c'), ord('C')):
            controller.confirm()
        elif key in (ord('e'), ord('E')):
            correction = ""
        elif key in (ord('r'), ord('R')):
            controller.reject()
        elif key == 27:
            return False
    return True


def run_app(controller: ReviewController,
            camera_source: WebcamFrameSource) -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    keep_running = True
    selecting_camera = False
    camera_error = None
    while keep_running:
        frame = controller.next_frame()
        if selecting_camera:
            footer = camera_error or (
                f"Select camera [0-9]   Current: {camera_source.device_number}"
                "   [Esc] cancel")
        else:
            footer = (f"[Space] capture   [C] camera "
                      f"({camera_source.device_number})   [Q/Esc] quit")
        cv2.imshow(WINDOW_NAME, _draw_footer(frame, footer))
        key = _read_key(1)

        if selecting_camera:
            if key in (ord('q'), ord('Q')):
                break
            if key == 27:
                selecting_camera = False
                camera_error = None
                continue
            if ord('0') <= key <= ord('9'):
                try:
                    camera_source.switch_to(int(chr(key)))
                except RuntimeError as error:
                    camera_error = f"{error}   Select camera [0-9]"
                else:
                    selecting_camera = False
                    camera_error = None
            continue

        if key in (27, ord('q'), ord('Q')):
            break
        if key in (ord('c'), ord('C')):
            selecting_camera = True
            camera_error = None
            continue
        if key != ord(' '):
            continue

        cv2.imshow(
            WINDOW_NAME,
            _render_status(frame, "Processing...",
                           "Reading the captured gauge"))
        _read_key(1)
        reading = controller.capture()
        if reading is None:
            cv2.imshow(
                WINDOW_NAME,
                _render_status(frame, "Reading failed", controller.last_error))
            if _read_key(2000) in (27, ord('q'), ord('Q')):
                break
            continue
        keep_running = _review_loop(controller, reading)


def main(arguments=None) -> int:
    args = parse_args(arguments)
    try:
        frame_source = WebcamFrameSource(args.camera)
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    try:
        session = ReviewSession(Path(args.base_path))
        model_paths = ModelPaths(detection=args.detection_model,
                                 key_point=args.key_point_model,
                                 segmentation=args.segmentation_model)
        reader = PipelineGaugeReader(model_paths=model_paths, session=session)
        controller = ReviewController(frame_source=frame_source,
                                      reader=reader,
                                      session=session)
        print(f"Session records: {controller.session_path}")
        run_app(controller, frame_source)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        frame_source.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
