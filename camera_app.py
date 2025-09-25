from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - handled at runtime
    mp = None  # type: ignore

from processing import PROCESSING_MODES, apply_processing


WINDOW_NAME = "Sight"


@dataclass
class AppConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    flip_horizontal: bool = True
    show_fps: bool = True
    processing_mode: str = "none"
    hands_enabled: bool = True
    hand_max_num: int = 2
    hand_min_det_conf: float = 0.6
    hand_min_track_conf: float = 0.6


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Camera app with processing and hand landmarks")
    parser.add_argument("--camera-index", "-c", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", "-W", type=int, default=1280, help="Frame width (default: 1280)")
    parser.add_argument("--height", "-H", type=int, default=720, help="Frame height (default: 720)")
    parser.add_argument("--no-flip", action="store_true", help="Disable horizontal flip (mirror)")
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="none",
        choices=PROCESSING_MODES,
        help=f"Initial processing mode (choices: {', '.join(PROCESSING_MODES)})",
    )
    parser.add_argument("--no-fps", action="store_true", help="Hide FPS overlay")
    parser.add_argument("--hands", dest="hands", action="store_true", help="Enable hand landmarks overlay")
    parser.add_argument("--no-hands", dest="hands", action="store_false", help="Disable hand landmarks overlay")
    parser.set_defaults(hands=True)
    parser.add_argument("--hand-max", type=int, default=2, help="Maximum number of hands to detect")
    parser.add_argument("--hand-det", type=float, default=0.6, help="Min detection confidence [0-1]")
    parser.add_argument("--hand-track", type=float, default=0.6, help="Min tracking confidence [0-1]")

    args = parser.parse_args()
    return AppConfig(
        camera_index=args.camera_index,
        frame_width=args.width,
        frame_height=args.height,
        flip_horizontal=not args.no_flip,
        show_fps=not args.no_fps,
        processing_mode=args.mode,
        hands_enabled=args.hands,
        hand_max_num=args.hand_max,
        hand_min_det_conf=args.hand_det,
        hand_min_track_conf=args.hand_track,
    )


def put_text(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    scale: float = 0.6,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_help_overlay(frame: np.ndarray, mode: str, hands_on: bool) -> None:
    lines = [
        "q: quit  |  m: cycle mode  |  h: toggle hands  |  s: save frame",
        f"mode: {mode}  |  hands: {'on' if hands_on else 'off'}",
    ]
    x, y = 10, 24
    for line in lines:
        put_text(frame, line, (x, y))
        y += 22


def ensure_output_dir() -> str:
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def initialize_camera(config: AppConfig) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        return None
    # Set desired resolution; camera might not honor it exactly
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(config.frame_width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(config.frame_height))
    return cap


def main() -> int:
    config = parse_args()

    cap = initialize_camera(config)
    if cap is None:
        print("ERROR: Could not open camera. Try a different --camera-index.")
        return 1

    # Prepare hand landmark pipeline if available and enabled
    hands_ctx = None
    drawing_utils = None
    if config.hands_enabled and mp is not None:
        mp_hands = mp.solutions.hands
        drawing_utils = mp.solutions.drawing_utils
        hands_ctx = mp_hands.Hands(
            model_complexity=1,
            max_num_hands=int(config.hand_max_num),
            min_detection_confidence=float(config.hand_min_det_conf),
            min_tracking_confidence=float(config.hand_min_track_conf),
        )
    elif config.hands_enabled and mp is None:
        print("WARNING: mediapipe is not installed; hand landmarks disabled.")
        config.hands_enabled = False

    mode_index = PROCESSING_MODES.index(config.processing_mode)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, config.frame_width, config.frame_height)

    out_dir = ensure_output_dir()
    last_time = time.time()
    fps = 0.0
    frame_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("WARNING: Failed to read frame from camera.")
                continue

            if config.flip_horizontal:
                frame = cv2.flip(frame, 1)

            # Apply processing to a copy so we can draw landmarks on it
            mode = PROCESSING_MODES[mode_index]
            processed = apply_processing(frame, mode)

            # Hand landmarks overlay
            if config.hands_enabled and hands_ctx is not None and drawing_utils is not None:
                # Mediapipe expects RGB input
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands_ctx.process(rgb)
                if result.multi_hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                        drawing_utils.draw_landmarks(
                            processed,
                            hand_landmarks,
                            mp.solutions.hands.HAND_CONNECTIONS,
                            drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2),
                        )
                # Optional: show handedness labels
                if result.multi_handedness:
                    h_text = ", ".join(
                        f"{c.classification[0].label}:{c.classification[0].score:.2f}" for c in result.multi_handedness
                    )
                    put_text(processed, h_text, (10, processed.shape[0] - 10))

            # FPS calculation
            frame_count += 1
            now = time.time()
            if now - last_time >= 0.5:
                fps = frame_count / (now - last_time)
                frame_count = 0
                last_time = now

            # Overlays
            draw_help_overlay(processed, mode, config.hands_enabled)
            if config.show_fps:
                put_text(processed, f"FPS: {fps:.1f}", (processed.shape[1] - 140, 24))

            cv2.imshow(WINDOW_NAME, processed)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                mode_index = (mode_index + 1) % len(PROCESSING_MODES)
            elif key == ord("h"):
                if mp is None:
                    print("mediapipe not installed; cannot enable hands.")
                else:
                    config.hands_enabled = not config.hands_enabled
            elif key == ord("s"):
                ts = time.strftime("%Y%m%d-%H%M%S")
                path = os.path.join(out_dir, f"frame-{ts}-{mode}.png")
                cv2.imwrite(path, processed)
                print(f"Saved {path}")

    finally:
        cap.release()
        if hands_ctx is not None:
            hands_ctx.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


