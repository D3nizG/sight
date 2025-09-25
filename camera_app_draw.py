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

try:
    from .processing import PROCESSING_MODES, apply_processing
    from .drawing_tracker import DrawingTracker
    from .drawing_classifier import DrawingClassifier
except ImportError:
    # Allow direct execution from within the sight directory
    from processing import PROCESSING_MODES, apply_processing
    from drawing_tracker import DrawingTracker
    from drawing_classifier import DrawingClassifier


WINDOW_NAME = "Sight - Drawing Recognition"


@dataclass
class AppConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    flip_horizontal: bool = True
    show_fps: bool = True
    processing_mode: str = "none"
    hands_enabled: bool = True
    hand_max_num: int = 1  # Use only 1 hand for drawing
    hand_min_det_conf: float = 0.6
    hand_min_track_conf: float = 0.6
    drawing_enabled: bool = True
    show_predictions: bool = True


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Camera app with drawing recognition")
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
    parser.add_argument("--hand-max", type=int, default=1, help="Maximum number of hands to detect")
    parser.add_argument("--hand-det", type=float, default=0.6, help="Min detection confidence [0-1]")
    parser.add_argument("--hand-track", type=float, default=0.6, help="Min tracking confidence [0-1]")
    parser.add_argument("--no-drawing", action="store_true", help="Disable drawing recognition")
    parser.add_argument("--no-predictions", action="store_true", help="Hide prediction text")

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
        drawing_enabled=not args.no_drawing,
        show_predictions=not args.no_predictions,
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


def draw_help_overlay(frame: np.ndarray, mode: str, hands_on: bool, drawing_on: bool) -> None:
    lines = [
        "q: quit  |  m: cycle mode  |  h: toggle hands  |  d: toggle drawing  |  c: clear  |  s: save",
        f"mode: {mode}  |  hands: {'on' if hands_on else 'off'}  |  drawing: {'on' if drawing_on else 'off'}",
    ]
    x, y = 10, 24
    for line in lines:
        put_text(frame, line, (x, y))
        y += 22


def draw_prediction_overlay(frame: np.ndarray, predictions_text: str) -> None:
    """Draw the prediction text in a prominent location."""
    if not predictions_text or predictions_text == "No prediction":
        return
    
    # Position in upper right area
    text_size = cv2.getTextSize(predictions_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    x = frame.shape[1] - text_size[0] - 20
    y = 80
    
    # Draw background rectangle
    padding = 10
    cv2.rectangle(frame, 
                 (x - padding, y - text_size[1] - padding),
                 (x + text_size[0] + padding, y + padding),
                 (0, 0, 0), -1)  # Black background
    cv2.rectangle(frame,
                 (x - padding, y - text_size[1] - padding),
                 (x + text_size[0] + padding, y + padding),
                 (255, 255, 255), 2)  # White border
    
    # Draw text
    put_text(frame, predictions_text, (x, y), scale=0.8, color=(0, 255, 255), thickness=2)


def draw_gesture_status(frame: np.ndarray, drawing_tracker: DrawingTracker) -> None:
    """Draw gesture status and drawing info in bottom left."""
    info_lines = []
    
    # Gesture status
    if drawing_tracker.is_drawing:
        info_lines.append("DRAWING: 👉 Index finger pointed & moving")
    else:
        info_lines.append("READY: Point index finger to start drawing")
    
    # Drawing stats
    drawing = drawing_tracker.get_current_drawing()
    if drawing.strokes:
        num_strokes = len(drawing.strokes)
        total_points = sum(len(s.points) for s in drawing.strokes)
        info_lines.append(f"Strokes: {num_strokes} | Points: {total_points}")
    
    # Draw info in bottom left
    x, y_start = 10, frame.shape[0] - 40
    for i, line in enumerate(info_lines):
        y = y_start - (i * 18)  # Stack upward
        put_text(frame, line, (x, y), scale=0.5, color=(200, 200, 200))


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


def save_drawing_data(drawing_tracker: DrawingTracker, current_predictions, output_dir: str) -> None:
    """Save current drawing data to a file."""
    if not drawing_tracker.get_current_drawing().strokes:
        print("No drawing to save.")
        return
    
    ts = time.strftime("%Y%m%d-%H%M%S")
    drawing_path = os.path.join(output_dir, f"drawing-{ts}.txt")
    
    with open(drawing_path, 'w') as f:
        drawing = drawing_tracker.get_current_drawing()
        f.write(f"Drawing saved at: {time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
        f.write(f"Total strokes: {len(drawing.strokes)}\\n")
        
        if current_predictions:
            f.write(f"\\nPredictions:\\n")
            for name, confidence in current_predictions:
                f.write(f"  {name}: {confidence:.2%}\\n")
        
        f.write(f"\\nDrawing details:\\n")
        for i, stroke in enumerate(drawing.strokes):
            f.write(f"Stroke {i+1}: {len(stroke.points)} points, duration: {stroke.duration():.2f}s\\n")
            
        # Save normalized coordinates for potential ML training
        normalized = drawing.normalize(256)
        f.write(f"\\nNormalized drawing (256x256):\\n")
        for i, stroke in enumerate(normalized.strokes):
            f.write(f"Stroke {i+1} points: ")
            for point in stroke.points:
                f.write(f"({point.x:.1f},{point.y:.1f}) ")
            f.write("\\n")
    
    print(f"Drawing data saved: {drawing_path}")


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

    # Initialize drawing recognition components
    drawing_tracker = DrawingTracker() if config.drawing_enabled else None
    drawing_classifier = DrawingClassifier() if config.drawing_enabled else None
    
    mode_index = PROCESSING_MODES.index(config.processing_mode)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, config.frame_width, config.frame_height)

    out_dir = ensure_output_dir()
    last_time = time.time()
    fps = 0.0
    frame_count = 0
    
    # Classification state
    last_classification_time = 0.0
    classification_interval = 0.5  # Classify every 0.5 seconds
    current_predictions = []
    predictions_text = "Point index finger to start drawing"
    
    print("\\n🎨 Drawing Recognition Controls:")
    print("👉 Point your INDEX FINGER (other fingers curled) to draw")
    print("✋ Open hand or make fist to stop drawing")
    print("📝 Press 'c' to clear and see final prediction")
    print("💾 Press 's' to save drawing data\\n")

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

            # Hand landmarks and drawing tracking
            hand_landmarks = None
            handedness_results = None
            
            if config.hands_enabled and hands_ctx is not None and drawing_utils is not None:
                # Mediapipe expects RGB input
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands_ctx.process(rgb)
                
                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks
                    handedness_results = result.multi_handedness
                    
                    # Draw hand landmarks
                    for hand_idx, landmarks in enumerate(hand_landmarks):
                        drawing_utils.draw_landmarks(
                            processed,
                            landmarks,
                            mp.solutions.hands.HAND_CONNECTIONS,
                            drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2),
                        )

            # Drawing tracking and recognition
            if drawing_tracker is not None and config.drawing_enabled:
                # Update drawing tracker with hand landmarks
                drawing_tracker.update(hand_landmarks, handedness_results, 
                                     config.frame_width, config.frame_height)
                
                # Draw the trajectory on the frame
                drawing_tracker.draw_trajectory(processed, color=(0, 255, 0), thickness=3)
                
                # Periodic classification
                current_time = time.time()
                if (current_time - last_classification_time > classification_interval and 
                    drawing_classifier is not None):
                    
                    drawing = drawing_tracker.get_current_drawing()
                    if drawing.strokes and any(len(s.points) > 0 for s in drawing.strokes):
                        current_predictions = drawing_classifier.classify(drawing, top_k=3)
                        predictions_text = drawing_classifier.get_prediction_text(current_predictions)
                    else:
                        predictions_text = "Point index finger to start drawing"
                    
                    last_classification_time = current_time
                
                # Draw gesture status and drawing info
                draw_gesture_status(processed, drawing_tracker)

            # FPS calculation
            frame_count += 1
            now = time.time()
            if now - last_time >= 0.5:
                fps = frame_count / (now - last_time)
                frame_count = 0
                last_time = now

            # Overlays
            draw_help_overlay(processed, mode, config.hands_enabled, config.drawing_enabled)
            
            # FPS in bottom left (moved from top right)
            if config.show_fps:
                put_text(processed, f"FPS: {fps:.1f}", (10, processed.shape[0] - 10), scale=0.5, color=(255, 255, 0))
            
            # Prediction overlay in top right
            if config.show_predictions and config.drawing_enabled:
                draw_prediction_overlay(processed, predictions_text)

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
                    print(f"Hand landmarks: {'enabled' if config.hands_enabled else 'disabled'}")
            elif key == ord("d"):
                if drawing_tracker is not None:
                    config.drawing_enabled = not config.drawing_enabled
                    print(f"Drawing recognition: {'enabled' if config.drawing_enabled else 'disabled'}")
            elif key == ord("c"):
                if drawing_tracker is not None:
                    completed_drawing = drawing_tracker.clear_drawing()
                    if completed_drawing.strokes:
                        print(f"\\n🎨 Cleared drawing with {len(completed_drawing.strokes)} strokes")
                        # Show final predictions for the completed drawing
                        if drawing_classifier is not None:
                            final_predictions = drawing_classifier.classify(completed_drawing)
                            final_text = drawing_classifier.get_prediction_text(final_predictions)
                            print(f"🔮 Final prediction: {final_text}")
                            if len(final_predictions) > 1:
                                print("📊 All predictions:")
                                for name, conf in final_predictions:
                                    print(f"   {name}: {conf:.1%}")
                    predictions_text = "Point index finger to start drawing"
                    print()
            elif key == ord("s"):
                ts = time.strftime("%Y%m%d-%H%M%S")
                path = os.path.join(out_dir, f"frame-{ts}-{mode}.png")
                cv2.imwrite(path, processed)
                print(f"💾 Saved frame: {path}")
                
                # Also save drawing data if available
                if drawing_tracker is not None:
                    save_drawing_data(drawing_tracker, current_predictions, out_dir)

    finally:
        cap.release()
        if hands_ctx is not None:
            hands_ctx.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())