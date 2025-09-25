from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import cv2


@dataclass
class Point:
    """A point in the drawing trajectory."""
    x: float
    y: float
    timestamp: float
    pressure: float = 1.0  # Could be used for gesture strength


@dataclass
class Stroke:
    """A continuous drawing stroke (pen down to pen up)."""
    points: List[Point] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    
    def add_point(self, x: float, y: float, pressure: float = 1.0) -> None:
        """Add a point to this stroke."""
        timestamp = time.time()
        if not self.points:
            self.start_time = timestamp
        self.end_time = timestamp
        self.points.append(Point(x, y, timestamp, pressure))
    
    def duration(self) -> float:
        """Get the duration of this stroke in seconds."""
        return self.end_time - self.start_time if self.points else 0.0
    
    def length(self) -> float:
        """Calculate the total path length of this stroke."""
        if len(self.points) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(self.points)):
            p1, p2 = self.points[i-1], self.points[i]
            dx, dy = p2.x - p1.x, p2.y - p1.y
            total_length += np.sqrt(dx*dx + dy*dy)
        
        return total_length


@dataclass
class Drawing:
    """A complete drawing made up of multiple strokes."""
    strokes: List[Stroke] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    
    def add_stroke(self, stroke: Stroke) -> None:
        """Add a stroke to this drawing."""
        self.strokes.append(stroke)
    
    def total_duration(self) -> float:
        """Total time spent drawing."""
        if not self.strokes:
            return 0.0
        return max(s.end_time for s in self.strokes) - min(s.start_time for s in self.strokes)
    
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Get the bounding box (min_x, min_y, max_x, max_y) of all strokes."""
        if not self.strokes or not any(s.points for s in self.strokes):
            return (0.0, 0.0, 0.0, 0.0)
        
        all_points = [point for stroke in self.strokes for point in stroke.points]
        min_x = min(p.x for p in all_points)
        max_x = max(p.x for p in all_points)
        min_y = min(p.y for p in all_points)
        max_y = max(p.y for p in all_points)
        
        return (min_x, min_y, max_x, max_y)
    
    def normalize(self, target_size: int = 256) -> Drawing:
        """Normalize the drawing to fit in a target_size x target_size box."""
        min_x, min_y, max_x, max_y = self.bounding_box()
        width, height = max_x - min_x, max_y - min_y
        
        if width == 0 and height == 0:
            return self
        
        # Scale to fit target size while preserving aspect ratio
        scale = target_size / max(width, height) if max(width, height) > 0 else 1.0
        
        normalized = Drawing()
        for stroke in self.strokes:
            new_stroke = Stroke()
            for point in stroke.points:
                new_x = (point.x - min_x) * scale
                new_y = (point.y - min_y) * scale
                new_stroke.points.append(Point(new_x, new_y, point.timestamp, point.pressure))
            if new_stroke.points:
                new_stroke.start_time = stroke.start_time
                new_stroke.end_time = stroke.end_time
                normalized.add_stroke(new_stroke)
        
        return normalized


class DrawingTracker:
    """Tracks hand movements and converts them into drawings."""
    
    def __init__(self):
        self.current_stroke: Optional[Stroke] = None
        self.current_drawing = Drawing()
        self.is_drawing = False
        self.last_drawing_point: Optional[Tuple[float, float]] = None
        
        # Drawing detection parameters
        self.drawing_gesture_threshold = 0.8  # Confidence for "drawing" gesture
        self.min_movement_threshold = 5.0     # Minimum pixel movement to register
        self.max_gap_time = 0.5              # Max time gap before ending a stroke (seconds)
        self.last_point_time = 0.0
        
    def detect_drawing_gesture(self, hand_landmarks, handedness_results) -> bool:
        """
        Detect if the user is making a drawing gesture.
        
        Drawing gesture = index finger extended, other fingers curled
        This is like pointing with your index finger.
        """
        if not hand_landmarks:
            return False
            
        # MediaPipe hand landmark indices:
        # Index finger: 5 (MCP), 6 (PIP), 7 (DIP), 8 (TIP)
        # Middle finger: 9 (MCP), 10 (PIP), 11 (DIP), 12 (TIP)
        # Ring finger: 13 (MCP), 14 (PIP), 15 (DIP), 16 (TIP)
        # Pinky: 17 (MCP), 18 (PIP), 19 (DIP), 20 (TIP)
        # Thumb: 1 (CMC), 2 (MCP), 3 (IP), 4 (TIP)
        
        landmarks = hand_landmarks.landmark
        
        # Check if index finger is extended (tip higher than pip joint)
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        index_extended = index_tip.y < index_pip.y  # In image coords, y decreases upward
        
        if not index_extended:
            return False
        
        # Check if other fingers are curled (tips lower than pip joints)
        # Middle finger
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        middle_curled = middle_tip.y > middle_pip.y
        
        # Ring finger
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        ring_curled = ring_tip.y > ring_pip.y
        
        # Pinky
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        pinky_curled = pinky_tip.y > pinky_pip.y
        
        # Drawing gesture = index extended + at least 2 other fingers curled
        fingers_curled = sum([middle_curled, ring_curled, pinky_curled])
        
        return index_extended and fingers_curled >= 2
    
    def update(self, hand_landmarks, handedness_results, frame_width: int, frame_height: int) -> None:
        """
        Update the tracker with new hand landmark data.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            handedness_results: Left/right hand classification
            frame_width: Width of the camera frame
            frame_height: Height of the camera frame
        """
        current_time = time.time()
        
        if not hand_landmarks:
            self._end_current_stroke()
            return
        
        # Use the first detected hand for now
        landmarks = hand_landmarks[0]
        
        # Get index finger tip position (landmark 8)
        index_tip = landmarks.landmark[8]
        finger_x = index_tip.x * frame_width
        finger_y = index_tip.y * frame_height
        
        is_drawing_gesture = self.detect_drawing_gesture(landmarks, handedness_results)
        
        if is_drawing_gesture:
            # Check if we should add this point
            if self.last_drawing_point is not None:
                last_x, last_y = self.last_drawing_point
                distance = np.sqrt((finger_x - last_x)**2 + (finger_y - last_y)**2)
                
                if distance < self.min_movement_threshold:
                    return  # Too small movement, ignore
            
            # Check for stroke continuation vs new stroke
            if (self.current_stroke is None or 
                current_time - self.last_point_time > self.max_gap_time):
                self._start_new_stroke()
            
            # Add point to current stroke
            if self.current_stroke is not None:
                self.current_stroke.add_point(finger_x, finger_y)
                self.last_drawing_point = (finger_x, finger_y)
                self.last_point_time = current_time
                self.is_drawing = True
        else:
            self._end_current_stroke()
    
    def _start_new_stroke(self) -> None:
        """Start a new stroke."""
        self._end_current_stroke()  # End previous stroke if exists
        self.current_stroke = Stroke()
    
    def _end_current_stroke(self) -> None:
        """End the current stroke and add it to the drawing."""
        if self.current_stroke is not None and len(self.current_stroke.points) > 1:
            self.current_drawing.add_stroke(self.current_stroke)
        
        self.current_stroke = None
        self.is_drawing = False
        self.last_drawing_point = None
    
    def get_current_drawing(self) -> Drawing:
        """Get the current drawing being created."""
        return self.current_drawing
    
    def clear_drawing(self) -> Drawing:
        """Clear the current drawing and return the completed one."""
        self._end_current_stroke()
        completed_drawing = self.current_drawing
        self.current_drawing = Drawing()
        return completed_drawing
    
    def draw_trajectory(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> None:
        """Draw the current trajectory on the frame."""
        # Draw completed strokes
        for stroke in self.current_drawing.strokes:
            if len(stroke.points) < 2:
                continue
            
            for i in range(1, len(stroke.points)):
                p1, p2 = stroke.points[i-1], stroke.points[i]
                cv2.line(frame, 
                        (int(p1.x), int(p1.y)), 
                        (int(p2.x), int(p2.y)), 
                        color, thickness)
        
        # Draw current stroke being created
        if self.current_stroke and len(self.current_stroke.points) >= 2:
            for i in range(1, len(self.current_stroke.points)):
                p1, p2 = self.current_stroke.points[i-1], self.current_stroke.points[i]
                cv2.line(frame, 
                        (int(p1.x), int(p1.y)), 
                        (int(p2.x), int(p2.y)), 
                        (255, 255, 0), thickness)  # Yellow for current stroke