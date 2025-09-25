from __future__ import annotations

import math
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
try:
    from .drawing_tracker import Drawing, Stroke, Point
except ImportError:
    from drawing_tracker import Drawing, Stroke, Point


class SimpleShapeClassifier:
    """
    A simple rule-based classifier for basic shapes and patterns.
    
    This is a starting point - in the future, this could be replaced with:
    - A CNN trained on Quick, Draw! dataset
    - A more sophisticated pattern matching system
    - An online learning system that adapts to user's drawing style
    """
    
    def __init__(self):
        self.shape_templates = self._initialize_shape_templates()
        self.min_confidence = 0.3
        
    def _initialize_shape_templates(self) -> Dict[str, callable]:
        """Initialize shape detection functions."""
        return {
            "circle": self._detect_circle,
            "square": self._detect_square,
            "rectangle": self._detect_rectangle,
            "triangle": self._detect_triangle,
            "line": self._detect_line,
            "house": self._detect_house,
            "star": self._detect_star,
            "heart": self._detect_heart,
            "letter_O": self._detect_letter_o,
            "letter_I": self._detect_letter_i,
            "letter_L": self._detect_letter_l,
            "tree": self._detect_tree,
        }
    
    def classify_drawing(self, drawing: Drawing, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Classify a drawing and return top predictions with confidence scores.
        
        Args:
            drawing: The drawing to classify
            top_k: Number of top predictions to return
            
        Returns:
            List of (class_name, confidence) tuples, sorted by confidence
        """
        if not drawing.strokes or not any(s.points for s in drawing.strokes):
            return [("empty", 0.0)]
        
        # Normalize the drawing for consistent analysis
        normalized = drawing.normalize(256)
        
        predictions = []
        
        # Test each shape template
        for shape_name, detector_func in self.shape_templates.items():
            try:
                confidence = detector_func(normalized)
                if confidence >= self.min_confidence:
                    predictions.append((shape_name, confidence))
            except Exception as e:
                # Skip failed detectors
                print(f"Warning: Shape detector '{shape_name}' failed: {e}")
                continue
        
        # Sort by confidence and return top k
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k] if predictions else [("unknown", 0.1)]
    
    def _get_drawing_features(self, drawing: Drawing) -> Dict[str, float]:
        """Extract basic features from a drawing."""
        if not drawing.strokes:
            return {}
        
        min_x, min_y, max_x, max_y = drawing.bounding_box()
        width = max_x - min_x
        height = max_y - min_y
        aspect_ratio = width / height if height > 0 else 1.0
        
        total_points = sum(len(s.points) for s in drawing.strokes)
        total_length = sum(s.length() for s in drawing.strokes)
        
        # Compactness (how much the drawing fills its bounding box)
        bounding_area = width * height
        compactness = total_length / bounding_area if bounding_area > 0 else 0.0
        
        return {
            "num_strokes": len(drawing.strokes),
            "total_points": total_points,
            "total_length": total_length,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "compactness": compactness,
            "bounding_area": bounding_area,
        }
    
    def _detect_circle(self, drawing: Drawing) -> float:
        """Detect if the drawing is a circle."""
        features = self._get_drawing_features(drawing)
        
        # Single stroke is better for circles
        if features["num_strokes"] > 2:
            return 0.0
        
        # Should have reasonable aspect ratio (close to square)
        if not (0.7 < features["aspect_ratio"] < 1.43):  # ~1/sqrt(2) to sqrt(2)
            return 0.0
        
        # Check if the stroke forms a closed loop
        if drawing.strokes:
            stroke = drawing.strokes[0]
            if len(stroke.points) < 10:
                return 0.0
            
            # Check closure (start and end points should be close)
            start_point = stroke.points[0]
            end_point = stroke.points[-1]
            closure_distance = math.sqrt((start_point.x - end_point.x)**2 + (start_point.y - end_point.y)**2)
            max_dim = max(features["width"], features["height"])
            closure_ratio = closure_distance / max_dim if max_dim > 0 else 1.0
            
            closure_score = max(0, 1.0 - closure_ratio * 3)  # Good if < 1/3 of size
            
            # Check circularity by measuring deviation from center
            center_x = (drawing.bounding_box()[0] + drawing.bounding_box()[2]) / 2
            center_y = (drawing.bounding_box()[1] + drawing.bounding_box()[3]) / 2
            expected_radius = max_dim / 2
            
            radius_deviations = []
            for point in stroke.points[::3]:  # Sample every 3rd point for performance
                actual_radius = math.sqrt((point.x - center_x)**2 + (point.y - center_y)**2)
                deviation = abs(actual_radius - expected_radius) / expected_radius
                radius_deviations.append(deviation)
            
            if radius_deviations:
                avg_deviation = sum(radius_deviations) / len(radius_deviations)
                circularity_score = max(0, 1.0 - avg_deviation * 2)
            else:
                circularity_score = 0.0
            
            return min(closure_score * circularity_score * 1.2, 1.0)
        
        return 0.0
    
    def _detect_square(self, drawing: Drawing) -> float:
        """Detect if the drawing is a square."""
        features = self._get_drawing_features(drawing)
        
        # Should be roughly square aspect ratio
        if not (0.8 < features["aspect_ratio"] < 1.25):
            return 0.0
        
        # Could be drawn in 1 stroke (closed) or 4 strokes (sides)
        if features["num_strokes"] > 4:
            return 0.0
        
        # Check for roughly 4 corners (direction changes)
        if drawing.strokes and len(drawing.strokes) == 1:
            stroke = drawing.strokes[0]
            corners = self._detect_corners(stroke)
            corner_score = min(len(corners) / 4.0, 1.0) if corners else 0.0
            return corner_score * 0.8
        
        return 0.5 if features["num_strokes"] == 4 else 0.2
    
    def _detect_rectangle(self, drawing: Drawing) -> float:
        """Detect if the drawing is a rectangle."""
        features = self._get_drawing_features(drawing)
        
        # Rectangle should NOT be square
        if 0.9 < features["aspect_ratio"] < 1.1:
            return 0.0  # Too square
        
        # Could be drawn in 1 stroke or 4 strokes
        if features["num_strokes"] > 4:
            return 0.0
        
        return 0.6 if features["num_strokes"] <= 2 else 0.4
    
    def _detect_triangle(self, drawing: Drawing) -> float:
        """Detect if the drawing is a triangle."""
        features = self._get_drawing_features(drawing)
        
        if features["num_strokes"] > 3:
            return 0.0
        
        # Look for 3 corners if single stroke
        if len(drawing.strokes) == 1:
            corners = self._detect_corners(drawing.strokes[0])
            if corners and len(corners) >= 3:
                return min(3.0 / len(corners), 1.0) * 0.8
        
        return 0.4 if features["num_strokes"] == 3 else 0.2
    
    def _detect_line(self, drawing: Drawing) -> float:
        """Detect if the drawing is a straight line."""
        features = self._get_drawing_features(drawing)
        
        if features["num_strokes"] != 1:
            return 0.0
        
        # Should have extreme aspect ratio (very long and thin)
        if features["aspect_ratio"] < 3.0 and features["aspect_ratio"] > 0.33:
            return 0.0
        
        stroke = drawing.strokes[0]
        if len(stroke.points) < 3:
            return 0.0
        
        # Check straightness
        start = stroke.points[0]
        end = stroke.points[-1]
        
        # Calculate how much the line deviates from straight
        line_length = math.sqrt((end.x - start.x)**2 + (end.y - start.y)**2)
        if line_length == 0:
            return 0.0
        
        max_deviation = 0
        for point in stroke.points[1:-1]:
            # Distance from point to line
            deviation = self._point_to_line_distance(point, start, end)
            max_deviation = max(max_deviation, deviation)
        
        straightness = max(0, 1.0 - (max_deviation / line_length) * 10)
        return straightness
    
    def _detect_house(self, drawing: Drawing) -> float:
        """Detect a simple house shape (square with triangle on top)."""
        features = self._get_drawing_features(drawing)
        
        # House typically needs multiple strokes
        if features["num_strokes"] < 2:
            return 0.0
        
        # Should be wider than tall or roughly square
        if features["aspect_ratio"] < 0.6:
            return 0.0
        
        # Simple heuristic: if we have a reasonable number of strokes and aspect ratio
        if 2 <= features["num_strokes"] <= 6 and 0.8 <= features["aspect_ratio"] <= 1.5:
            return 0.6
        
        return 0.0
    
    def _detect_star(self, drawing: Drawing) -> float:
        """Detect a star shape."""
        features = self._get_drawing_features(drawing)
        
        # Star should be roughly square
        if not (0.7 < features["aspect_ratio"] < 1.43):
            return 0.0
        
        # Stars often have many direction changes
        if len(drawing.strokes) == 1:
            corners = self._detect_corners(drawing.strokes[0])
            if corners and len(corners) >= 5:  # Stars have many points
                return min(0.8, len(corners) / 10.0)
        
        return 0.0
    
    def _detect_heart(self, drawing: Drawing) -> float:
        """Detect a heart shape."""
        features = self._get_drawing_features(drawing)
        
        # Heart should be taller than wide
        if features["aspect_ratio"] > 1.2:
            return 0.0
        
        # Usually drawn in 1-2 strokes
        if features["num_strokes"] > 3:
            return 0.0
        
        # Simple heuristic based on aspect ratio
        if 0.7 <= features["aspect_ratio"] <= 1.1:
            return 0.5
        
        return 0.0
    
    def _detect_letter_o(self, drawing: Drawing) -> float:
        """Detect letter O (similar to circle but might be more oval)."""
        circle_confidence = self._detect_circle(drawing)
        if circle_confidence > 0.3:
            return circle_confidence * 0.9  # Slightly lower than circle
        return 0.0
    
    def _detect_letter_i(self, drawing: Drawing) -> float:
        """Detect letter I."""
        features = self._get_drawing_features(drawing)
        
        # I should be much taller than wide
        if features["aspect_ratio"] > 0.5:
            return 0.0
        
        # Usually 1-3 strokes (vertical line + optional serifs/dot)
        if 1 <= features["num_strokes"] <= 3:
            return 0.7
        
        return 0.0
    
    def _detect_letter_l(self, drawing: Drawing) -> float:
        """Detect letter L."""
        features = self._get_drawing_features(drawing)
        
        # L should be taller than wide
        if features["aspect_ratio"] > 0.8:
            return 0.0
        
        # Usually 1-2 strokes
        if 1 <= features["num_strokes"] <= 2:
            return 0.6
        
        return 0.0
    
    def _detect_tree(self, drawing: Drawing) -> float:
        """Detect a simple tree shape."""
        features = self._get_drawing_features(drawing)
        
        # Tree should be taller than wide
        if features["aspect_ratio"] > 0.8:
            return 0.0
        
        # Trees usually have multiple parts (trunk + foliage)
        if features["num_strokes"] < 2:
            return 0.0
        
        if 2 <= features["num_strokes"] <= 5:
            return 0.5
        
        return 0.0
    
    def _detect_corners(self, stroke: Stroke, angle_threshold: float = 45.0) -> List[Point]:
        """Detect corner points in a stroke based on direction changes."""
        if len(stroke.points) < 3:
            return []
        
        corners = []
        angle_threshold_rad = math.radians(angle_threshold)
        
        for i in range(1, len(stroke.points) - 1):
            p1 = stroke.points[i-1]
            p2 = stroke.points[i]
            p3 = stroke.points[i+1]
            
            # Calculate vectors
            v1 = (p2.x - p1.x, p2.y - p1.y)
            v2 = (p3.x - p2.x, p3.y - p2.y)
            
            # Calculate angle between vectors
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to [-1, 1]
                angle = math.acos(abs(cos_angle))
                
                if angle > angle_threshold_rad:
                    corners.append(p2)
        
        return corners
    
    def _point_to_line_distance(self, point: Point, line_start: Point, line_end: Point) -> float:
        """Calculate the perpendicular distance from a point to a line segment."""
        # Vector from line_start to line_end
        line_vec = (line_end.x - line_start.x, line_end.y - line_start.y)
        
        # Vector from line_start to point
        point_vec = (point.x - line_start.x, point.y - line_start.y)
        
        line_len_sq = line_vec[0]**2 + line_vec[1]**2
        
        if line_len_sq == 0:
            # Line is actually a point
            return math.sqrt((point.x - line_start.x)**2 + (point.y - line_start.y)**2)
        
        # Project point onto line
        t = max(0, min(1, (point_vec[0] * line_vec[0] + point_vec[1] * line_vec[1]) / line_len_sq))
        
        # Find the closest point on the line
        closest_x = line_start.x + t * line_vec[0]
        closest_y = line_start.y + t * line_vec[1]
        
        # Return distance
        return math.sqrt((point.x - closest_x)**2 + (point.y - closest_y)**2)


class DrawingClassifier:
    """Main classifier that can be extended with different classification approaches."""
    
    def __init__(self):
        self.shape_classifier = SimpleShapeClassifier()
        
        # In the future, we could add:
        # self.neural_classifier = NeuralDrawingClassifier()
        # self.quickdraw_classifier = QuickDrawClassifier()
    
    def classify(self, drawing: Drawing, top_k: int = 3) -> List[Tuple[str, float]]:
        """Classify a drawing using available classifiers."""
        # For now, just use the simple shape classifier
        results = self.shape_classifier.classify_drawing(drawing, top_k)
        
        # In the future, we could ensemble multiple classifiers:
        # shape_results = self.shape_classifier.classify_drawing(drawing, top_k)
        # neural_results = self.neural_classifier.classify_drawing(drawing, top_k)
        # combined_results = self._ensemble_results([shape_results, neural_results])
        
        return results
    
    def get_prediction_text(self, predictions: List[Tuple[str, float]]) -> str:
        """Format predictions into a readable string."""
        if not predictions:
            return "No prediction"
        
        best_pred, confidence = predictions[0]
        
        if confidence < 0.3:
            return "Unknown drawing"
        elif confidence < 0.5:
            return f"Maybe: {best_pred.replace('_', ' ').title()}"
        else:
            return f"{best_pred.replace('_', ' ').title()} ({confidence:.1%})"
