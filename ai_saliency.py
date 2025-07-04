import cv2
import numpy as np
import os
import sys

# Configure paths
current_dir = os.path.dirname(os.path.abspath(__file__))
deepgaze_path = os.path.join(current_dir, 'deepgaze')
sys.path.insert(0, deepgaze_path)

try:
    from deepgaze.saliency_map import FasaSaliencyMapping
    DEEPGAZE_AVAILABLE = True
    print("DeepGaze loaded successfully!")
except ImportError as e:
    print(f"DeepGaze import failed: {e}")
    DEEPGAZE_AVAILABLE = False
    from cv2.saliency import StaticSaliencyFineGrained_create

class SaliencyPredictor:
    def __init__(self, default_width=256, default_height=256):
        if DEEPGAZE_AVAILABLE:
            print("Using DeepGaze for saliency detection")
            self.model = FasaSaliencyMapping(image_w=default_width, image_h=default_height)
            self.predict = self._predict_deepgaze
        else:
            print("Falling back to OpenCV saliency")
            self.model = StaticSaliencyFineGrained_create()
            self.predict = self._predict_opencv
    
    def _predict_deepgaze(self, frame):
    
    
        resized = cv2.resize(frame, (256, 256))
        heatmap = self.model.returnMask(resized)  # This is likely float32 in [0, 1]

    # Normalize and convert to uint8 (0-255 grayscale image)
        heatmap = (heatmap * 255).astype(np.uint8)

        return heatmap


    def _predict_opencv(self, frame):
        """OpenCV fallback prediction"""
        success, heatmap = self.model.computeSaliency(frame)
        if not success:
            # Default to center if detection fails
            x, y = frame.shape[1]//2, frame.shape[0]//2
            return self._format_output(x, y, frame.shape, 1.0)
        
        max_idx = np.argmax(heatmap)
        y, x = np.unravel_index(max_idx, heatmap.shape)
        return self._format_output(x, y, frame.shape, heatmap.max())

    def _format_output(self, x, y, frame_shape, saliency_value):
        """Format output to match expected JSON structure"""
        face = self._get_face_name(x, y, frame_shape)
        return {
            "0": {
                "name": face,
                "row": str(y),
                "column": str(x),
                "width": "50",
                "saliency": str(saliency_value)
            }
        }

    def _get_face_name(self, x, y, frame_shape):
        """Map coordinates to cube face"""
        height, width = frame_shape[:2]
        if y < height // 2:  # Top row
            if x < width // 3: return "R"
            elif x < 2 * width // 3: return "L"
            else: return "U"
        else:  # Bottom row
            if x < width // 3: return "D"
            elif x < 2 * width // 3: return "F"
            else: return "B"