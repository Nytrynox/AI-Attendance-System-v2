# src/face_detector.py

import dlib
import cv2
import numpy as np
import os

class FaceDetector:
    def __init__(self, predictor_path='models/shape_predictor_68_face_landmarks.dat'):
        # Primary HOG face detector (fast and reliable)
        self.detector = dlib.get_frontal_face_detector() if hasattr(dlib, 'get_frontal_face_detector') else None  # type: ignore
        
        # Backup OpenCV cascade classifier
        try:
            self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # type: ignore
            self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        except AttributeError:
            # Fallback if cv2.data is not available
            self.cascade_path = 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        
        # Landmark predictor (optional)
        self.predictor = None
        if os.path.exists(predictor_path):
            try:
                self.predictor = dlib.shape_predictor(predictor_path) if hasattr(dlib, 'shape_predictor') else None  # type: ignore
                print(f"[INFO] Landmark predictor loaded: {predictor_path}")
            except Exception as e:
                print(f"[WARNING] Failed to load predictor: {e}")
        else:
            print(f"[WARNING] Landmark predictor not found: {predictor_path}")

    def detect_faces(self, image):
        """
        Detect faces in the image using multiple methods for robustness.
        Returns a list of tuples: (x, y, w, h) or (x, y, w, h, face_crop, landmarks)
        """
        if image is None or image.size == 0:
            return []
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = []

        # Method 1: Try dlib HOG detector first (most accurate)
        try:
            if self.detector is not None:
                faces = self.detector(gray)
                for rect in faces:
                    x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
                    # Ensure coordinates are within image bounds
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, image.shape[1] - x)
                    h = min(h, image.shape[0] - y)
                    
                    if w > 20 and h > 20:  # Minimum face size
                        face_crop = image[y:y+h, x:x+w]
                        
                        # Get landmarks if predictor is available
                        landmarks = None
                        if self.predictor:
                            try:
                                shape = self.predictor(gray, rect)
                                landmarks = [(pt.x, pt.y) for pt in shape.parts()]
                            except:
                                landmarks = None
                        
                        if landmarks:
                            results.append((x, y, w, h, face_crop, landmarks))
                        else:
                            results.append((x, y, w, h))
                        
        except Exception as e:
            print(f"[WARNING] dlib detector failed: {e}")

        # Method 2: If no faces found, try OpenCV cascade classifier
        if not results:
            try:
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                for (x, y, w, h) in faces:
                    # Ensure coordinates are within image bounds
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, image.shape[1] - x)
                    h = min(h, image.shape[0] - y)
                    
                    if w > 20 and h > 20:  # Minimum face size
                        results.append((x, y, w, h))
                        
            except Exception as e:
                print(f"[WARNING] OpenCV cascade detector failed: {e}")
        
        # Method 3: Final fallback - try with different parameters
        if not results:
            try:
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.05, 
                    minNeighbors=3, 
                    minSize=(20, 20),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                for (x, y, w, h) in faces:
                    # Ensure coordinates are within image bounds
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, image.shape[1] - x)
                    h = min(h, image.shape[0] - y)
                    
                    if w > 15 and h > 15:  # Even more lenient minimum face size
                        results.append((x, y, w, h))
                        
            except Exception as e:
                print(f"[WARNING] Fallback cascade detector failed: {e}")

        return results
