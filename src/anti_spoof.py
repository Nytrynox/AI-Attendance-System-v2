# src/anti_spoof.py

import numpy as np
import cv2
try:
    from tensorflow.keras.models import load_model  # type: ignore
    TENSORFLOW_AVAILABLE = True
    HAS_LOAD_MODEL = True
except ImportError:
    try:
        from keras.models import load_model  # type: ignore
        TENSORFLOW_AVAILABLE = True
        HAS_LOAD_MODEL = True
    except ImportError:
        TENSORFLOW_AVAILABLE = False
        HAS_LOAD_MODEL = False
        load_model = None  # type: ignore
        print("Warning: TensorFlow/Keras not available. Using basic anti-spoofing.")
import os

class AntiSpoofingDetector:
    def __init__(self, model_path='models/anti_spoof_model.h5'):
        self.model_path = model_path
        self.model = None
        self.input_size = (160, 160)  # Assumed input size for anti-spoof model
        
        if TENSORFLOW_AVAILABLE and HAS_LOAD_MODEL and load_model and os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                print(f"Anti-spoofing model loaded from {model_path}")
            except Exception as e:
                print(f"Failed to load anti-spoofing model: {e}")
                self.model = None
        else:
            print("Using basic anti-spoofing detection (no ML model)")

    def preprocess(self, face_img):
        """
        Preprocess the face image for anti-spoofing prediction.
        Resize, normalize and expand dims.
        """
        resized_face = cv2.resize(face_img, self.input_size)
        normalized_face = resized_face / 255.0  # Normalize to [0, 1]
        input_tensor = np.expand_dims(normalized_face, axis=0)
        return input_tensor

    def predict(self, face_img, debug=False, threshold=None):
        """
        Predict whether the given face image is real or spoofed.
        Returns: True (Real), False (Spoof)
        Args:
            face_img: Input face image
            debug: Whether to print debug information
            threshold: Custom threshold for spoof detection (default: 0.7 for aggressive, 0.3-0.4 for lenient)
        """
        if self.model is not None:
            processed_img = self.preprocess(face_img)
            prediction = self.model.predict(processed_img)[0][0]  # type: ignore
            if debug:
                print(f"[AntiSpoof] Prediction value: {prediction}")
            
            # Use provided threshold or default aggressive threshold
            spoof_threshold = threshold if threshold is not None else 0.7
            return prediction >= spoof_threshold, prediction  # True = Real, False = Fake
        else:
            # Basic anti-spoofing without ML model
            return self._basic_anti_spoof_check(face_img, debug), 0.8  # Assume real for basic check

    def _basic_anti_spoof_check(self, face_img, debug=False):
        """
        Basic anti-spoofing check using computer vision techniques
        """
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Check image quality metrics
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Very blurry images might indicate a photo/screen
            if laplacian_var < 50:
                if debug:
                    print(f"[BasicAntiSpoof] Image too blurry: {laplacian_var}")
                return False
                
            # Check for basic patterns that might indicate a screen
            # This is a very basic check - real ML models would be much more sophisticated
            
            if debug:
                print(f"[BasicAntiSpoof] Sharpness score: {laplacian_var}")
                
            return True  # Default to assuming real face
            
        except Exception as e:
            if debug:
                print(f"[BasicAntiSpoof] Error: {e}")
            return True  # Default to assuming real face

    def check_if_real(self, face_img, debug=False, threshold=None):
        """Check if face is real using specified threshold with phone detection"""
        # Basic anti-spoofing check
        basic_check, prediction = self.predict(face_img, debug=debug, threshold=threshold)
        
        # Add phone/screen detection for extra security
        if basic_check:  # Only check for phone if basic check passes
            phone_detected = self.detect_phone_screen(face_img, debug=debug)
            if phone_detected:
                if debug:
                    print("[ANTI-SPOOF] Phone/screen detected - overriding basic check")
                return False
        
        return basic_check
    
    def is_real_face(self, face_img, debug=False):
        """Main method used by the application - check if face is real"""
        return self.check_if_real(face_img, debug=debug, threshold=0.5)
    
    def check_if_real_lenient(self, face_img, debug=False):
        """Check if face is real using lenient threshold for registration"""
        return self.check_if_real(face_img, debug=debug, threshold=0.3)
    
    def check_if_real_aggressive(self, face_img, debug=False):
        """Check if face is real using aggressive threshold for attendance"""
        # First do basic anti-spoofing check
        basic_check = self.check_if_real(face_img, debug=debug, threshold=0.7)
        
        # Add phone/screen detection
        phone_detected = self.detect_phone_screen(face_img, debug=debug)
        
        if phone_detected:
            if debug:
                print("[ANTI-SPOOF] Phone/screen detected - marking as spoof")
            return False
        
        return basic_check
    
    def detect_phone_screen(self, face_img, debug=False):
        """
        Detect if the face image shows a phone screen or digital display
        Returns: True if phone/screen detected, False otherwise
        """
        try:
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
            
            # Method 1: Check for rectangular screen edges
            screen_edges = self._detect_rectangular_edges(gray, debug)
            
            # Method 2: Check for digital display characteristics
            digital_display = self._detect_digital_display_patterns(face_img, debug)
            
            # Method 3: Check for screen reflection patterns
            reflections = self._detect_screen_reflections(face_img, debug)
            
            # Method 4: Check for pixel grid patterns (common in screens)
            pixel_grid = self._detect_pixel_grid(gray, debug)
            
            # Combine all detection methods
            detection_score = 0
            if screen_edges: detection_score += 2
            if digital_display: detection_score += 2  
            if reflections: detection_score += 1
            if pixel_grid: detection_score += 1
            
            phone_detected = detection_score >= 3  # Require strong evidence
            
            if debug:
                print(f"[PHONE DETECTION] Edges:{screen_edges}, Display:{digital_display}, Reflections:{reflections}, Grid:{pixel_grid}, Score:{detection_score}")
            
            return phone_detected
            
        except Exception as e:
            if debug:
                print(f"[PHONE DETECTION] Error: {e}")
            return False
    
    def _detect_rectangular_edges(self, gray, debug=False):
        """Detect strong rectangular edges that might indicate a phone screen"""
        try:
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for rectangular contours
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's a rectangle (4 corners) and reasonably large
                if len(approx) == 4:
                    area = cv2.contourArea(contour)
                    image_area = gray.shape[0] * gray.shape[1]
                    
                    # If rectangle covers significant portion of image, might be a screen
                    if area > image_area * 0.3:  # 30% of image
                        return True
            
            return False
            
        except Exception as e:
            if debug:
                print(f"[EDGE DETECTION] Error: {e}")
            return False
    
    def _detect_digital_display_patterns(self, img, debug=False):
        """Detect patterns common in digital displays"""
        try:
            # Check for overly uniform color distributions (common in digital displays)
            for channel in range(3):  # B, G, R channels
                hist = cv2.calcHist([img], [channel], None, [256], [0, 256])
                
                # Check for artificial color peaks (common in displays)
                peaks = np.where(hist > np.mean(hist) * 3)[0]
                if len(peaks) > 10:  # Too many artificial peaks
                    return True
            
            # Check for unnatural saturation levels
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            saturation = hsv[:,:,1]
            
            # Digital displays often have unnatural saturation patterns
            sat_std = np.std(saturation)
            if sat_std < 20:  # Too uniform saturation
                return True
                
            return False
            
        except Exception as e:
            if debug:
                print(f"[DISPLAY PATTERN] Error: {e}")
            return False
    
    def _detect_screen_reflections(self, img, debug=False):
        """Detect reflections common on phone/computer screens"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Look for bright spots that could be reflections
            bright_threshold = np.mean(gray) + 2 * np.std(gray)
            bright_spots = np.where(gray > bright_threshold)
            
            # Check if bright spots form lines (common in screen reflections)
            if len(bright_spots[0]) > 0:
                # Simple heuristic: if many bright pixels, might be screen glare
                bright_ratio = len(bright_spots[0]) / (gray.shape[0] * gray.shape[1])
                if bright_ratio > 0.15:  # More than 15% bright pixels
                    return True
            
            return False
            
        except Exception as e:
            if debug:
                print(f"[REFLECTION DETECTION] Error: {e}")
            return False
    
    def _detect_pixel_grid(self, gray, debug=False):
        """Detect pixel grid patterns common in digital displays"""
        try:
            # Apply FFT to detect regular patterns
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Look for regular patterns in frequency domain
            # High frequency regular patterns suggest pixel grids
            high_freq_energy = np.sum(magnitude_spectrum[magnitude_spectrum.shape[0]//4:3*magnitude_spectrum.shape[0]//4, 
                                                        magnitude_spectrum.shape[1]//4:3*magnitude_spectrum.shape[1]//4])
            
            total_energy = np.sum(magnitude_spectrum)
            
            # If too much high frequency energy, might be pixel grid
            if total_energy > 0 and (high_freq_energy / total_energy) > 0.3:
                return True
                
            return False
            
        except Exception as e:
            if debug:
                print(f"[PIXEL GRID] Error: {e}")
            return False
    
    def get_model_info(self):
        """Return information about the anti-spoofing model."""
        return {
            'model_path': self.model_path,
            'input_size': self.input_size,
            'model_loaded': self.model is not None
        }
