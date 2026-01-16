# src/utils/image_utils.py

import cv2
# Keep numpy import as it's used in function signature type hints
import numpy as np  # noqa: F401 - Used for type hints in docstrings
import face_recognition

def resize_image(image, width=600):
    """
    Resize image while maintaining aspect ratio.
    Args:
        image (numpy.ndarray): Input image.
        width (int): The new width of the image.
    Returns:
        numpy.ndarray: Resized image.
    """
    aspect_ratio = image.shape[1] / float(image.shape[0])
    height = int(width / aspect_ratio)
    resized_image = cv2.resize(image, (width, height))
    return resized_image

def convert_to_rgb(image):
    """
    Convert a BGR image to RGB.
    Args:
        image (numpy.ndarray): Input BGR image.
    Returns:
        numpy.ndarray: Converted RGB image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def preprocess_for_face_recognition(image):
    """
    Preprocess image for face recognition model input with improved robustness.
    Args:
        image (numpy.ndarray): Input image.
    Returns:
        list: List of face encodings.
    """
    try:
        # Validate input
        if image is None or image.size == 0:
            return []
            
        # Ensure minimum size
        height, width = image.shape[:2]
        if height < 20 or width < 20:
            return []
        
        # Resize if image is too small for good face recognition
        if height < 80 or width < 80:
            scale_factor = max(80 / height, 80 / width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Convert to RGB
        rgb_image = convert_to_rgb(image)
        
        # Try multiple face detection models for better coverage
        face_encodings = []
        
        # Method 1: Try with HOG model (faster, good for frontal faces)
        try:
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                if face_encodings:
                    return face_encodings
        except Exception:
            pass
        
        # Method 2: Try with CNN model (more accurate but slower)
        try:
            face_locations = face_recognition.face_locations(rgb_image, model="cnn")
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                if face_encodings:
                    return face_encodings
        except Exception:
            pass
        
        # Method 3: Fallback - use whole image with different parameters
        try:
            # Use the whole image as face location
            height, width = rgb_image.shape[:2]
            whole_image_location = [(0, width, height, 0)]  # (top, right, bottom, left)
            
            face_encodings = face_recognition.face_encodings(rgb_image, whole_image_location)
            if face_encodings:
                return face_encodings
        except Exception:
            pass
        
        # Method 4: Try with enhanced image
        try:
            # Apply simple enhancement
            enhanced = cv2.convertScaleAbs(rgb_image, alpha=1.2, beta=10)  # Increase contrast and brightness
            
            face_locations = face_recognition.face_locations(enhanced, model="hog")
            if face_locations:
                face_encodings = face_recognition.face_encodings(enhanced, face_locations)
                if face_encodings:
                    return face_encodings
        except Exception:
            pass
        
        # Method 5: Try with histogram equalization
        try:
            # Convert to grayscale for equalization
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            equalized = cv2.equalizeHist(gray)
            # Convert back to RGB
            enhanced_rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
            
            face_locations = face_recognition.face_locations(enhanced_rgb, model="hog")
            if face_locations:
                face_encodings = face_recognition.face_encodings(enhanced_rgb, face_locations)
                if face_encodings:
                    return face_encodings
                    
            # Try with whole image as well
            height, width = enhanced_rgb.shape[:2]
            whole_image_location = [(0, width, height, 0)]
            face_encodings = face_recognition.face_encodings(enhanced_rgb, whole_image_location)
            if face_encodings:
                return face_encodings
        except Exception:
            pass
        
        # If all methods fail, return empty list
        return []
        
    except Exception:
        # If anything goes wrong, return empty list instead of crashing
        return []

def load_image(image_path):
    """
    Load image from file path and return it as numpy array.
    Args:
        image_path (str): Path to image file.
    Returns:
        numpy.ndarray: Loaded image.
    """
    image = cv2.imread(image_path)
    return image

def show_image(image, window_name="Image"):
    """
    Show image using OpenCV.
    Args:
        image (numpy.ndarray): Input image.
        window_name (str): Window name.
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(image, file_path):
    """
    Save image to the specified file path.
    Args:
        image (numpy.ndarray): Image to save.
        file_path (str): File path to save image.
    """
    cv2.imwrite(file_path, image)

def crop_face(image, face_location):
    """
    Crop a detected face from the image using its coordinates.
    Args:
        image (numpy.ndarray): Input image.
        face_location (tuple): The (top, right, bottom, left) coordinates of the face.
    Returns:
        numpy.ndarray: Cropped face image.
    """
    top, right, bottom, left = face_location
    cropped_face = image[top:bottom, left:right]
    return cropped_face

def setup_camera(camera_index=0):
    """
    Setup and initialize camera for video capture.
    Args:
        camera_index (int): Camera index (default: 0 for default camera).
    Returns:
        cv2.VideoCapture: Initialized camera object.
    """
    camera = cv2.VideoCapture(camera_index)
    
    if not camera.isOpened():
        raise ValueError(f"Could not open camera with index {camera_index}")
    
    # Set camera properties for better performance
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    return camera
