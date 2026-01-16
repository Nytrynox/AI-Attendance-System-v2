# src/utils/camera_utils.py

import cv2
import time
import requests
import socket
import threading

class CameraManager:
    """Utility class for managing camera initialization and configuration"""
    
    def __init__(self):
        self.camera_sources = {
            'laptop': [],
            'mobile': []
        }
        self.current_camera = None
        self.camera_type = None
    
    @staticmethod
    def initialize_camera(camera_index=0, max_retries=3, retry_delay=1.0):
        """
        Initialize camera with proper error handling and retries.
        
        Args:
            camera_index (int): Camera index (0 for default camera)
            max_retries (int): Maximum number of retry attempts
            retry_delay (float): Delay between retry attempts in seconds
            
        Returns:
            cv2.VideoCapture: Camera object if successful, None otherwise
        """
        for attempt in range(max_retries):
            try:
                print(f"Attempting to initialize camera {camera_index} (attempt {attempt + 1}/{max_retries})")
                
                # Try different backends for better compatibility
                # First, try with no backend specified (use system default)
                cap = cv2.VideoCapture(camera_index)
                
                if cap.isOpened():
                    # Test if we can actually read from the camera
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Successfully opened with default backend
                        # Configure camera settings for better performance
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
                        
                        print("Camera initialized successfully with default backend")
                        return cap
                    else:
                        print("Default backend returned no frames, releasing and trying specific backends")
                        cap.release()
                
                # Try specific backends for better compatibility
                backends = []
                
                # Add available backends based on the platform
                if hasattr(cv2, 'CAP_DSHOW'):  # Windows DirectShow
                    backends.append(cv2.CAP_DSHOW)
                if hasattr(cv2, 'CAP_MSMF'):   # Windows Media Foundation
                    backends.append(cv2.CAP_MSMF)
                if hasattr(cv2, 'CAP_AVFOUNDATION'):  # macOS
                    backends.append(cv2.CAP_AVFOUNDATION)
                if hasattr(cv2, 'CAP_V4L2'):  # Linux
                    backends.append(cv2.CAP_V4L2)
                
                backends.append(cv2.CAP_ANY)  # Generic backend as last resort
                
                for backend in backends:
                    try:
                        cap = cv2.VideoCapture(camera_index, backend)
                        
                        if cap.isOpened():
                            # Test if we can actually read from the camera
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                # Configure camera settings for better performance
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                cap.set(cv2.CAP_PROP_FPS, 30)
                                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
                                
                                print(f"Camera initialized successfully with backend {backend}")
                                return cap
                            else:
                                print(f"Backend {backend} returned no frames, releasing")
                                cap.release()
                        else:
                            print(f"Failed to open camera with backend {backend}")
                    except Exception as e:
                        print(f"Error with backend {backend}: {e}")
                
                print(f"Camera initialization attempt {attempt + 1} failed")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    
            except Exception as e:
                print(f"Error initializing camera on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        print("Failed to initialize camera after all attempts")
        return None
    
    @staticmethod
    def initialize_mobile_camera(url, max_retries=3, retry_delay=1.0):
        """
        Initialize mobile camera (DroidCam) connection.
        
        Args:
            url (str): DroidCam URL (e.g., "http://192.168.1.100:4747/video")
            max_retries (int): Maximum number of retry attempts
            retry_delay (float): Delay between retry attempts in seconds
            
        Returns:
            cv2.VideoCapture: Camera object if successful, None otherwise
        """
        for attempt in range(max_retries):
            try:
                print(f"Attempting to connect to mobile camera {url} (attempt {attempt + 1}/{max_retries})")
                
                # Test connection first
                response = requests.get(url, timeout=5, stream=True)
                if response.status_code != 200:
                    print(f"HTTP error: {response.status_code}")
                    continue
                
                # Initialize VideoCapture with the URL
                cap = cv2.VideoCapture(url)
                
                if cap.isOpened():
                    # Test if we can actually read from the camera
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Configure camera settings
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
                        
                        print("Mobile camera initialized successfully")
                        return cap
                    else:
                        print("Mobile camera returned no frames")
                        cap.release()
                else:
                    print("Failed to open mobile camera connection")
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    
            except Exception as e:
                print(f"Error connecting to mobile camera on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        print("Failed to connect to mobile camera after all attempts")
        return None
    
    def initialize_camera_from_config(self, camera_config):
        """
        Initialize camera based on configuration from camera selection window.
        
        Args:
            camera_config (dict): Camera configuration with 'type' and 'source'
            
        Returns:
            cv2.VideoCapture: Camera object if successful, None otherwise
        """
        self.camera_type = camera_config.get('type')
        source = camera_config.get('source')
        
        if self.camera_type == 'laptop':
            self.current_camera = self.initialize_camera(source)
        elif self.camera_type == 'mobile':
            self.current_camera = self.initialize_mobile_camera(source)
        else:
            print(f"Unknown camera type: {self.camera_type}")
            return None
            
        return self.current_camera
    
    @staticmethod
    def test_droidcam_connection(ip, port="4747"):
        """
        Test DroidCam connection.
        
        Args:
            ip (str): IP address of the device running DroidCam
            port (str): Port number (default: 4747)
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            url = f"http://{ip}:{port}/video"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    @staticmethod
    def scan_for_droidcam_devices():
        """
        Scan local network for DroidCam devices.
        
        Returns:
            list: List of found DroidCam IP addresses
        """
        found_devices = []
        
        try:
            # Get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            
            # Get network base (e.g., 192.168.1.)
            network_base = '.'.join(local_ip.split('.')[:-1]) + '.'
            
            def test_ip(ip):
                if CameraManager.test_droidcam_connection(ip):
                    found_devices.append(ip)
            
            # Create threads for faster scanning
            threads = []
            for i in range(1, 255):
                ip = network_base + str(i)
                thread = threading.Thread(target=test_ip, args=(ip,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete (with timeout)
            for thread in threads:
                thread.join(timeout=1)
                
        except Exception as e:
            print(f"Error scanning for DroidCam devices: {e}")
        
        return found_devices
    
    @staticmethod
    def release_camera(cap):
        """Safely release camera resources"""
        if cap is not None:
            try:
                cap.release()
                print("Camera released successfully")
            except Exception as e:
                print(f"Error releasing camera: {e}")
    
    @staticmethod
    def test_camera_availability():
        """Test if any cameras are available on the system"""
        available_cameras = []
        
        for i in range(5):  # Test first 5 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available_cameras.append(i)
                cap.release()
        
        return available_cameras
    
    def get_camera_info(self):
        """Get information about current camera"""
        if self.current_camera and self.camera_type:
            return {
                'type': self.camera_type,
                'is_opened': self.current_camera.isOpened(),
                'width': self.current_camera.get(cv2.CAP_PROP_FRAME_WIDTH),
                'height': self.current_camera.get(cv2.CAP_PROP_FRAME_HEIGHT),
                'fps': self.current_camera.get(cv2.CAP_PROP_FPS)
            }
        return None
