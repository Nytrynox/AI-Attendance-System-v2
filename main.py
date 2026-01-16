import os
import logging
import argparse
import cv2
import time
import threading
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from PIL import Image, ImageTk
import requests

# Add the project root to path to allow imports
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Core imports
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.anti_spoof import AntiSpoofingDetector
from src.attendance_liveness_detector import AttendanceLivenessDetector
from src.attendance_manager import AttendanceManager
from src.user_manager import UserManager
from src.user_registration_enhanced import EnhancedUserRegistration
from src.sound_manager import SoundManager
from src.email_notification import EmailNotificationManager
from src.utils.data_utils import ensure_directories, load_registered_users
from src.security_logger import log_spoof_attempt
from src.security_event_handler import SecurityEventHandler

# Setup logging - ensure logs directory exists first
os.makedirs(os.path.join('data', 'logs'), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('data', 'logs', f'app_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CameraManager:
    """Integrated camera management system"""
    @staticmethod
    def initialize_camera(camera_index=0, max_retries=3, retry_delay=1.0):
        """Initialize camera with proper error handling and retries. Uses AVFoundation backend on macOS to avoid segfaults."""
        import platform
        # Guard: ensure VideoCapture is available
        if not hasattr(cv2, 'VideoCapture'):
            logger.error("cv2 module does not support VideoCapture. Ensure opencv-python or opencv-contrib-python is installed correctly.")
            return None
        
        is_mac = platform.system() == "Darwin"
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to initialize camera {camera_index} (attempt {attempt + 1}/{max_retries})")
                
                # Force AVFoundation backend on macOS for better stability
                cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION if is_mac else cv2.CAP_ANY)

                if not cap or not cap.isOpened():
                    logger.warning("Camera could not be opened.")
                    if cap:
                        cap.release()
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    continue
                
                # Test read without setting properties first
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning("Camera returned no frames on initial read.")
                    cap.release()
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    continue
                
                # Only set essential properties
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception as e:
                    logger.debug(f"Could not set CAP_PROP_BUFFERSIZE: {e}")
                
                # Defensive: try another read
                ret2, frame2 = cap.read()
                if not ret2 or frame2 is None:
                    logger.warning("Camera failed on second read after setting properties.")
                    cap.release()
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    continue
                
                logger.info("Camera initialized successfully with AVFoundation backend" if is_mac else "Camera initialized successfully with default backend")
                return cap

            except Exception as e:
                logger.error(f"Error initializing camera on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        logger.error("Failed to initialize camera after all attempts")
        return None
    
    @staticmethod
    def initialize_mobile_camera(url, max_retries=3, retry_delay=2.0):
        """Initialize mobile camera (DroidCam) connection with enhanced error handling."""
        logger.info(f"Initializing DroidCam connection to {url}")
        
        for attempt in range(max_retries):
            try:
                logger.info(f"DroidCam connection attempt {attempt + 1}/{max_retries}")
                
                # First, test HTTP connectivity
                try:
                    response = requests.get(url, timeout=3, stream=False)
                    logger.info(f"HTTP test: Status {response.status_code}")
                    
                    if response.status_code != 200:
                        logger.warning(f"DroidCam HTTP error: {response.status_code}")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                except requests.exceptions.RequestException as e:
                    logger.error(f"DroidCam HTTP test failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                
                # Try OpenCV connection with different backends
                backends_to_try = [cv2.CAP_FFMPEG, cv2.CAP_ANY]
                
                for backend in backends_to_try:
                    try:
                        logger.info(f"Trying OpenCV backend: {backend}")
                        cap = cv2.VideoCapture(url, backend)
                        
                        if cap.isOpened():
                            # Set properties for better performance
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for stability
                            
                            # Test frame reading
                            ret, frame = cap.read()
                            if ret and frame is not None and frame.size > 0:
                                logger.info(f"DroidCam connected successfully! Frame shape: {frame.shape}")
                                return cap
                            else:
                                logger.warning("DroidCam opened but no valid frames")
                                cap.release()
                        else:
                            logger.warning(f"Failed to open DroidCam with backend {backend}")
                            if cap:
                                cap.release()
                                
                    except Exception as e:
                        logger.error(f"OpenCV backend {backend} failed: {e}")
                        continue
                
                if attempt < max_retries - 1:
                    logger.info(f"Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    
            except Exception as e:
                logger.error(f"Unexpected error in DroidCam connection attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        logger.error("Failed to connect to DroidCam after all attempts")
        logger.error("Common solutions:")
        logger.error("  1. Restart DroidCam app on phone")
        logger.error("  2. Check WiFi connection")
        logger.error("  3. Verify IP address in DroidCam app")
        logger.error("  4. Try USB connection instead")
        return None
        return None
    
    @staticmethod
    def test_droidcam_connection(ip, port="4747"):
        """Test DroidCam connection."""
        try:
            url = f"http://{ip}:{port}/video"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    @staticmethod
    def test_camera_availability():
        """Test if any cameras are available on the system"""
        available_cameras = []
        
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available_cameras.append(i)
                cap.release()
        
        return available_cameras


class DualCameraWindow:
    """Integrated dual camera window for simultaneous laptop and mobile camera operation"""
    
    def __init__(self, master):
        self.master = master
        self.master.title("Face Attendance System - Dual Camera Mode")
        self.master.geometry("1600x900")
          # Initialize models
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.anti_spoof_detector = AntiSpoofingDetector()
        self.liveness_detector = AttendanceLivenessDetector()
        self.camera_manager = CameraManager()
        
        # Load registered users with proper error handling
        self.registered_users = {}
        try:
            # Ensure the registered users directory exists
            ensure_directories([os.path.join('data', 'registered_users')])
            
            # Load registered users
            users_data = load_registered_users()
            
            # Convert to dictionary format for easier access
            for user_id, name, encoding in users_data:
                self.registered_users[user_id] = {'name': name, 'encoding': encoding}
            
            logger.info(f"Loaded {len(self.registered_users)} registered users")
            
            # Reload face recognizer data to ensure consistency
            self.face_recognizer.reload_user_data()
            
            if len(self.registered_users) > 0:
                logger.info("User database loaded successfully")
                for user_id, user_data in self.registered_users.items():
                    logger.info(f"  - User: {user_data['name']} (ID: {user_id})")
            else:
                logger.info("No registered users found. System ready for new registrations.")
                logger.info("Note: If you had users registered before, they may need to be re-registered")
                logger.info("This can happen after system updates that change the data format")
                
        except Exception as e:
            logger.error(f"Error loading registered users: {e}")
            self.registered_users = {}
            logger.warning("Continuing with empty user database")

        # Camera variables
        self.left_camera = None
        self.right_camera = None
        self.left_camera_running = False
        self.right_camera_running = False
        self.left_video_thread = None
        self.right_video_thread = None
        
        # Attendance mode
        self.attendance_mode = False
        
        # Create GUI
        self.create_widgets()
        
        # Initialize cameras
        self.initialize_cameras()
    
    def create_widgets(self):
        """Create GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Camera frames
        camera_frame = ttk.Frame(main_frame)
        camera_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left camera (Laptop)
        left_frame = ttk.LabelFrame(camera_frame, text="Laptop Camera", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.left_label = tk.Label(left_frame, text="Laptop Camera Feed", bg='black', fg='white')
        self.left_label.pack(fill=tk.BOTH, expand=True)
        
        left_controls = ttk.Frame(left_frame)
        left_controls.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(left_controls, text="Start Left Camera", command=self.start_left_camera).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(left_controls, text="Stop Left Camera", command=self.stop_left_camera).pack(side=tk.LEFT)
        
        # Right camera (Mobile/DroidCam)
        right_frame = ttk.LabelFrame(camera_frame, text="Mobile Camera (DroidCam)", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.right_label = tk.Label(right_frame, text="Mobile Camera Feed", bg='black', fg='white')
        self.right_label.pack(fill=tk.BOTH, expand=True)
        
        right_controls = ttk.Frame(right_frame)
        right_controls.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(right_controls, text="Start Right Camera", command=self.start_right_camera).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(right_controls, text="Stop Right Camera", command=self.stop_right_camera).pack(side=tk.LEFT)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Attendance mode toggle
        self.attendance_var = tk.BooleanVar()
        attendance_check = ttk.Checkbutton(control_frame, text="Attendance Mode", 
                                         variable=self.attendance_var, 
                                         command=self.toggle_attendance_mode)
        attendance_check.pack(side=tk.LEFT, padx=(0, 10))
        
        # Status label
        self.status_label = tk.Label(control_frame, text="Ready", fg='green')
        self.status_label.pack(side=tk.LEFT, padx=(20, 0))
          # Close button
        ttk.Button(control_frame, text="Close", command=self.close_application).pack(side=tk.RIGHT)
    
    def initialize_cameras(self):
        """Initialize both cameras with a small delay to ensure GUI is ready"""
        # Small delay to ensure GUI mainloop is established
        self.master.after(500, self._start_cameras_delayed)
    
    def _start_cameras_delayed(self):
        """Start cameras after GUI is ready"""
        self.start_left_camera()
        self.start_right_camera()
    
    def start_left_camera(self):
        """Start left camera (laptop camera)"""
        if not self.left_camera_running:
            self.left_camera = CameraManager.initialize_camera(0)
            if self.left_camera:
                self.left_camera_running = True
                self.left_video_thread = threading.Thread(target=self.update_left_camera, daemon=True)
                self.left_video_thread.start()
                self.status_label.config(text="Left camera started", fg='green')
            else:
                self.status_label.config(text="Failed to start left camera", fg='red')
    
    def stop_left_camera(self):
        """Stop left camera"""
        self.left_camera_running = False
        if self.left_camera:
            self.left_camera.release()
            self.left_camera = None
        self.status_label.config(text="Left camera stopped", fg='orange')
    
    def start_right_camera(self):
        """Start right camera (mobile camera)"""
        if not self.right_camera_running:
            mobile_url = "http://192.168.29.90:4747/video"
            self.status_label.config(text="Connecting to DroidCam...", fg='orange')
            self.right_camera = CameraManager.initialize_mobile_camera(mobile_url)
            if self.right_camera:
                self.right_camera_running = True
                self.right_video_thread = threading.Thread(target=self.update_right_camera, daemon=True)
                self.right_video_thread.start()
                self.status_label.config(text="DroidCam connected successfully!", fg='green')
            else:
                self.status_label.config(text="DroidCam connection failed", fg='red')
                # Show helpful dialog after a short delay
                self.master.after(1000, self.show_droidcam_help)
    
    def stop_right_camera(self):
        """Stop right camera"""
        self.right_camera_running = False
        if self.right_camera:
            self.right_camera.release()
            self.right_camera = None
        self.status_label.config(text="Right camera stopped", fg='orange')
    
    def update_left_camera(self):
        """Update left camera feed"""
        while self.left_camera_running and self.left_camera:
            try:
                ret, frame = self.left_camera.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    
                    # Process frame for face detection if in attendance mode
                    if self.attendance_mode:
                        frame = self.process_frame_for_attendance(frame)
                      # Convert and display - thread-safe GUI update
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img = img.resize((640, 480), Image.Resampling.LANCZOS)
                    
                    # Schedule GUI update on main thread
                    self.master.after(0, self._update_left_display, img)
                    
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in left camera feed: {e}")
                break
    
    def update_right_camera(self):
        """Update right camera feed"""
        while self.right_camera_running and self.right_camera:
            try:
                ret, frame = self.right_camera.read()
                if ret:
                    # Process frame for face detection if in attendance mode
                    if self.attendance_mode:
                        frame = self.process_frame_for_attendance(frame)
                    
                    # Convert and display - thread-safe GUI update
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img = img.resize((640, 480), Image.Resampling.LANCZOS)
                      # Schedule GUI update on main thread
                    self.master.after(0, self._update_right_display, img)
                    
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in right camera feed: {e}")
                break
    
    def process_frame_for_attendance(self, frame):
        """Process frame for face detection and recognition with liveness verification"""
        try:
            faces = self.face_detector.detect_faces(frame)
            
            for face in faces:
                x, y, w, h, face_crop, landmarks = face
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Extract face for recognition (use pre-extracted face_crop)
                face_img = face_crop
                
                # Recognize face first
                user_id = self.face_recognizer.recognize_face(face_img)
                
                if user_id:
                    # For recognized users, perform liveness verification
                    is_live, verification_complete, status_message, progress = self.liveness_detector.verify_liveness_comprehensive(
                        frame, face, landmarks, user_id
                    )
                    
                    if verification_complete:
                        if is_live:
                            cv2.putText(frame, f"✅ {user_id} - VERIFIED", (x, y-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(frame, "✅ LIVE PERSON", (x, y+h+20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, f"🚫 {user_id} - SPOOF", (x, y-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(frame, "❌ USE REAL FACE", (x, y+h+20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, f"🔍 Verifying {user_id}...", (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                        cv2.putText(frame, "👁️ Blink & move head", (x, y+h+20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                else:
                    # Unknown user - basic check
                    is_real = self.anti_spoof_detector.is_real_face(face_img)
                    if is_real:
                        cv2.putText(frame, "❓ Unknown User", (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "🚫 SPOOF DETECTED", (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
        
        return frame
    
    def toggle_attendance_mode(self):
        """Toggle attendance mode on/off"""
        self.attendance_mode = self.attendance_var.get()
        mode_text = "ON" if self.attendance_mode else "OFF"
        self.status_label.config(text=f"Attendance Mode: {mode_text}", 
                               fg='blue' if self.attendance_mode else 'green')
    
    def close_application(self):
        """Close the application properly"""
        self.stop_left_camera()
        self.stop_right_camera()
        self.master.quit()
        self.master.destroy()
    
    def _update_left_display(self, img):
        """Update left camera display on main thread (thread-safe)"""
        try:
            photo = ImageTk.PhotoImage(img)
            self.left_label.config(image=photo)
            setattr(self.left_label, 'image', photo)  # Prevent garbage collection
        except Exception as e:
            logger.error(f"Error updating left display: {e}")
    
    def _update_right_display(self, img):
        """Update right camera display on main thread (thread-safe)"""
        try:
            photo = ImageTk.PhotoImage(img)
            self.right_label.config(image=photo)
            setattr(self.right_label, 'image', photo)  # Prevent garbage collection
        except Exception as e:
            logger.error(f"Error updating right display: {e}")
    
    def show_droidcam_help(self):
        """Show DroidCam connection help dialog"""
        help_window = tk.Toplevel(self.master)
        help_window.title("DroidCam Setup Help")
        help_window.geometry("500x400")
        help_window.resizable(False, False)
        
        # Make it stay on top
        help_window.transient(self.master)
        help_window.grab_set()
        
        # Help content
        help_text = """
🔍 DroidCam Connection Failed!

To connect your mobile camera:

📱 1. Install DroidCam app on your phone (Android/iOS)
🌐 2. Make sure your phone and computer are on the same WiFi network  
📋 3. Open DroidCam app and note the IP address shown
🔧 4. Current expected IP: 192.168.29.90:4747
✅ 5. Make sure the IP address matches your phone's IP

🛠️ Troubleshooting:
• Check if DroidCam app is running on your phone
• Verify both devices are on the same WiFi network
• Try restarting DroidCam app
• Make sure phone's firewall allows DroidCam
• Check if port 4747 is not blocked
• Try different WiFi network if issues persist

📞 Alternative: Use DroidCam USB mode if WiFi fails
        """
        
        # Text widget with scrollbar
        text_frame = ttk.Frame(help_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Arial', 10))
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)
          # Close button
        button_frame = ttk.Frame(help_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Close", command=help_window.destroy).pack(side=tk.RIGHT)
        retry_btn = ttk.Button(button_frame, text="Retry Connection", 
                              command=lambda: [help_window.destroy(), self.start_right_camera()])
        retry_btn.pack(side=tk.RIGHT, padx=(0, 10))


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Face Attendance System')
    parser.add_argument('--no-gui', action='store_true', help='Run in CLI mode without GUI')
    parser.add_argument('--test', action='store_true', help='Run system tests')
    parser.add_argument('--dual-camera', action='store_true', help='Enable dual camera mode')
    parser.add_argument('--enhanced-camera', action='store_true', help='Enable enhanced camera features')
    parser.add_argument('--registration-mode', action='store_true', help='Run in enhanced registration mode')
    parser.add_argument('--camera', type=int, default=0, help='Camera index to use')
    parser.add_argument('--mobile-camera', type=str, default='http://192.168.29.90:4747/video', 
                       help='Mobile camera URL (default: http://192.168.29.90:4747/video)')
    parser.add_argument('--droidcam-ip', type=str, default='192.168.29.90', 
                       help='DroidCam IP address (default: 192.168.29.90)')
    parser.add_argument('--droidcam-port', type=str, default='4747', 
                       help='DroidCam port (default: 4747)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()


def initialize_system():
    """Initialize system components and ensure required directories exist."""
    logger.info("Initializing Facial Attendance System...")
    
    # Ensure necessary directories exist
    ensure_directories([
        os.path.join('data', 'attendance'),
        os.path.join('data', 'registered_users'),
        os.path.join('data', 'logs'),
        os.path.join('data', 'spoof_images')
    ])
    
    # Initialize components
    model_paths = {
        'face_landmarks': os.path.join('models', 'shape_predictor_68_face_landmarks.dat'),
        'anti_spoof': os.path.join('models', 'anti_spoof_model.h5'),
        'face_recognition': os.path.join('models', 'face_recognition_model.h5')
    }
    
    # Check if models exist (optional check)
    missing_models = []
    for name, path in model_paths.items():
        if not os.path.exists(path):
            missing_models.append(path)
    
    if missing_models:
        logger.warning(f"Some model files not found: {missing_models}")
        logger.info("System will continue with basic functionality")
      # Initialize components
    try:
        face_detector = FaceDetector()
        face_recognizer = FaceRecognizer()
        anti_spoof = AntiSpoofingDetector()
        liveness_detector = AttendanceLivenessDetector()  # For attendance mode with strict verification
        attendance_manager = AttendanceManager()
        user_manager = UserManager()
        
        # Initialize enhanced components
        sound_manager = SoundManager()
        email_manager = EmailNotificationManager()
        enhanced_registration = EnhancedUserRegistration(
            face_detector=face_detector,
            liveness_detector=liveness_detector,
            user_manager=user_manager,
            sound_manager=sound_manager,
            email_manager=email_manager
        )
        
        # Load initial user data
        try:
            logger.info("Loading registered user data...")
            users_data = load_registered_users()
            face_recognizer.reload_user_data()
            
            if len(users_data) > 0:
                logger.info(f"Successfully loaded {len(users_data)} registered users")
                for user_id, name, encoding in users_data:
                    logger.info(f"  - {name} (ID: {user_id}) - Encoding ready")
            else:
                logger.info("No registered users found")
                
                # Check for incompatible users and provide helpful feedback
                from src.utils.data_utils import check_and_clean_user_data
                incompatible_users = check_and_clean_user_data()
                if incompatible_users:
                    logger.info(f"Found {len(incompatible_users)} users that need re-registration:")
                    for user_id, user_name in incompatible_users:
                        logger.info(f"  - {user_name} (ID: {user_id}) needs re-registration")
                    logger.info("📝 Note: These users can re-register using any of these methods:")
                    logger.info("  1. Run: python enhanced_registration_tool.py")
                    logger.info("  2. Run: python main.py --registration-mode")
                    logger.info("  3. Use the registration feature in the main application")
                    logger.info("  4. Run: python reset_user_data.py (to start fresh)")
                else:
                    logger.info("System ready for new user registrations")
                
        except Exception as e:
            logger.warning(f"Could not load user data: {e}")
            logger.info("System will continue with empty user database")
            logger.info("💡 Tip: If you had users before, try running: python fix_user_data.py")
        
        logger.info("System initialization completed successfully")
        
        return {
            'face_detector': face_detector,
            'face_recognizer': face_recognizer,
            'anti_spoof': anti_spoof,
            'liveness_detector': liveness_detector,
            'attendance_manager': attendance_manager,
            'user_manager': user_manager,
            'sound_manager': sound_manager,
            'email_manager': email_manager,
            'enhanced_registration': enhanced_registration
        }
    
    except Exception as e:
        logger.error(f"Error initializing system components: {e}")
        return None


def run_cli_mode(components, args):
    """Run the system in command-line interface mode."""
    logger.info("Starting in CLI mode")
    
    face_detector = components['face_detector']
    face_recognizer = components['face_recognizer']
    anti_spoof_detector = components['anti_spoof']
    liveness_detector = components['liveness_detector']
    attendance_manager = components['attendance_manager']
    user_manager = components['user_manager']
    sound_manager = components['sound_manager']
    email_manager = components['email_manager']
    enhanced_registration = components['enhanced_registration']
    
    # Verify user data loading
    try:
        print("\n🔍 Checking registered users...")
        users_data = load_registered_users()
        print(f"Found {len(users_data)} registered users in database:")
        for user_id, name, encoding in users_data:
            print(f"  - {name} (ID: {user_id}) - Encoding shape: {encoding.shape}")
        
        # Verify face recognizer has the data
        face_recognizer.reload_user_data()
        print(f"Face recognizer loaded {len(face_recognizer.known_face_encodings)} encodings")
        
    except Exception as e:
        print(f"⚠️  Error loading user data: {e}")
        logger.exception("User data loading error in CLI mode")
    
    # Initialize camera
    if args.mobile_camera and CameraManager.test_droidcam_connection(args.droidcam_ip, args.droidcam_port):
        camera = CameraManager.initialize_mobile_camera(args.mobile_camera)
        print("Using mobile camera (DroidCam)")
    else:
        camera = CameraManager.initialize_camera(args.camera)
        print(f"Using local camera {args.camera}")
    
    if not camera:
        print("ERROR: Could not initialize camera")
        return
    
    print("\n=== Face Attendance System (CLI Mode) ===")
    print("Controls:")
    print("  'q' - Quit")
    print("  'a' - Mark attendance for recognized user")
    print("  'r' - Register new user (Enhanced with background analysis)")
    print("  'd' - Switch to dual camera mode")
    print("  ESC - Exit")
    print("=" * 50)
    
    current_user_id = None
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                logger.error("Failed to capture frame from camera")
                break
              # Flip frame horizontally for mirror effect (except for mobile cameras)
            if args.camera is not None:
                frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces = face_detector.detect_faces(frame)
            
            for face in faces:
                x, y, w, h, face_crop, landmarks = face
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Extract face region (use pre-extracted face_crop)
                face_img = face_crop
                
                # First do face recognition to identify the user
                user_id = face_recognizer.recognize_face(face_img)
                
                if user_id:
                    # For recognized users, perform comprehensive liveness verification
                    is_live, verification_complete, status_message = liveness_detector.verify_liveness(
                        frame, face, landmarks, user_id
                    )
                    
                    if verification_complete:
                        if is_live:
                            # Liveness verified - allow attendance marking
                            current_user_id = user_id
                            cv2.putText(frame, f"✅ {user_id} - VERIFIED", (x, y-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.putText(frame, "Press 'a' for attendance", (x, y+h+25), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.putText(frame, "✅ LIVE PERSON", (x, y+h+45), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            # Liveness failed - spoof detected
                            current_user_id = None
                            cv2.putText(frame, f"🚫 {user_id} - SPOOF DETECTED!", (x, y-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            cv2.putText(frame, "❌ USE REAL FACE ONLY", (x, y+h+25), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            log_spoof_attempt(f"CLI_MODE_{user_id}")
                    else:
                        # Still verifying liveness
                        current_user_id = None
                        cv2.putText(frame, f"🔍 Verifying {user_id}...", (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
                        cv2.putText(frame, status_message, (x, y+h+25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                        cv2.putText(frame, "👁️ Blink naturally & move head", (x, y+h+45), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                else:
                    # Unknown user - just basic anti-spoofing
                    is_real = anti_spoof_detector.is_real_face(face_img)
                    if is_real:
                        current_user_id = None
                        cv2.putText(frame, "❓ Unknown User", (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        cv2.putText(frame, "Press 'r' to register", (x, y+h+25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    else:
                        cv2.putText(frame, "🚫 SPOOF DETECTED!", (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        log_spoof_attempt("CLI_MODE_UNKNOWN")
            
            # Display the frame
            cv2.imshow('Face Attendance System', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                if current_user_id:
                    attendance_manager.mark_attendance(current_user_id)
                    print(f"Attendance marked for {current_user_id}")
                else:
                    print("No recognized user to mark attendance for")
            elif key == ord('r'):
                print("\nStarting enhanced user registration with background analysis...")
                try:
                    # Use enhanced registration with 5-second background analysis
                    success = enhanced_registration.register_user_with_camera(camera)
                    if success:
                        face_recognizer.reload_user_data()
                        print("Enhanced user registration completed successfully.")
                    else:
                        print("Enhanced user registration failed or was cancelled.")
                except Exception as e:
                    logger.exception(f"Enhanced registration error: {e}")
                    print(f"Enhanced registration failed: {e}")
            elif key == ord('d'):
                print("\nSwitching to dual camera mode...")
                camera.release()
                cv2.destroyAllWindows()
                run_dual_camera_mode()
                return
    
    except Exception as e:
        logger.exception(f"Error in CLI mode: {e}")
        print(f"Error: {e}")
    
    finally:
        camera.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # OpenCV GUI functions may not be available in headless environment
            pass


def run_dual_camera_mode():
    """Run dual camera mode"""
    try:
        root = tk.Tk()
        app = DualCameraWindow(root)
        root.protocol("WM_DELETE_WINDOW", app.close_application)
        root.mainloop()
    except Exception as e:
        logger.error(f"Dual camera mode error: {e}")
        print(f"Error in dual camera mode: {e}")


def run_gui_mode(components, args):
    """Run the system with a graphical user interface."""
    logger.info("Starting in GUI mode")
    
    try:
        # Check if dual camera mode is requested
        if args.dual_camera:
            run_dual_camera_mode()
            return
        
        # Try to load available GUI modules
        try:
            from src.gui.main_window_complete import CompleteMainWindow
            logger.info("Loading Complete GUI")
            
            root = tk.Tk()
            root.title("Face Attendance System")
            root.geometry("1200x800")
            
            app = CompleteMainWindow(root, components)
            
            # Create a simple fallback for window closing
            def on_closing():
                try:
                    if hasattr(app, 'close_application'):
                        app.close_application()
                    else:
                        root.quit()
                        root.destroy()
                except Exception:
                    root.quit()
                    root.destroy()
            
            root.protocol("WM_DELETE_WINDOW", on_closing)
            root.mainloop()
            
        except ImportError:
            logger.warning("Complete GUI not available, using dual camera mode as fallback")
            run_dual_camera_mode()
        
    except Exception as e:
        logger.error(f"GUI error: {e}")
        print("Falling back to CLI mode...")
        run_cli_mode(components, args)


def run_registration_mode(components, args):
    """Run the system in dedicated enhanced registration mode."""
    logger.info("Starting in Enhanced Registration mode")
    
    enhanced_registration = components['enhanced_registration']
    
    print("\n=== Enhanced Registration Mode ===")
    print("This mode provides:")
    print("  • 5-second background analysis")
    print("  • Advanced liveness detection")
    print("  • Automatic pattern learning")
    print("  • Spoof detection with alerts")
    print("=" * 50)
    
    # Initialize camera
    if args.mobile_camera and CameraManager.test_droidcam_connection(args.droidcam_ip, args.droidcam_port):
        camera = CameraManager.initialize_mobile_camera(args.mobile_camera)
        print("Using mobile camera (DroidCam)")
    else:
        camera = CameraManager.initialize_camera(args.camera)
        print(f"Using local camera {args.camera}")
    
    if not camera:
        print("ERROR: Could not initialize camera")
        return
    
    try:
        # Run continuous registration mode
        print("\nPress 'r' to start registration or 'q' to quit...")
        
        while True:
            ret, frame = camera.read()
            if not ret:
                continue
            
            # Display preview
            cv2.imshow('Enhanced Registration Mode - Press R to Register, Q to Quit', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("\nStarting enhanced registration with background analysis...")
                success = enhanced_registration.register_user_with_camera(camera)
                if success:
                    print("✅ Registration completed successfully!")
                else:
                    print("❌ Registration failed or was cancelled.")
                print("\nPress 'r' to register another user or 'q' to quit...")
                
    except Exception as e:
        logger.exception(f"Error in registration mode: {e}")
        print(f"Error: {e}")
    
    finally:
        camera.release()
        cv2.destroyAllWindows()


def run_system_tests():
    """Run integrated system tests"""
    print("🧪 Running Face Attendance System Tests")
    print("=" * 50)
    
    # Test camera availability
    print("Testing camera availability...")
    cameras = CameraManager.test_camera_availability()
    print(f"Available cameras: {cameras}")
    
    # Test DroidCam connection
    print("Testing DroidCam connection...")
    droidcam_available = CameraManager.test_droidcam_connection("192.168.29.90")
    print(f"DroidCam available: {droidcam_available}")
    
    # Test user data loading
    print("\nTesting user data loading...")
    try:
        users_data = load_registered_users()
        print(f"✓ Found {len(users_data)} registered users:")
        for user_id, name, encoding in users_data:
            print(f"  - {name} (ID: {user_id}) - Encoding shape: {encoding.shape}")
        
        # Test directory structure
        users_dir = os.path.join('data', 'registered_users')
        if os.path.exists(users_dir):
            print(f"✓ Users directory exists: {users_dir}")
            subdirs = [d for d in os.listdir(users_dir) if os.path.isdir(os.path.join(users_dir, d)) and not d.startswith('.')]
            print(f"  User subdirectories: {subdirs}")
        else:
            print(f"✗ Users directory missing: {users_dir}")
            
    except Exception as e:
        print(f"✗ User data loading failed: {e}")
    
    # Test model files
    print("\nTesting model files...")
    model_files = [
        'models/shape_predictor_68_face_landmarks.dat',
        'models/anti_spoof_model.h5',
        'models/face_recognition_model.h5'
    ]
    
    for model_file in model_files:
        exists = os.path.exists(model_file)
        print(f"  {model_file}: {'✓' if exists else '✗'}")
    
    print("\n✅ System test completed!")


def main():
    """Main function to run the Face Attendance System."""
    args = parse_arguments()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
        
        # Show user data information in debug mode
        try:
            print("\n🔍 DEBUG: User Data Information")
            print("-" * 40)
            users_data = load_registered_users()
            print(f"Registered users found: {len(users_data)}")
            
            users_dir = os.path.join('data', 'registered_users')
            if os.path.exists(users_dir):
                subdirs = [d for d in os.listdir(users_dir) if os.path.isdir(os.path.join(users_dir, d)) and not d.startswith('.')]
                print(f"User directories: {subdirs}")
                
                for subdir in subdirs:
                    subdir_path = os.path.join(users_dir, subdir)
                    files = os.listdir(subdir_path)
                    print(f"  {subdir}/: {files}")
            print("-" * 40)
        except Exception as e:
            print(f"Debug user data check failed: {e}")
    
    # Run tests if requested
    if args.test:
        run_system_tests()
        return 0
    
    try:
        components = initialize_system()
        if not components:
            return 1
        
        if args.registration_mode:
            run_registration_mode(components, args)
        elif args.no_gui:
            run_cli_mode(components, args)
        else:
            run_gui_mode(components, args)
            
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        print(f"ERROR: {e}")
        return 1
    
    logger.info("Application closed normally")
    return 0


if __name__ == "__main__":
    sys.exit(main())
