# src/gui/main_window_complete.py

import tkinter as tk
from tkinter import messagebox
import cv2
import os
import numpy as np
import dlib
from datetime import datetime
import threading
import time
from PIL import Image, ImageTk
from src.utils.data_utils import load_registered_users, save_attendance
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.anti_spoof import AntiSpoofingDetector
from src.utils.camera_utils import CameraManager
from src.security_event_handler import SecurityEventHandler


class CompleteMainWindow:
    def __init__(self, master, components=None):
        self.master = master
        self.master.title("Complete Facial Attendance System")
        self.master.geometry("1200x800")
        
        # Use provided components or initialize new ones
        if components:
            self.face_detector = components['face_detector']
            self.face_recognizer = components['face_recognizer']
            self.anti_spoof_detector = components['anti_spoof']
            self.enhanced_registration = components['enhanced_registration']
            self.sound_manager = components['sound_manager']
            self.email_manager = components['email_manager']
            print("[INFO] Using shared system components with enhanced registration")
        else:
            # Initialize models (fallback for backward compatibility)
            self.face_detector = FaceDetector()
            self.face_recognizer = FaceRecognizer()
            self.anti_spoof_detector = AntiSpoofingDetector()
            self.enhanced_registration = None
            self.sound_manager = None
            self.email_manager = None
            print("[INFO] Initialized new components (enhanced registration not available)")
        
        self.camera_manager = CameraManager()
        
        # Initialize security event handler for sound and email alerts
        self.security_handler = SecurityEventHandler()
          # Load registered users and ensure face recognizer is synced
        self.registered_users = load_registered_users()
        self.face_recognizer.reload_user_data()  # Ensure face recognizer has latest data
        
        print(f"[INFO] Loaded {len(self.registered_users)} users for attendance")
        print(f"[INFO] Face recognizer has {len(self.face_recognizer.known_ids)} users")
        print(f"[INFO] Security alerts initialized (Sound + Email)")
        
        # Display guidance message if no users are registered
        self.has_registered_users = len(self.registered_users) > 0
          # Camera variables
        self.cap = None
        self.camera_running = False
        self.attendance_mode = False
        self._current_photo = None
        
        # Liveness detection for attendance (automatic verification)
        self.user_liveness_data = {}  # Track liveness per user_id
        self.liveness_required_frames = 30  # Frames to verify liveness (1 second at 30fps)
        self.blink_threshold = 0.26
        self.movement_threshold = 25
        
        # Phone detection for attendance 
        self.phone_detection_enabled = True
        
        # Initialize dlib predictor for liveness detection
        try:
            predictor_path = "models/shape_predictor_68_face_landmarks.dat"
            if os.path.exists(predictor_path) and hasattr(dlib, 'shape_predictor'):
                self.predictor = dlib.shape_predictor(predictor_path)  # type: ignore
                self.use_landmarks = True
                print("[INFO] Landmark predictor loaded for liveness detection")
            else:
                self.use_landmarks = False
                print("[WARNING] Landmark predictor not found, using simplified liveness detection")
        except Exception as e:
            self.use_landmarks = False
            print(f"[WARNING] Failed to load landmark predictor: {e}")
        
        # Create UI
        self.create_ui()
        
    def create_ui(self):
        """Create the complete user interface"""
        # Main container
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top control panel
        control_panel = tk.Frame(main_frame, height=100)
        control_panel.pack(fill=tk.X, pady=(0, 10))
        control_panel.pack_propagate(False)
        
        # Title
        title_label = tk.Label(control_panel, text="Complete Facial Attendance System", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
          # Button frame
        button_frame = tk.Frame(control_panel)
        button_frame.pack()
        
        # Buttons
        self.register_btn = tk.Button(button_frame, text="Register New User", 
                                     command=self.open_register_window, 
                                     bg="lightblue", font=("Arial", 10))
        self.register_btn.pack(side=tk.LEFT, padx=5)
        
        # Highlight the register button if no users exist
        if not self.has_registered_users:
            self.register_btn.config(bg="yellow", fg="darkred", 
                                   font=("Arial", 10, "bold"),
                                   relief=tk.RAISED, bd=3)
        
        self.start_attendance_btn = tk.Button(button_frame, text="Start Attendance Mode", 
                                            command=self.start_attendance_mode, 
                                            bg="lightgreen", font=("Arial", 10))
        self.start_attendance_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_attendance_btn = tk.Button(button_frame, text="Stop Attendance Mode", 
                                           command=self.stop_attendance_mode, 
                                           state=tk.DISABLED, bg="lightcoral", font=("Arial", 10))
        self.stop_attendance_btn.pack(side=tk.LEFT, padx=5)
        
        self.view_attendance_btn = tk.Button(button_frame, text="View Attendance", 
                                           command=self.view_attendance, 
                                           bg="lightyellow", font=("Arial", 10))
        self.view_attendance_btn.pack(side=tk.LEFT, padx=5)
        
        self.refresh_btn = tk.Button(button_frame, text="Refresh Data", 
                                   command=self.refresh_data, 
                                   bg="lightcyan", font=("Arial", 10))
        self.refresh_btn.pack(side=tk.LEFT, padx=5)
        
        self.quit_btn = tk.Button(button_frame, text="Quit", 
                                command=self.close_application, 
                                bg="gray", font=("Arial", 10))
        self.quit_btn.pack(side=tk.LEFT, padx=5)
        
        # Main content area
        content_frame = tk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for camera
        left_panel = tk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Camera frame
        camera_frame = tk.LabelFrame(left_panel, text="Live Camera Feed", padx=10, pady=10)
        camera_frame.pack(fill=tk.BOTH, expand=True)
        
        self.camera_label = tk.Label(camera_frame, text="Camera not started", 
                                   bg="black", fg="white", width=80, height=35)
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Legend
        legend_frame = tk.Frame(left_panel)
        legend_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(legend_frame, text="Legend:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        tk.Label(legend_frame, text="🟢 Recognized User", fg="green").pack(side=tk.LEFT, padx=10)
        tk.Label(legend_frame, text="🔴 Spoofing Attempt", fg="red").pack(side=tk.LEFT, padx=10)
        tk.Label(legend_frame, text="🟠 Unknown User", fg="orange").pack(side=tk.LEFT, padx=10)
        tk.Label(legend_frame, text="🔵 Processing", fg="blue").pack(side=tk.LEFT, padx=10)
        
        # Right panel for status and attendance
        right_panel = tk.Frame(content_frame, width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)
        
        # Status frame
        status_frame = tk.LabelFrame(right_panel, text="System Status", padx=10, pady=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_text = tk.Text(status_frame, height=8, wrap=tk.WORD, state=tk.DISABLED)
        status_scrollbar = tk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scrollbar.set)
        
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
          # Today's attendance frame
        attendance_frame = tk.LabelFrame(right_panel, text="Today's Attendance", padx=10, pady=10)
        attendance_frame.pack(fill=tk.BOTH, expand=True)
        
        # Attendance listbox with scrollbar
        attendance_list_frame = tk.Frame(attendance_frame)
        attendance_list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.attendance_listbox = tk.Listbox(attendance_list_frame)
        
        # Show a message if no users are registered
        if not self.has_registered_users:
            self.attendance_listbox.insert(tk.END, "")  # Empty line
            self.attendance_listbox.insert(tk.END, "⭐ No users registered yet")
            self.attendance_listbox.insert(tk.END, "")
            self.attendance_listbox.insert(tk.END, "Please register new users first")
            self.attendance_listbox.insert(tk.END, "using the 'Register New User'")
            self.attendance_listbox.insert(tk.END, "button above")
            self.attendance_listbox.insert(tk.END, "")
            self.attendance_listbox.itemconfig(1, fg="blue")  # Color the title
        attendance_listbox_scrollbar = tk.Scrollbar(attendance_list_frame, orient=tk.VERTICAL, 
                                                   command=self.attendance_listbox.yview)
        self.attendance_listbox.configure(yscrollcommand=attendance_listbox_scrollbar.set)
        
        self.attendance_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        attendance_listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Refresh attendance button
        refresh_btn = tk.Button(attendance_frame, text="Refresh Attendance", 
                              command=self.refresh_attendance_list, bg="lightcyan")
        refresh_btn.pack(pady=5)
        
        # Statistics frame
        stats_frame = tk.LabelFrame(right_panel, text="Statistics", padx=10, pady=10)
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.stats_label = tk.Label(stats_frame, text="Registered Users: 0\nToday's Attendance: 0", 
                                  justify=tk.LEFT)
        self.stats_label.pack()        # Initialize
        self.update_status("System initialized. Ready to start attendance mode.")
        self.refresh_attendance_list()
        self.update_statistics()
    
    def update_status(self, message):
        """Update status display with timestamp"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)
    
    def open_register_window(self):
        """Open the simple registration window"""
        try:
            # Import the simple registration GUI
            from src.gui.simple_registration_gui import SimpleRegistrationWindow
            
            self.update_status("🔄 Opening registration window...")
            
            # Create registration window
            register_window = tk.Toplevel(self.master)
            register_window.transient(self.master)
            
            # Create registration app
            def on_registration_close():
                """Handle registration window close"""
                self.update_status("✅ Registration window closed")
                # Reload face recognizer data to include new user
                if hasattr(self, 'face_recognizer'):
                    self.face_recognizer.reload_user_data()
                    self.update_status("🔄 Face recognition data reloaded")
            
            registration_app = SimpleRegistrationWindow(register_window, on_close_callback=on_registration_close)
            self.update_status("📷 Registration window opened - Follow the instructions")
                
        except ImportError as e:
            self.update_status(f"❌ Error: Could not open registration window. {str(e)}")
        except Exception as e:
            self.update_status(f"❌ Unexpected error: {str(e)}")

    def start_attendance_mode(self):
        """Start attendance marking with improved camera initialization"""
        if self.camera_running:
            return
            
        # Initialize camera with retry logic
        self.cap = self.camera_manager.initialize_camera()
        if not self.cap:
            messagebox.showerror("Camera Error", 
                               "Cannot access camera! Please check:\n"
                               "1. Camera is connected\n"
                               "2. Camera is not used by another application\n"
                               "3. Camera drivers are installed")
            self.update_status("Failed to start attendance mode - camera error")
            return
            
        self.camera_running = True
        self.attendance_mode = True
        
        # Update button states
        self.start_attendance_btn.config(state=tk.DISABLED)
        self.stop_attendance_btn.config(state=tk.NORMAL)
        
        self.update_status("Attendance mode started. Live recognition active.")
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self.attendance_camera_loop, daemon=True)
        self.camera_thread.start()
        
    def stop_attendance_mode(self):
        """Stop attendance mode and camera"""
        self.camera_running = False
        self.attendance_mode = False

        if self.cap:
            self.camera_manager.release_camera(self.cap)
            self.cap = None
          # Update button states
        self.start_attendance_btn.config(state=tk.NORMAL)
        self.stop_attendance_btn.config(state=tk.DISABLED)
        
        self.update_status("Attendance mode stopped.")
        
        # Clear camera display
        self.camera_label.configure(image="", text="Camera stopped")
        setattr(self.camera_label, 'image', None)  # Clear reference
        
    def attendance_camera_loop(self):
        """Camera loop for attendance marking"""
        last_recognition_time = {}
        recognition_cooldown = 2  # 2 seconds cooldown
        last_reload_check = time.time()
        reload_check_interval = 1.0  # Check for new users every 1 second
        
        while self.camera_running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    break
                    
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    continue
                    
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Periodically check for new user registrations
                current_time = time.time()
                if current_time - last_reload_check >= reload_check_interval:
                    # Check if new users were registered and reload if needed
                    self.face_recognizer.check_and_reload_if_needed()
                    
                    # Also reload the registered users list for consistency
                    try:
                        from src.utils.data_utils import load_registered_users
                        new_users = load_registered_users()
                        if len(new_users) != len(self.registered_users):
                            old_count = len(self.registered_users)
                            self.registered_users = new_users
                            self.update_status_threadsafe(f"Users updated: {len(new_users)} registered users (was {old_count})")
                            
                            # Update the UI statistics as well
                            self.master.after(0, self.update_statistics)
                    except Exception as e:
                        pass  # Don't break attendance mode for reload issues
                    
                    last_reload_check = current_time
                
                # Process frame for attendance
                frame_with_feedback = self.process_attendance_frame(frame, last_recognition_time, recognition_cooldown)
                
                # Display frame
                self.display_frame(frame_with_feedback)
                
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                self.update_status_threadsafe(f"Camera error: {str(e)}")
                break
                
        # Clean up if loop exits due to errors
        if self.camera_running:
            self.master.after(0, self.stop_attendance_mode)
            
    def process_attendance_frame(self, frame, last_recognition_time, recognition_cooldown):
        """Process frame for attendance marking with visual feedback"""
        frame_copy = frame.copy()
        current_time = time.time()
        
        try:
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            
            if not faces:
                return frame_copy
                
            for face in faces:
                if isinstance(face, tuple) and len(face) >= 4:
                    if len(face) == 6:  # (x, y, w, h, face_crop, landmarks)
                        left, top, width, height = face[:4]
                        right = left + width
                        bottom = top + height
                        face_crop = face[4] if face[4] is not None else frame[top:bottom, left:right]
                    else:  # (left, top, right, bottom)
                        left, top, right, bottom = face
                        face_crop = frame[top:bottom, left:right]
                else:
                    continue
                
                # Check if face is real with enhanced feedback
                is_real, prediction = self.anti_spoof_detector.predict(face_crop, debug=True)
                is_phone = self.detect_phone_screen(face_crop)
                # Only mark as spoof if phone detected or model is confident it's fake
                is_spoof = is_phone or (prediction < 0.3)
                if is_spoof:
                    cv2.rectangle(frame_copy, (left, top), (right, bottom), (0, 0, 255), 6)
                    cv2.putText(frame_copy, f"AntiSpoof: {prediction:.2f}", (left, bottom+45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    if is_phone:
                        cv2.putText(frame_copy, "📱 PHONE DETECTED", (left, bottom+65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(frame_copy, "🚫 SPOOF DETECTED - PHONE/PHOTO REJECTED", (left, top-90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
                    cv2.putText(frame_copy, "❌ FAKE FACE NOT ALLOWED", (left, top-65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame_copy, "📵 NO PHONES OR PICTURES ALLOWED", (left, top-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame_copy, "👤 USE REAL LIVE FACE ONLY", (left, top-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame_copy, "⚠️ SYSTEM WILL NOT MARK ATTENDANCE", (left, bottom+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    blink = int(cv2.getTickCount() / cv2.getTickFrequency() * 10) % 2
                    if blink:
                        cv2.rectangle(frame_copy, (left-15, top-15), (right+15, bottom+15), (0, 0, 200), 8)
                        marker_size = 20
                        cv2.line(frame_copy, (left-marker_size, top-marker_size), (left, top), (0, 0, 255), 5)
                        cv2.line(frame_copy, (right+marker_size, top-marker_size), (right, top), (0, 0, 255), 5)
                        cv2.line(frame_copy, (left-marker_size, bottom+marker_size), (left, bottom), (0, 0, 255), 5)
                        cv2.line(frame_copy, (right+marker_size, bottom+marker_size), (right, bottom), (0, 0, 255), 5)
                    
                    # Trigger security alerts (sound + email with captured image)
                    spoof_type = "Phone Screen" if is_phone else "Fake Face/Photo"
                    person_description = "Unidentified Person"
                    
                    # Try to get some description from the frame context if possible
                    try:
                        h, w = face_crop.shape[:2] if face_crop is not None else (0, 0)
                        person_description = f"Person at position ({left},{top}) - Face size: {w}x{h}"
                    except:
                        person_description = "Unknown Person"
                    
                    # Handle comprehensive spoof detection (includes repeated attempt tracking)
                    self.security_handler.handle_comprehensive_spoof_detection(
                        frame_copy, person_description, prediction, spoof_type
                    )
                    
                    self.update_status_threadsafe("🚫 SPOOF ATTEMPT BLOCKED - Phone/Photo/Fake face rejected! Use real live face only.")
                    continue
                    
                # Recognize face
                recognition_result = self.face_recognizer.recognize_face(face_crop)
                user_id, recognized_name, confidence = recognition_result
                
                if user_id and user_id in [u[0] for u in self.registered_users]:
                    # Check cooldown
                    if user_id in last_recognition_time:
                        time_since_last = current_time - last_recognition_time[user_id]
                        if time_since_last < recognition_cooldown:
                            # Still in cooldown
                            cv2.rectangle(frame_copy, (left, top), (right, bottom), (0, 255, 255), 3)
                            cv2.putText(frame_copy, f"COOLDOWN: {int(recognition_cooldown - time_since_last)}s", 
                                      (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            continue
                    
                    # Get user name
                    user_name = next((u[1] for u in self.registered_users if u[0] == user_id), "Unknown")
                    
                    # Mark attendance
                    save_attendance(user_id, user_name)
                    last_recognition_time[user_id] = current_time
                    
                    # Trigger success sound and notification
                    self.security_handler.handle_successful_attendance(user_id, user_name, confidence)
                    
                    # Visual feedback with enhanced styling
                    cv2.rectangle(frame_copy, (left, top), (right, bottom), (0, 255, 0), 4)
                    cv2.putText(frame_copy, "✅ ATTENDANCE MARKED", 
                              (left, top-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame_copy, f"{user_name} (ID: {user_id})", 
                              (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Add green pulsing effect for successful marking
                    pulse = int(abs(cv2.getTickCount() / cv2.getTickFrequency() * 2) % 2)
                    if pulse:
                        cv2.rectangle(frame_copy, (left-3, top-3), (right+3, bottom+3), (0, 200, 0), 2)
                    
                    # Update UI
                    self.update_status_threadsafe(f"Attendance marked for {user_name} (ID: {user_id})")
                    self.master.after(100, self.refresh_attendance_list)
                    
                else:
                    # Unknown user
                    cv2.rectangle(frame_copy, (left, top), (right, bottom), (0, 165, 255), 3)
                    cv2.putText(frame_copy, "UNKNOWN USER", 
                              (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    
                    # Trigger unknown user sound
                    self.security_handler.handle_unknown_user(confidence)
                    
        except Exception as e:
            self.update_status_threadsafe(f"Frame processing error: {str(e)}")
            
        return frame_copy
        
    def update_status_threadsafe(self, message):
        """Thread-safe status update"""
        self.master.after(0, lambda: self.update_status(message))
        
    def display_frame(self, frame):
        """Convert and display frame in GUI"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)
            
            # Resize to fit display
            display_size = (640, 480)
            pil_image = pil_image.resize(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Store reference to prevent garbage collection
            self._current_photo = photo
            
            # Update display in main thread
            self.master.after(0, lambda p=photo: self.update_camera_display(p))
            
        except Exception:
            pass  # Silently ignore display errors
            
    def update_camera_display(self, photo):
        """Update camera display (main thread)"""
        try:
            if hasattr(self, 'camera_label') and self.camera_label.winfo_exists():
                self.camera_label.configure(image=photo, text="")
                setattr(self.camera_label, 'image', photo)  # Keep reference to prevent GC
        except Exception:
            pass  # Silently ignore display errors
    
    def refresh_attendance_list(self):
        """Refresh today's attendance list"""
        try:
            self.attendance_listbox.delete(0, tk.END)
            
            # Get today's date
            today = datetime.now().strftime("%Y-%m-%d")
            attendance_file = os.path.join("data/attendance", f"{today}.csv")
            
            if os.path.exists(attendance_file):
                import csv
                with open(attendance_file, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 3:
                            self.attendance_listbox.insert(tk.END, f"{row[1]} (ID: {row[0]}) - {row[2]}")
            else:
                self.attendance_listbox.insert(tk.END, "No attendance records for today")
                
        except Exception as e:
            self.update_status(f"Error refreshing attendance: {str(e)}")
    
    def update_statistics(self):
        """Update statistics display"""
        try:
            # Count registered users
            num_users = len(self.registered_users)

            # Count today's attendance
            today = datetime.now().strftime("%Y-%m-%d")
            attendance_file = os.path.join("data/attendance", f"{today}.csv")

            num_attendance = 0
            if os.path.exists(attendance_file):
                import csv
                with open(attendance_file, 'r') as f:
                    reader = csv.reader(f)
                    num_attendance = sum(1 for row in reader)

            self.stats_label.config(text=f"Registered Users: {num_users}\nToday's Attendance: {num_attendance}")
        except Exception as e:
            self.update_status(f"Error updating statistics: {str(e)}")
    
    def view_attendance(self):
        """Open attendance records in a new window"""
        try:
            import subprocess
            import sys
            
            # Get today's attendance file
            today = datetime.now().strftime("%Y-%m-%d")
            attendance_file = os.path.join("data/attendance", f"{today}.csv")
            
            if os.path.exists(attendance_file):
                if sys.platform.startswith('win'):
                    os.startfile(attendance_file)  # type: ignore  # Windows only
                elif sys.platform.startswith('darwin'):
                    subprocess.call(['open', attendance_file])
                else:
                    subprocess.call(['xdg-open', attendance_file])
            else:
                messagebox.showinfo("No Records", "No attendance records found for today.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Could not open attendance file: {str(e)}")
    
    def close_application(self):
        """Close the application safely"""
        if self.camera_running:
            self.stop_attendance_mode()
            
        self.master.quit()
        self.master.destroy()
        
    def refresh_data(self):
        """Refresh registered users, attendance list, and statistics."""
        try:
            # Reload registered users
            self.registered_users = load_registered_users()
            
            # Ensure face recognizer is synced with updated data
            self.face_recognizer.reload_user_data()

            # Refresh attendance list
            self.refresh_attendance_list()

            # Update statistics
            self.update_statistics()

            # Update status
            self.update_status(f"Data refreshed successfully. Users: {len(self.registered_users)}")
            print(f"[INFO] Refresh complete - Attendance users: {len(self.registered_users)}, Recognizer users: {len(self.face_recognizer.known_ids)}")
        except Exception as e:
            self.update_status(f"Error refreshing data: {str(e)}")
            print(f"[ERROR] Refresh failed: {e}")
    
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        """Calculate Eye Aspect Ratio for blink detection"""
        try:
            # Calculate distances between eye landmarks
            A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
            B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
            C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
            ear = (A + B) / (2.0 * C)
            return ear
        except Exception as e:
            print(f"Error in calculate_eye_aspect_ratio: {e}")
            return 0.3
    
    def detect_blink(self, frame, face_landmarks):
        """Detect blink using Eye Aspect Ratio"""
        try:
            if self.use_landmarks and face_landmarks is not None:
                left_eye = face_landmarks[36:42]
                right_eye = face_landmarks[42:48] 
                left_ear = self.calculate_eye_aspect_ratio(left_eye)
                right_ear = self.calculate_eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
                return ear < self.blink_threshold
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                variance = cv2.Laplacian(blur, cv2.CV_64F).var()
                return variance < 50
        except Exception as e:
            print(f"Error in detect_blink: {e}")
            return False
    
    def detect_head_movement(self, current_position, previous_positions):
        """Detect head movement by comparing face positions"""
        if not previous_positions or len(previous_positions) < 5:
            return False
        
        # Calculate movement from average of recent positions
        recent_avg = np.mean(previous_positions[-5:], axis=0)
        distance = np.linalg.norm(np.array(current_position) - recent_avg)
        return distance > self.movement_threshold
    
    def detect_phone_screen(self, face_crop):
        """Enhanced phone screen detection for attendance"""
        try:
            # Convert to different color spaces
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            color_std = np.std(face_crop.reshape(-1, 3), axis=0)
            color_uniformity = np.mean(color_std)
            brightness = np.mean(gray)
            brightness_std = np.std(gray)
            phone_indicators = 0
            if laplacian_var < 40:
                phone_indicators += 1
            if edge_density < 0.08:
                phone_indicators += 1
            if color_uniformity < 12:
                phone_indicators += 1
            if brightness > 200 or brightness_std < 20:
                phone_indicators += 1
            return phone_indicators >= 2
        except Exception as e:
            print(f"Error in detect_phone_screen: {e}")
            return False
    
    def verify_user_liveness(self, user_id, frame, face_landmarks, face_position):
        """Verify liveness for a specific user during attendance"""
        if user_id not in self.user_liveness_data:
            self.user_liveness_data[user_id] = {
                'blink_count': 0,
                'movement_detected': False,
                'frame_count': 0,
                'positions': [],
                'ear_history': [],
                'last_blink_frame': 0,
                'verified': False
            }
        
        data = self.user_liveness_data[user_id]
        data['frame_count'] += 1
        data['positions'].append(face_position)
        
        # Keep only recent positions
        if len(data['positions']) > 10:
            data['positions'].pop(0)
        
        # Detect blink
        if self.detect_blink(frame, face_landmarks):
            if data['frame_count'] - data['last_blink_frame'] > 10:  # Avoid counting same blink
                data['blink_count'] += 1
                data['last_blink_frame'] = data['frame_count']
        
        # Detect movement
        if not data['movement_detected']:
            data['movement_detected'] = self.detect_head_movement(face_position, data['positions'])
        
        # Check if liveness verified
        if (data['blink_count'] >= 1 and 
            data['movement_detected'] and 
            data['frame_count'] >= self.liveness_required_frames):
            data['verified'] = True
        
        return data['verified'], data['blink_count'], data['movement_detected'], data['frame_count']
        
def launch_complete_main_window():
    """Launch the complete main window"""
    root = tk.Tk()
    app = CompleteMainWindow(root)
    root.protocol("WM_DELETE_WINDOW", app.close_application)
    root.mainloop()


if __name__ == "__main__":
    launch_complete_main_window()
