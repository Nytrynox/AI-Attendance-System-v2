# src/gui/add_user_window_enhanced.py

import tkinter as tk
from tkinter import messagebox
import cv2
import os
import pickle
from src.utils.image_utils import resize_image, preprocess_for_face_recognition
from src.utils.data_utils import save_attendance
from src.face_detector import FaceDetector
from src.anti_spoof import AntiSpoofingDetector
from src.utils.camera_utils import CameraManager
from datetime import datetime
import threading
import time
from PIL import Image, ImageTk


class EnhancedAddUserWindow:
    def __init__(self, master, on_close_callback=None):
        self.master = master
        self.on_close_callback = on_close_callback
        self.master.title("Enhanced Camera Interface")
        self.master.geometry("900x700")
        
        # Register the close callback
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
          # Initialize detection models
        self.face_detector = FaceDetector()
        self.anti_spoof_detector = AntiSpoofingDetector()
        self.camera_manager = CameraManager()
          # Camera variables
        self.cap = None
        self.camera_running = False
        self.current_frame = None
        self.face_captured = False
        self.face_encoding = None
        self.captured_frame = None
        self.user_data = None
        self._current_photo = None  # Store current photo reference
        
        # Create UI
        self.create_ui()
        
    def create_ui(self):
        """Create the enhanced user interface"""
        # Main container
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for user info and controls
        left_panel = tk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Right panel for camera feed
        right_panel = tk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # User information section
        info_frame = tk.LabelFrame(left_panel, text="User Information", padx=10, pady=10)
        info_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(info_frame, text="Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.name_entry = tk.Entry(info_frame, width=25)
        self.name_entry.grid(row=0, column=1, pady=5)
        
        tk.Label(info_frame, text="ID:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.id_entry = tk.Entry(info_frame, width=25)
        self.id_entry.grid(row=1, column=1, pady=5)
        
        # Camera controls section
        control_frame = tk.LabelFrame(left_panel, text="Camera Controls", padx=10, pady=10)
        control_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.start_camera_btn = tk.Button(control_frame, text="Start Camera", 
                                         command=self.start_camera, bg="lightblue")
        self.start_camera_btn.pack(fill=tk.X, pady=5)
        
        self.capture_btn = tk.Button(control_frame, text="Capture Face", 
                                   command=self.capture_face, state=tk.DISABLED, bg="orange")
        self.capture_btn.pack(fill=tk.X, pady=5)
        
        self.save_btn = tk.Button(control_frame, text="Save User", 
                                command=self.save_user, state=tk.DISABLED, bg="lightgreen")
        self.save_btn.pack(fill=tk.X, pady=5)
        
        self.stop_camera_btn = tk.Button(control_frame, text="Stop Camera", 
                                       command=self.stop_camera, state=tk.DISABLED, bg="lightcoral")
        self.stop_camera_btn.pack(fill=tk.X, pady=5)
        
        # Status section
        status_frame = tk.LabelFrame(left_panel, text="Status", padx=10, pady=10)
        status_frame.pack(fill=tk.BOTH, expand=True)
        
        self.status_text = tk.Text(status_frame, wrap=tk.WORD, state=tk.DISABLED)
        status_scrollbar = tk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scrollbar.set)
        
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Camera display section
        camera_frame = tk.LabelFrame(right_panel, text="Live Camera Feed", padx=10, pady=10)
        camera_frame.pack(fill=tk.BOTH, expand=True)
        
        self.camera_label = tk.Label(camera_frame, text="Camera not started", 
                                   bg="black", fg="white", width=60, height=30)
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Legend for color codes
        legend_frame = tk.Frame(right_panel)
        legend_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(legend_frame, text="Legend:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        tk.Label(legend_frame, text="🟢 Real Face", fg="green").pack(side=tk.LEFT, padx=10)
        tk.Label(legend_frame, text="🔴 Spoofing", fg="red").pack(side=tk.LEFT, padx=10)
        tk.Label(legend_frame, text="🟡 Processing", fg="orange").pack(side=tk.LEFT, padx=10)
          # Close button
        close_btn = tk.Button(right_panel, text="Close", command=self.close_window, bg="gray")
        close_btn.pack(pady=5)
        
        # Initialize status
        self.update_status("Ready! Enter user details and start camera.")
        
    def update_status(self, message):
        """Update status display with timestamp"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)
    
    def start_camera(self):
        """Start the camera feed"""
        if self.camera_running:
            return
            
        self.cap = self.camera_manager.initialize_camera()
        if not self.cap:
            messagebox.showerror("Camera Error", 
                               "Cannot access camera! Please check:\n"
                               "1. Camera is connected\n"
                               "2. Camera is not used by another application\n"
                               "3. Camera drivers are installed")
            return
            
        self.camera_running = True
        self.start_camera_btn.config(state=tk.DISABLED)
        self.capture_btn.config(state=tk.NORMAL)
        self.stop_camera_btn.config(state=tk.NORMAL)
        
        self.update_status("Camera started. Real-time face detection active.")
          # Start camera thread
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        
    def camera_loop(self):
        """Main camera processing loop"""
        while self.camera_running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    break
                    
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    continue
                    
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect faces and apply visual feedback
                frame_with_feedback = self.process_frame_with_feedback(frame)
                
                # Convert frame to display format
                self.display_frame(frame_with_feedback)
                
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                print(f"Camera loop error: {e}")
                break
            
    def process_frame_with_feedback(self, frame):
        """Process frame and add visual feedback with colored boxes"""
        frame_copy = frame.copy()

        try:
            # Detect faces
            faces = self.face_detector.detect_faces(frame)

            if faces:
                for face in faces:
                    # Extract face coordinates
                    if isinstance(face, tuple) and len(face) >= 4:
                        if len(face) == 6:  # (x, y, w, h, face_crop, landmarks)
                            x, y, w, h = face[:4]
                            left, top, right, bottom = x, y, x + w, y + h
                            face_crop = face[4] if face[4] is not None else frame[y:y+h, x:x+w]
                        else:  # (left, top, right, bottom)
                            left, top, right, bottom = face
                            face_crop = frame[top:bottom, left:right]
                    elif hasattr(face, 'left'):  # dlib rectangle
                        left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()  # type: ignore
                        face_crop = frame[top:bottom, left:right]
                    else:
                        continue

                    # Ensure face_crop is valid
                    if face_crop is None or face_crop.size == 0:
                        continue

                    # Check for spoofing
                    try:
                        is_real = self.anti_spoof_detector.check_if_real(face_crop)

                        if is_real:
                            # Green box for real face
                            color = (0, 255, 0)
                            label = "Real Face ✓"
                        else:
                            # Red box for spoofing
                            color = (0, 0, 255)
                            label = "Spoofing ✗"
                    except Exception:
                        # Orange box for processing/unknown
                        color = (0, 165, 255)
                        label = "Processing..."

                    # Draw bounding box
                    cv2.rectangle(frame_copy, (left, top), (right, bottom), color, 3)

                    # Draw label
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame_copy, (left, top - label_size[1] - 10), 
                                  (left + label_size[0], top), color, -1)
                    cv2.putText(frame_copy, label, (left, top - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Store current frame for potential capture
                    self.current_frame = frame.copy()
            else:
                # No faces detected
                cv2.putText(frame_copy, "No face detected", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except Exception as e:
            cv2.putText(frame_copy, f"Detection Error: {str(e)[:50]}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return frame_copy
        
    def display_frame(self, frame):
        """Convert and display frame in GUI"""
        try:
            # Resize frame to fit display
            height, width = frame.shape[:2]
            display_width = 640
            display_height = int(height * (display_width / width))
            
            frame_resized = cv2.resize(frame, (display_width, display_height))
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Store reference to prevent garbage collection
            self._current_photo = photo
            
            # Update label in main thread
            self.master.after(0, lambda p=photo: self.update_camera_display(p))
            
        except Exception as e:
            print(f"Display error: {e}")
            
    def update_camera_display(self, photo):
        """Update camera display (called from main thread)"""
        try:
            if self.camera_label.winfo_exists():
                self.camera_label.configure(image=photo)
                setattr(self.camera_label, 'image', photo)  # Keep reference to prevent GC
        except tk.TclError:
            # Widget was destroyed, stop camera
            self.camera_running = False
        
    def capture_face(self):
        """Capture the current face for registration"""
        name = self.name_entry.get().strip()
        user_id = self.id_entry.get().strip()
        
        if not name or not user_id:
            messagebox.showerror("Error", "Please enter both name and ID!")
            return
            
        if self.current_frame is None:
            messagebox.showerror("Error", "No camera frame available!")
            return
            
        try:
            # Process the current frame for face recognition
            faces = self.face_detector.detect_faces(self.current_frame)
            
            if not faces:
                messagebox.showwarning("Warning", "No face detected in current frame!")
                return
                
            # Use the first detected face
            face = faces[0]
            
            # Extract face crop
            if isinstance(face, tuple) and len(face) >= 4:
                if len(face) == 6:  # (x, y, w, h, face_crop, landmarks)
                    face_crop = face[4] if face[4] is not None else self.current_frame[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]
                else:  # (left, top, right, bottom)
                    left, top, right, bottom = face
                    face_crop = self.current_frame[top:bottom, left:right]
            elif hasattr(face, 'left'):  # dlib rectangle
                left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()  # type: ignore
                face_crop = self.current_frame[top:bottom, left:right]
            else:
                messagebox.showerror("Error", "Invalid face format!")
                return
                
            # Check if it's a real face
            is_real = self.anti_spoof_detector.check_if_real(face_crop)
            
            if not is_real:
                messagebox.showwarning("Spoofing Detected", "Spoofing attempt detected! Please show your real face.")
                self.update_status("❌ Spoofing detected - capture rejected")
                return
                
            # Get face encoding for recognition
            face_encodings = preprocess_for_face_recognition(face_crop)
            
            if not face_encodings:
                messagebox.showerror("Error", "Could not extract face features!")
                return
                
            self.face_encoding = face_encodings[0]
            self.captured_frame = self.current_frame.copy()
            self.face_captured = True
            
            self.update_status(f"✓ Face captured successfully for {name}")
            self.save_btn.config(state=tk.NORMAL)
            
            messagebox.showinfo("Success", "Face captured successfully! You can now save the user.")
            
        except Exception as e:
            error_msg = f"Capture failed: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.update_status(f"❌ {error_msg}")
            
    def save_user(self):
        """Save the captured user data"""
        if not self.face_captured:
            messagebox.showerror("Error", "No face captured yet!")
            return
            
        name = self.name_entry.get().strip()
        user_id = self.id_entry.get().strip()
        
        try:
            # Use the centralized save_user_data function to ensure proper reload notifications
            from src.utils.data_utils import save_user_data
            
            success = save_user_data(user_id, name, [self.face_encoding], self.captured_frame)
            
            if success:
                self.user_data = {
                    'id': user_id,
                    'name': name,
                    'encoding': self.face_encoding,
                    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
                }
                
                self.update_status(f"✓ User {name} registered successfully!")
                messagebox.showinfo("Success", f"User {name} has been registered successfully!")
                
                # Trigger callback to refresh main window data
                if self.on_close_callback:
                    self.on_close_callback()
                
                # Reset for next user
                self.reset_for_next_user()
            else:
                raise Exception("Failed to save user data to database")
            
        except Exception as e:
            error_msg = f"Failed to save user: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.update_status(f"❌ {error_msg}")
            
    def reset_for_next_user(self):
        """Reset the interface for registering another user"""
        self.name_entry.delete(0, tk.END)
        self.id_entry.delete(0, tk.END)
        self.face_captured = False
        self.face_encoding = None
        self.captured_frame = None
        self.user_data = None
        self.save_btn.config(state=tk.DISABLED)
        self.update_status("Ready for next user registration.")
        
    def stop_camera(self):
        """Stop the camera feed"""
        self.camera_running = False
        
        if self.cap:
            self.camera_manager.release_camera(self.cap)
            self.cap = None
            
        self.start_camera_btn.config(state=tk.NORMAL)
        self.capture_btn.config(state=tk.DISABLED)
        self.stop_camera_btn.config(state=tk.DISABLED)
        
        self.camera_label.configure(image="", text="Camera stopped")
        setattr(self.camera_label, 'image', None)  # Clear reference
        
        self.update_status("Camera stopped.")
        
    def close_window(self):
        """Close the window and cleanup"""
        self.stop_camera()
        self.master.quit()
        self.master.destroy()
        
    def on_closing(self):
        """Handle window closing event"""
        if self.on_close_callback:
            self.on_close_callback()
        self.close_window()


def launch_enhanced_add_user_window():
    """Launch the enhanced add user window"""
    root = tk.Tk()
    app = EnhancedAddUserWindow(root)
    root.protocol("WM_DELETE_WINDOW", app.close_window)
    root.mainloop()


if __name__ == "__main__":
    launch_enhanced_add_user_window()
