
import cv2
import numpy as np
import time
import os
import pickle
import face_recognition
from datetime import datetime

class EnhancedUserRegistration:
    def __init__(self, face_detector, liveness_detector, user_manager, sound_manager=None, email_manager=None):
        self.face_detector = face_detector
        self.liveness_detector = liveness_detector
        self.user_manager = user_manager
        self.sound_manager = sound_manager
        self.email_manager = email_manager
        
        # Background analysis parameters
        self.background_analysis_time = 5.0  # 5 seconds
        self.analysis_frame_count = 0
        self.min_analysis_frames = 30  # Minimum frames for analysis
        
        # Analysis states
        self.ANALYSIS_PHASE = "analysis"
        self.REGISTRATION_PHASE = "registration"
        self.COMPLETE_PHASE = "complete"
        
        # Current state
        self.current_phase = self.ANALYSIS_PHASE
        self.analysis_start_time = None
        self.background_data = {
            'face_positions': [],
            'blink_patterns': [],
            'head_movements': [],
            'texture_scores': [],
            'movement_baseline': None,
            'natural_patterns_detected': False
        }
        
        print("[INFO] 🔧 Enhanced User Registration initialized")
        print("[INFO] 📊 Background analysis: 5 seconds")
    
    def start_background_analysis(self, user_id):
        """Start the 5-second background analysis period"""
        print(f"[INFO] 🔍 Starting background analysis for {user_id}")
        print("[INFO] ⏳ Please look at camera naturally for 5 seconds...")
        
        self.current_phase = self.ANALYSIS_PHASE
        self.analysis_start_time = time.time()
        self.analysis_frame_count = 0
        self.background_data = {
            'face_positions': [],
            'blink_patterns': [],
            'head_movements': [],
            'texture_scores': [],
            'movement_baseline': None,
            'natural_patterns_detected': False
        }
        
        # Initialize tracking in liveness detector
        if user_id not in self.liveness_detector.user_data:
            self.liveness_detector.initialize_user_tracking(user_id)
    
    def analyze_background_frame(self, frame, face_bbox, face_landmarks, user_id):
        """
        Analyze a single frame during background analysis phase
        Returns: (analysis_complete, status_message, progress_percentage)
        """
        if self.analysis_start_time is None:
            self.start_background_analysis(user_id)
        
        current_time = time.time()
        elapsed_time = current_time - (self.analysis_start_time or current_time)
        progress = min(100, (elapsed_time / self.background_analysis_time) * 100)
        self.analysis_frame_count += 1
        
        # Collect background data
        if face_bbox:
            x, y, w, h = face_bbox[:4]
            face_center = (x + w//2, y + h//2)
            self.background_data['face_positions'].append(face_center)
            
            # Analyze face texture (for baseline)
            try:
                face_crop = frame[y:y+h, x:x+w]
                gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                texture_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                self.background_data['texture_scores'].append(texture_score)
            except:
                pass
            
            # Collect blink data if landmarks available
            if face_landmarks and len(face_landmarks) >= 68:
                # Calculate EAR for blink detection
                try:
                    left_eye = np.array(face_landmarks[36:42])
                    right_eye = np.array(face_landmarks[42:48])
                    
                    def calculate_ear(eye):
                        A = np.linalg.norm(eye[1] - eye[5])
                        B = np.linalg.norm(eye[2] - eye[4])
                        C = np.linalg.norm(eye[0] - eye[3])
                        return (A + B) / (2.0 * C)
                    
                    left_ear = calculate_ear(left_eye)
                    right_ear = calculate_ear(right_eye)
                    avg_ear = (left_ear + right_ear) / 2.0
                    
                    self.background_data['blink_patterns'].append(avg_ear)
                except:
                    pass
            
            # Analyze head movement patterns
            if len(self.background_data['face_positions']) >= 5:
                recent_positions = self.background_data['face_positions'][-5:]
                movement_variance = np.var(recent_positions, axis=0)
                movement_magnitude = np.sqrt(np.sum(movement_variance))
                self.background_data['head_movements'].append(movement_magnitude)
        
        # Check if analysis is complete
        analysis_complete = (elapsed_time >= self.background_analysis_time and 
                           self.analysis_frame_count >= self.min_analysis_frames)
        
        if analysis_complete:
            self._finalize_background_analysis(user_id)
            status_message = "✅ Background analysis complete - Starting registration"
            return True, status_message, 100
        
        # Create status message
        time_remaining = max(0, self.background_analysis_time - elapsed_time)
        status_message = f"🔍 Analyzing natural patterns... {time_remaining:.1f}s remaining"
        
        return False, status_message, progress
    
    def _finalize_background_analysis(self, user_id):
        """Finalize the background analysis and prepare for registration"""
        print(f"[INFO] ✅ Background analysis complete for {user_id}")
        
        # Calculate movement baseline
        if self.background_data['head_movements']:
            self.background_data['movement_baseline'] = {
                'mean': np.mean(self.background_data['head_movements']),
                'std': np.std(self.background_data['head_movements']),
                'max': np.max(self.background_data['head_movements'])
            }
        
        # Analyze blink patterns
        if self.background_data['blink_patterns']:
            blink_variance = np.var(self.background_data['blink_patterns'])
            natural_blink_detected = blink_variance > 0.001  # Some variation expected
            self.background_data['natural_patterns_detected'] = natural_blink_detected
        
        # Update liveness detector with baseline data
        if user_id in self.liveness_detector.user_data:
            user_data = self.liveness_detector.user_data[user_id]
            
            # Adjust thresholds based on background analysis
            if self.background_data['movement_baseline']:
                baseline = self.background_data['movement_baseline']
                # Set more lenient thresholds based on natural movement
                user_data['movement_baseline'] = baseline
                print(f"[INFO] 📊 Movement baseline: {baseline['mean']:.2f} ± {baseline['std']:.2f}")
            
            # Mark as ready for registration
            user_data['background_analysis_complete'] = True
            user_data['registration_ready'] = True
        
        # Transition to registration phase
        self.current_phase = self.REGISTRATION_PHASE
        print("[INFO] 🎯 Ready for liveness verification and registration!")
    
    def register_user_with_analysis(self, frame, user_name, user_id):
        """
        Main registration function with background analysis
        Returns: (registration_complete, success, status_message, progress)
        """
        faces = self.face_detector.detect_faces(frame)
        
        if not faces:
            return False, False, "❌ No face detected - Please face the camera", 0
        
        # Get face data
        face_data = faces[0]
        x, y, w, h = face_data[:4]
        face_landmarks = face_data[5] if len(face_data) > 5 else None
        face_bbox = (x, y, w, h)
        
        # Phase 1: Background Analysis
        if self.current_phase == self.ANALYSIS_PHASE:
            analysis_complete, status_msg, progress = self.analyze_background_frame(
                frame, face_bbox, face_landmarks, user_id
            )
            
            if analysis_complete:
                self.current_phase = self.REGISTRATION_PHASE
                return False, False, "🎯 Starting registration verification...", 100
            
            return False, False, status_msg, progress
        
        # Phase 2: Registration with Liveness Verification
        elif self.current_phase == self.REGISTRATION_PHASE:
            # Use the enhanced liveness detector with background-adjusted thresholds
            is_live, verification_complete, status_message, progress = self.liveness_detector.verify_liveness_comprehensive(
                frame, face_bbox, face_landmarks, user_id
            )
            
            if verification_complete:
                if is_live:
                    # Liveness verified - proceed with registration
                    success = self._save_user_registration(frame, face_bbox, user_name, user_id)
                    if success:
                        self.current_phase = self.COMPLETE_PHASE
                        return True, True, f"✅ Registration successful for {user_name}!", 100
                    else:
                        return True, False, "❌ Failed to save user registration", 100
                else:
                    # Liveness verification failed
                    return True, False, "❌ Liveness verification failed - Please try again", 100
            
            # Still verifying
            return False, False, status_message, progress
        
        # Phase 3: Complete
        else:
            return True, True, "✅ Registration completed", 100
    
    def _save_user_registration(self, frame, face_bbox, user_name, user_id):
        """Save the user registration data"""
        try:
            x, y, w, h = face_bbox
            face_crop = frame[y:y+h, x:x+w]
            
            # Create user directory
            user_dir = f"data/registered_users/{user_id}"
            os.makedirs(user_dir, exist_ok=True)
            
            # Save face image
            face_image_path = os.path.join(user_dir, f"{user_id}_face.jpg")
            cv2.imwrite(face_image_path, face_crop)
            
            # Generate face encoding
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_face)
            
            if encodings:
                # Save encoding
                encoding_path = os.path.join(user_dir, f"{user_id}_encoding.pkl")
                with open(encoding_path, 'wb') as f:
                    pickle.dump({
                        'user_id': user_id,
                        'user_name': user_name,
                        'encoding': encodings[0],
                        'registration_date': datetime.now().isoformat(),
                        'background_analysis': self.background_data
                    }, f)
                
                print(f"[INFO] ✅ User {user_name} registered successfully!")
                print(f"[INFO] 📁 Data saved to {user_dir}")
                
                # Add to user manager
                self.user_manager.add_user(user_id, user_name, encodings[0])
                
                return True
            else:
                print("[ERROR] ❌ Failed to generate face encoding")
                return False
                
        except Exception as e:
            print(f"[ERROR] Registration failed: {e}")
            return False
    
    def reset_registration(self, user_id):
        """Reset the registration process"""
        self.current_phase = self.ANALYSIS_PHASE
        self.analysis_start_time = None
        self.analysis_frame_count = 0
        self.background_data = {
            'face_positions': [],
            'blink_patterns': [],
            'head_movements': [],
            'texture_scores': [],
            'movement_baseline': None,
            'natural_patterns_detected': False
        }
        
        # Reset liveness detector data
        if user_id in self.liveness_detector.user_data:
            del self.liveness_detector.user_data[user_id]
        
        print(f"[INFO] 🔄 Registration reset for {user_id}")
    
    def get_registration_status(self):
        """Get current registration status"""
        return {
            'phase': self.current_phase,
            'analysis_frames': self.analysis_frame_count,
            'background_data_collected': len(self.background_data['face_positions']),
            'natural_patterns_detected': self.background_data['natural_patterns_detected']
        }
    
    def draw_registration_ui(self, frame):
        """Draw registration UI elements on frame"""
        if self.current_phase == self.ANALYSIS_PHASE:
            # Background analysis UI
            cv2.putText(frame, "🔍 BACKGROUND ANALYSIS", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, "Look naturally at camera", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            if self.analysis_start_time:
                elapsed = time.time() - self.analysis_start_time
                remaining = max(0, self.background_analysis_time - elapsed)
                cv2.putText(frame, f"Time: {remaining:.1f}s", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                
                # Progress bar
                progress = min(1.0, elapsed / self.background_analysis_time)
                bar_width = 200
                bar_height = 10
                cv2.rectangle(frame, (10, 100), (10 + bar_width, 110 + bar_height), (100, 100, 100), -1)
                cv2.rectangle(frame, (10, 100), (10 + int(progress * bar_width), 110 + bar_height), (0, 255, 255), -1)
        
        elif self.current_phase == self.REGISTRATION_PHASE:
            cv2.putText(frame, "🎯 REGISTRATION MODE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Performing liveness verification", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        elif self.current_phase == self.COMPLETE_PHASE:
            cv2.putText(frame, "✅ REGISTRATION COMPLETE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame

    def register_user_with_camera(self, camera):
        """
        Complete camera-based registration workflow with background analysis
        
        Args:
            camera: OpenCV VideoCapture object
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        import tkinter as tk
        from tkinter import simpledialog, messagebox
        
        print("[INFO] 🚀 Starting enhanced registration with camera")
        
        # Get user information
        try:
            # Create a temporary root window for dialogs
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            
            user_name = simpledialog.askstring(
                "User Registration", 
                "Enter full name for registration:",
                parent=root
            )
            
            if not user_name or not user_name.strip():
                print("[INFO] ❌ Registration cancelled - no name provided")
                root.destroy()
                return False
            
            user_name = user_name.strip()
            
            # Generate user ID (timestamp-based)
            import time
            user_id = str(int(time.time() * 1000) % 1000000)  # Last 6 digits of timestamp
            
            print(f"[INFO] 📝 Registering user: {user_name} (ID: {user_id})")
            
            root.destroy()
            
        except Exception as e:
            print(f"[ERROR] Failed to get user input: {e}")
            return False
        
        # Start background analysis
        self.start_background_analysis(user_id)
        print(f"[INFO] 🔍 Starting {self.background_analysis_time}-second background analysis...")
        
        registration_complete = False
        analysis_start_time = time.time()
        
        try:
            while not registration_complete:
                ret, frame = camera.read()
                if not ret:
                    print("[ERROR] ❌ Failed to capture frame from camera")
                    break
                
                # Calculate elapsed time for timeout check
                current_time = time.time()
                elapsed_time = current_time - analysis_start_time
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect faces
                faces = self.face_detector.detect_faces(frame)
                
                if faces:
                    face = faces[0]  # Use first detected face
                    x, y, w, h, face_crop, landmarks = face
                    face_bbox = (x, y, w, h)
                    
                    current_time = time.time()
                    elapsed_time = current_time - analysis_start_time
                    
                    if self.current_phase == self.ANALYSIS_PHASE:
                        # Background analysis phase
                        self.analyze_background_frame(frame, face_bbox, landmarks, user_id)
                        
                        # Check if analysis is complete
                        if elapsed_time >= self.background_analysis_time:
                            print(f"[INFO] ✅ Background analysis complete ({elapsed_time:.1f}s)")
                            if self.sound_manager:
                                # Play a subtle sound to indicate transition
                                try:
                                    self.sound_manager.play_unknown_sound()  # Neutral transition sound
                                except:
                                    pass
                            self.current_phase = self.REGISTRATION_PHASE
                    
                    elif self.current_phase == self.REGISTRATION_PHASE:
                        # Registration verification phase
                        try:
                            success = self.register_user_with_analysis(frame, user_name, user_id)
                            if success:
                                print(f"[INFO] 🎉 Registration completed successfully!")
                                if self.sound_manager:
                                    self.sound_manager.play_success_sound()
                                registration_complete = True
                            
                        except Exception as e:
                            print(f"[ERROR] Registration verification failed: {e}")
                            if self.sound_manager:
                                self.sound_manager.play_spoof_sound()
                            break
                    
                    # Draw UI feedback
                    frame = self.draw_registration_ui(frame)
                    
                    # Show progress for analysis phase
                    if self.current_phase == self.ANALYSIS_PHASE:
                        progress = min(elapsed_time / self.background_analysis_time, 1.0)
                        progress_width = int(400 * progress)
                        cv2.rectangle(frame, (10, 100), (410, 120), (100, 100, 100), 2)
                        cv2.rectangle(frame, (12, 102), (12 + progress_width, 118), (0, 255, 0), -1)
                        cv2.putText(frame, f"Analysis: {elapsed_time:.1f}s / {self.background_analysis_time:.1f}s", 
                                   (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                else:
                    # No face detected
                    cv2.putText(frame, "👤 Please position your face in the camera", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "🎯 Make sure your face is clearly visible", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                
                # Display frame
                cv2.imshow('Enhanced Registration - Press ESC to Cancel', frame)
                
                # Check for user cancellation
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    print("[INFO] ❌ Registration cancelled by user")
                    break
                
                # Timeout protection (max 60 seconds total)
                if elapsed_time > 60:
                    print("[INFO] ⏰ Registration timeout (60 seconds exceeded)")
                    break
            
            cv2.destroyAllWindows()
            
            if registration_complete:
                print(f"[INFO] ✅ User {user_name} (ID: {user_id}) registered successfully!")
                return True
            else:
                print(f"[INFO] ❌ Registration failed or cancelled for {user_name}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Registration process error: {e}")
            cv2.destroyAllWindows()
            return False
        
        finally:
            # Reset registration state
            self.reset_registration(user_id)
            cv2.destroyAllWindows()
