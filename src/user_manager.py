# src/user_manager.py

import os
import cv2
import face_recognition
import pickle
import uuid

class UserManager:
    def __init__(self,
                 training_data_dir='data/training_data',
                 registered_dir='data/registered_users'):
        self.training_data_dir = training_data_dir
        self.registered_dir = registered_dir

        os.makedirs(self.training_data_dir, exist_ok=True)
        os.makedirs(self.registered_dir, exist_ok=True)

    def register_new_user(self):
        user_id = input("Enter User ID: ").strip()
        name = input("Enter Name: ").strip()
        user_dir = os.path.join(self.training_data_dir, name)
        os.makedirs(user_dir, exist_ok=True)

        print("[INFO] Starting webcam for face capture...")

        cap = cv2.VideoCapture(0)
        captured = False

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb_frame)

            # Initialize variables for capturing
            top = bottom = left = right = None
            img_path = None
            
            for (top, right, bottom, left) in boxes:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.imshow("Register Face - Press 'c' to capture, 'q' to quit", frame)

            key = cv2.waitKey(1)
            if key == ord('c') and boxes and top is not None:
                face_img = rgb_frame[top:bottom, left:right]
                img_path = os.path.join(user_dir, f"{user_id}_{uuid.uuid4().hex[:6]}.jpg")
                cv2.imwrite(img_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                print(f"[INFO] Image saved: {img_path}")
                captured = True
                break
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if captured:
            self._generate_encoding(user_id, name, img_path)

    def _generate_encoding(self, user_id, name, image_path):
        print("[INFO] Encoding face...")
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if not encodings:
            print("[ERROR] No face found in the captured image.")
            return

        encoding = encodings[0]
        data = {
            'id': user_id,
            'name': name,
            'encoding': encoding
        }

        # Save encoding file in user-specific subdirectory as <user_id>_encoding.pkl
        user_dir = os.path.join(self.registered_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
        encoding_file = os.path.join(user_dir, f"{user_id}_encoding.pkl")
        with open(encoding_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"[SUCCESS] Registered and saved encoding as: {encoding_file}")
        
        # Trigger reload notification for real-time updates
        try:
            from src.utils.data_utils import trigger_user_reload_notification
            trigger_user_reload_notification()
            print("[INFO] Triggered user data reload notification")
        except Exception as e:
            print(f"[WARNING] Could not trigger reload notification: {e}")
