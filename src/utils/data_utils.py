# src/utils/data_utils.py

import os
import pickle
import csv
import cv2
from datetime import datetime

REGISTERED_USERS_DIR = "data/registered_users"
ATTENDANCE_DIR = "data/attendance"

def load_registered_users():
    """
    Loads all registered user encodings from .pkl files.
    Supports both old format (direct .pkl files) and new format (subdirectories).
    Returns:
        List of tuples (user_id, name, encoding)
    """
    users = []
    
    if not os.path.exists(REGISTERED_USERS_DIR):
        return users
        
    # Check for files in subdirectories (new format)
    for item in os.listdir(REGISTERED_USERS_DIR):
        item_path = os.path.join(REGISTERED_USERS_DIR, item)
        
        if os.path.isdir(item_path) and not item.startswith('.'):
            # Look for encoding files in subdirectory
            encoding_loaded = False
            user_id = item
            user_name = None
            user_encoding = None
            
            # First try the new format: {user_id}_encoding.pkl
            for filename in os.listdir(item_path):
                if filename.endswith("_encoding.pkl"):
                    try:
                        # Implement comprehensive numpy compatibility fix
                        import sys
                        import types
                        
                        # Create numpy._core modules if they don't exist
                        numpy_core_modules = [
                            'numpy._core',
                            'numpy._core.multiarray',
                            'numpy._core.umath',
                            'numpy._core.numeric',
                            'numpy._core._multiarray_umath'
                        ]
                        
                        for module_name in numpy_core_modules:
                            if module_name not in sys.modules:
                                # Create a minimal compatible module
                                fake_module = types.ModuleType(module_name)
                                sys.modules[module_name] = fake_module
                                
                                # Add to parent if needed
                                if '.' in module_name:
                                    parent_name, child_name = module_name.rsplit('.', 1)
                                    if parent_name in sys.modules:
                                        setattr(sys.modules[parent_name], child_name, fake_module)
                        
                        # Try loading with numpy compatibility
                        try:
                            with open(os.path.join(item_path, filename), "rb") as f:
                                data = pickle.load(f)
                        except Exception as pickle_error:
                            # If pickle fails, try alternative approach
                            print(f"[INFO] Standard pickle failed for {filename}, trying compatibility mode...")
                            
                            # Try to load and re-encode the face from the image
                            face_image_path = os.path.join(item_path, f"{user_id}_face.jpg")
                            if os.path.exists(face_image_path):
                                import face_recognition
                                import cv2
                                import numpy as np
                                
                                print(f"[INFO] Regenerating encoding from face image for user {user_id}")
                                
                                # Load face image and generate new encoding
                                image = cv2.imread(face_image_path)
                                if image is not None:
                                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    encodings = face_recognition.face_encodings(rgb_image)
                                    
                                    if encodings:
                                        user_encoding = encodings[0]
                                        
                                        # Get user info from user_info.txt
                                        user_info_file = os.path.join(item_path, "user_info.txt")
                                        user_name = f"User_{user_id}"
                                        if os.path.exists(user_info_file):
                                            try:
                                                with open(user_info_file, 'r') as info_f:
                                                    lines = info_f.read().strip().split('\n')
                                                    for line in lines:
                                                        if line.startswith('Name: '):
                                                            user_name = line.replace('Name: ', '')
                                                            break
                                            except Exception:
                                                pass
                                        
                                        # Create new compatible encoding file
                                        new_data = {
                                            'id': user_id,
                                            'name': user_name,
                                            'encoding': user_encoding,
                                            'regenerated': True,
                                            'original_file': filename
                                        }
                                        
                                        # Save the new compatible version
                                        new_filename = f"{user_id}_encoding_compatible.pkl"
                                        new_filepath = os.path.join(item_path, new_filename)
                                        
                                        with open(new_filepath, 'wb') as f:
                                            pickle.dump(new_data, f)
                                        
                                        print(f"[INFO] Created compatible encoding file: {new_filename}")
                                        
                                        # Use the regenerated data
                                        data = new_data
                                    else:
                                        print(f"[WARNING] Could not detect face in image {face_image_path}")
                                        continue
                                else:
                                    print(f"[WARNING] Could not load face image {face_image_path}")
                                    continue
                            else:
                                print(f"[WARNING] No face image found for regeneration: {face_image_path}")
                                continue
                        
                        # Handle both dictionary and raw numpy array formats
                        if isinstance(data, dict):
                            # Dictionary format with metadata
                            user_encoding = data['encoding']
                            user_name = data['name']
                            user_id = data['id']
                            regenerated = data.get('regenerated', False)
                            status = " (regenerated)" if regenerated else ""
                            print(f"[INFO] Loaded user: {user_name} (ID: {user_id}) from {filename}{status}")
                        else:
                            # Raw numpy array format - need to get name from user_info.txt
                            user_encoding = data
                            user_info_file = os.path.join(item_path, "user_info.txt")
                            if os.path.exists(user_info_file):
                                try:
                                    with open(user_info_file, 'r') as info_f:
                                        lines = info_f.read().strip().split('\n')
                                        for line in lines:
                                            if line.startswith('Name: '):
                                                user_name = line.replace('Name: ', '')
                                            elif line.startswith('ID: '):
                                                user_id = line.replace('ID: ', '')
                                except Exception:
                                    pass
                            
                            if user_name is None:
                                user_name = f"User_{user_id}"
                            
                            print(f"[INFO] Loaded user: {user_name} (ID: {user_id}) from {filename} (array format)")
                        
                        users.append((user_id, user_name, user_encoding))
                        encoding_loaded = True
                        break
                        
                    except Exception as e:
                        error_msg = str(e)
                        print(f"[WARNING] Failed to load user data from {filename}: {e}")
                        # Continue to try other files or legacy format
            
            # If new format not found, try legacy format: face_encoding.pkl
            if not encoding_loaded:
                legacy_file = os.path.join(item_path, "face_encoding.pkl")
                if os.path.exists(legacy_file):
                    try:
                        with open(legacy_file, "rb") as f:
                            data = pickle.load(f)
                            
                            # Handle both dictionary and raw numpy array formats
                            if isinstance(data, dict):
                                users.append((data['id'], data['name'], data['encoding']))
                                print(f"[INFO] Loaded user: {data['name']} (ID: {data['id']}) from legacy face_encoding.pkl (dict format)")
                            else:
                                # Raw array - get name from user_info.txt
                                user_encoding = data
                                user_name = f"User_{user_id}"
                                user_info_file = os.path.join(item_path, "user_info.txt")
                                if os.path.exists(user_info_file):
                                    try:
                                        with open(user_info_file, 'r') as info_f:
                                            lines = info_f.read().strip().split('\n')
                                            for line in lines:
                                                if line.startswith('Name: '):
                                                    user_name = line.replace('Name: ', '')
                                    except Exception:
                                        pass
                                
                                users.append((user_id, user_name, user_encoding))
                                print(f"[INFO] Loaded user: {user_name} (ID: {user_id}) from legacy face_encoding.pkl (array format)")
                            
                            encoding_loaded = True
                    except Exception as e:
                        error_msg = str(e)
                        if "numpy._core" in error_msg or "_reconstruct" in error_msg:
                            print(f"[WARNING] Numpy version incompatibility detected for legacy face_encoding.pkl")
                            print(f"[INFO] User in directory {user_id} may need to be re-registered due to numpy version changes")
                        else:
                            print(f"[WARNING] Failed to load legacy user data from face_encoding.pkl: {e}")
            
            if not encoding_loaded:
                print(f"[WARNING] No valid encoding file found in directory: {item}")
        
        elif item.endswith(".pkl"):
            # Legacy format - direct .pkl files in root directory
            try:
                with open(item_path, "rb") as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        users.append((data['id'], data['name'], data['encoding']))
                        print(f"[INFO] Loaded legacy user: {data['name']} (ID: {data['id']}) from root directory")
                    else:
                        print(f"[WARNING] Unexpected data format in {item}")
            except Exception as e:
                print(f"[WARNING] Failed to load legacy user data from {item}: {e}")
    
    print(f"[INFO] Total registered users loaded: {len(users)}")
    
    # Check for incompatible users and provide helpful feedback
    if len(users) == 0:
        incompatible_users = check_and_clean_user_data()
        if incompatible_users:
            print(f"[INFO] Found {len(incompatible_users)} users with incompatible data:")
            for user_id, user_name in incompatible_users:
                print(f"  - {user_name} (ID: {user_id}) needs re-registration")
            print("[INFO] These users will need to register again due to system updates")
            print("[INFO] Use the registration feature in the application to re-register")
    
    return users

def save_attendance(user_id, name):
    """
    Records attendance with timestamp for a given user.
    Attendance is saved in a daily CSV file.
    """
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
    file_path = os.path.join(ATTENDANCE_DIR, f"{date_str}.csv")

    # Check if user already marked
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2 and row[0] == user_id:
                    return  # already marked

    # Write new entry
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([user_id, name, time_str])
        print(f"[INFO] Attendance marked: {name} at {time_str}")

def has_marked_attendance(user_id, date=None):
    """
    Check if a user has already marked attendance for a given date.
    """
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    file_path = os.path.join(ATTENDANCE_DIR, f"{date}.csv")
    if not os.path.exists(file_path):
        return False

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 1 and row[0] == user_id:
                return True
    return False

def ensure_directories(directories):
    """
    Ensure that the specified directories exist, creating them if necessary.
    Args:
        directories (list): List of directory paths to create.
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"[INFO] Directory ensured: {directory}")

def get_datetime_string():
    """
    Get current datetime as a formatted string.
    Returns:
        str: Current datetime in YYYY-MM-DD HH:MM:SS format.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def save_user_data(user_id, name, face_encodings, face_image):
    """
    Save user registration data including face encodings and image.
    
    Args:
        user_id (str): Unique identifier for the user
        name (str): Full name of the user
        face_encodings (list): List of face encodings for the user
        face_image (numpy.ndarray): Face image/crop
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure registered users directory exists
        os.makedirs(REGISTERED_USERS_DIR, exist_ok=True)
        
        # Create user directory
        user_dir = os.path.join(REGISTERED_USERS_DIR, str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save face image
        image_path = os.path.join(user_dir, f"{user_id}_{timestamp}.jpg")
        cv2.imwrite(image_path, face_image)
        
        # Prepare user data
        # Use the first encoding if multiple encodings provided
        primary_encoding = face_encodings[0] if isinstance(face_encodings, list) else face_encodings
        
        user_data = {
            'id': user_id,
            'name': name,
            'encoding': primary_encoding,
            'encodings': face_encodings,  # Store all encodings for better accuracy
            'image_path': image_path,
            'timestamp': timestamp,
            'method': 'multiple_captures'  # Indicate this used multiple face captures
        }
        
        # Save user encoding data
        encoding_file = os.path.join(user_dir, f"{user_id}_encoding.pkl")
        with open(encoding_file, 'wb') as f:
            pickle.dump(user_data, f)
        
        print(f"[INFO] User {name} (ID: {user_id}) registered successfully with {len(face_encodings) if isinstance(face_encodings, list) else 1} encodings")
        
        # Create user info file for compatibility
        user_info_file = os.path.join(user_dir, "user_info.txt")
        with open(user_info_file, 'w') as f:
            f.write(f"Name: {name}\n")
            f.write(f"ID: {user_id}\n")
            f.write(f"Registration Date: {timestamp}\n")
            f.write(f"Liveness Verified: True\n")
            f.write(f"Security Level: Advanced\n")
        
        # Trigger automatic reload notification for all active face recognizers
        trigger_user_reload_notification()
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to save user data for {name} (ID: {user_id}): {str(e)}")
        return False


def trigger_user_reload_notification():
    """
    Create a notification file that signals face recognizers to reload user data.
    This enables real-time user data updates without restarting the application.
    """
    try:
        notification_file = os.path.join(REGISTERED_USERS_DIR, ".reload_trigger")
        with open(notification_file, 'w') as f:
            f.write(f"reload_requested_at_{datetime.now().isoformat()}")
        print("[INFO] User reload notification sent to all active recognizers")
    except Exception as e:
        print(f"[WARNING] Failed to create reload notification: {e}")


def clean_incompatible_user_data():
    """
    Clean up user data that can't be loaded due to compatibility issues.
    This function will backup and remove problematic user directories.
    """
    if not os.path.exists(REGISTERED_USERS_DIR):
        return
    
    print("[INFO] Checking for incompatible user data...")
    
    for item in os.listdir(REGISTERED_USERS_DIR):
        item_path = os.path.join(REGISTERED_USERS_DIR, item)
        
        if os.path.isdir(item_path) and not item.startswith('.'):
            user_id = item
            has_valid_encoding = False
            
            # Check if this directory has any loadable encoding files
            for filename in os.listdir(item_path):
                if filename.endswith("_encoding.pkl"):
                    try:
                        with open(os.path.join(item_path, filename), "rb") as f:
                            pickle.load(f)
                        has_valid_encoding = True
                        break
                    except Exception:
                        continue
            
            if not has_valid_encoding:
                # Create backup directory
                backup_dir = os.path.join(REGISTERED_USERS_DIR, f"_backup_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                
                try:
                    # Move the problematic directory to backup
                    import shutil
                    shutil.move(item_path, backup_dir)
                    print(f"[INFO] Moved incompatible user data {user_id} to backup: {backup_dir}")
                    print(f"[INFO] User {user_id} will need to re-register")
                except Exception as e:
                    print(f"[WARNING] Could not backup user {user_id}: {e}")


def check_and_clean_user_data():
    """
    Check for incompatible user data and offer to clean it up.
    Returns the number of users that need re-registration.
    """
    if not os.path.exists(REGISTERED_USERS_DIR):
        return 0
    
    incompatible_users = []
    
    for item in os.listdir(REGISTERED_USERS_DIR):
        item_path = os.path.join(REGISTERED_USERS_DIR, item)
        
        if os.path.isdir(item_path) and not item.startswith('.'):
            user_id = item
            has_valid_encoding = False
            
            # Check if this directory has any loadable encoding files
            for filename in os.listdir(item_path):
                if filename.endswith("_encoding.pkl"):
                    try:
                        with open(os.path.join(item_path, filename), "rb") as f:
                            pickle.load(f)
                        has_valid_encoding = True
                        break
                    except Exception:
                        continue
            
            if not has_valid_encoding:
                # Try to get user name from user_info.txt
                user_name = user_id
                user_info_file = os.path.join(item_path, "user_info.txt")
                if os.path.exists(user_info_file):
                    try:
                        with open(user_info_file, 'r') as f:
                            lines = f.read().strip().split('\n')
                            for line in lines:
                                if line.startswith('Name: '):
                                    user_name = line.replace('Name: ', '')
                                    break
                    except Exception:
                        pass
                
                incompatible_users.append((user_id, user_name))
    
    return incompatible_users


def check_reload_notification():
    """
    Check if a user reload notification exists and consume it.
    
    Returns:
        bool: True if reload is needed, False otherwise
    """
    try:
        notification_file = os.path.join(REGISTERED_USERS_DIR, ".reload_trigger")
        if os.path.exists(notification_file):
            os.remove(notification_file)  # Consume the notification
            return True
        return False
    except Exception:
        return False
