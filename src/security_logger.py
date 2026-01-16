import os
from datetime import datetime
import cv2

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
SPOOF_LOG = os.path.join(LOG_DIR, 'spoof_attempts.log')
SPOOF_IMG_DIR = os.path.join(LOG_DIR, 'spoof_images')
os.makedirs(SPOOF_IMG_DIR, exist_ok=True)

def log_spoof_attempt(reason, frame=None, user_id=None, context=None):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry = f"[{timestamp}] Reason: {reason}"
    if user_id:
        entry += f", User: {user_id}"
    if context:
        entry += f", Context: {context}"
    entry += '\n'
    with open(SPOOF_LOG, 'a') as f:
        f.write(entry)
    if frame is not None:
        img_name = f"spoof_{timestamp.replace(':','-').replace(' ','_')}.jpg"
        img_path = os.path.join(SPOOF_IMG_DIR, img_name)
        cv2.imwrite(img_path, frame)
        with open(SPOOF_LOG, 'a') as f:
            f.write(f"    Image: {img_path}\n")
