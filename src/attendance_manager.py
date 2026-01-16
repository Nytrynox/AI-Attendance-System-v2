# src/attendance_manager.py

import os
import csv
from datetime import datetime

class AttendanceManager:
    def __init__(self, attendance_dir='data/attendance'):
        self.attendance_dir = attendance_dir
        os.makedirs(self.attendance_dir, exist_ok=True)

    def _get_today_filename(self):
        """Generate filename based on today's date."""
        today = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.attendance_dir, f"{today}.csv")

    def _load_existing_entries(self, filepath):
        """Load existing entries to avoid duplicates."""
        if not os.path.exists(filepath):
            return set()
        with open(filepath, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            return set(row[0] for row in reader if row)

    def mark_attendance(self, user_id, name):
        """
        Log attendance for a given user if not already marked today.
        Params:
            user_id (str): Unique ID of the user
            name (str): Name of the user
        """
        filename = self._get_today_filename()
        existing_ids = self._load_existing_entries(filename)

        if user_id in existing_ids:
            print(f"[INFO] Attendance already marked for {user_id} - {name}")
            return False  # Already marked

        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%Y-%m-%d")

        with open(filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([user_id, name, date_str, time_str])

        print(f"[INFO] Attendance marked for {user_id} - {name}")
        return True
