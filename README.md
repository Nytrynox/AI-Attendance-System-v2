# AI Attendance Management System v2.0

## Overview
The second iteration of the AI Attendance System, featuring enhanced accuracy, real-time cloud syncing, and a dedicated mobile app for students and teachers. Optimized for large-scale institutional use.

## Features
-   **Live Recognition**: Real-time processing at high frame rates.
-   **Anti-Spoofing**: Liveness detection to prevent photo-based bypasses.
-   **Cloud Sync**: Centralized database accessible from multiple classrooms.
-   **Mobile App**: Students can view their attendance stats; teachers can manage overrides.

## Technology Stack
-   **Vision**: OpenCV, DeepFace.
-   **Backend**: Python Flask / FastAPI.
-   **Database**: MongoDB Atlas.
-   **Mobile**: Flutter / React Native.

## Usage Flow
1.  **Setup**: Install cameras in classrooms connected to the central server.
2.  **Process**: Continuously monitor and log student presence.
3.  **Sync**: Push data to the cloud dashboard.
4.  **Access**: Students check their app for real-time updates.

## Quick Start
```bash
# Clone the repository
git clone "https://github.com/Nytrynox/AI-Attendance-System-2.0.git"

# Cloud Setup
# Configure MongoDB connection string in .env

# Run Server
python app.py
```

## License
MIT License

## Author
**Karthik Idikuda**
