#!/usr/bin/env python3
"""
Email Notification Manager for Face Attendance System
Sends email alerts when spoof attempts are detected
"""

import os
import smtplib
import logging
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import cv2
import tempfile
import json

# Import location service
try:
    from src.location_service import LocationService
except ImportError:
    try:
        from location_service import LocationService
    except ImportError:
        LocationService = None

logger = logging.getLogger(__name__)


class EmailNotificationManager:
    """Manages email notifications for security events"""
    
    def __init__(self, config_file="email_config.json"):
        """Initialize email notification manager"""
        self.config_file = config_file
        self.config = self.load_or_create_config()
        self.enabled = self.config.get('enabled', False)
        
        # Initialize location service
        try:
            if LocationService is not None:
                self.location_service = LocationService()
                logger.info("Location service initialized for email notifications")
            else:
                self.location_service = None
                logger.warning("Location service not available - emails will not include detailed location")
        except Exception as e:
            logger.warning(f"Failed to initialize location service: {e}")
            self.location_service = None
        
        if self.enabled:
            logger.info("Email notifications enabled")
        else:
            logger.info("Email notifications disabled - configure email_config.json to enable")
    
    def load_or_create_config(self):
        """Load email configuration or create default config"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.info("Email configuration loaded")
                return config
            except Exception as e:
                logger.error(f"Failed to load email config: {e}")
                return self.create_default_config()
        else:
            return self.create_default_config()
    
    def create_default_config(self):
        """Create default email configuration file"""
        default_config = {
            "enabled": False,
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": "your-email@gmail.com",
            "sender_password": "your-app-password",
            "recipient_emails": ["security@company.com", "admin@company.com"],
            "subject_template": "🚨 SECURITY ALERT - Spoof Attempt Detected",
            "use_tls": True,
            "timeout": 30,
            "max_image_size": 1024000,  # 1MB
            "notes": [
                "Set 'enabled' to true to activate email notifications",
                "For Gmail, use an App Password instead of your regular password",
                "Generate App Password: Google Account > Security > App passwords",
                "Make sure to enable 'Less secure app access' if not using App Password"
            ]
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            logger.info(f"Created default email config: {self.config_file}")
            print(f"\n📧 EMAIL CONFIGURATION CREATED")
            print(f"Please edit {self.config_file} with your email settings")
            print(f"Set 'enabled': true to activate email notifications")
        except Exception as e:
            logger.error(f"Failed to create email config: {e}")
        
        return default_config
    
    def validate_config(self):
        """Validate email configuration"""
        required_fields = ['smtp_server', 'smtp_port', 'sender_email', 'sender_password', 'recipient_emails']
        
        for field in required_fields:
            if field not in self.config or not self.config[field]:
                logger.error(f"Missing required email config field: {field}")
                return False
        
        if not isinstance(self.config['recipient_emails'], list) or len(self.config['recipient_emails']) == 0:
            logger.error("recipient_emails must be a non-empty list")
            return False
        
        return True
    
    def save_spoof_image(self, frame, person_description="Unknown"):
        """Save spoof attempt image to temporary file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = tempfile.gettempdir()
            image_path = os.path.join(temp_dir, f"spoof_attempt_{timestamp}.jpg")
            
            # Add timestamp and warning text to image
            annotated_frame = frame.copy()
            height, width = annotated_frame.shape[:2]
            
            # Add red warning border
            cv2.rectangle(annotated_frame, (0, 0), (width-1, height-1), (0, 0, 255), 10)
            
            # Add warning text
            font = cv2.FONT_HERSHEY_SIMPLEX
            texts = [
                "🚨 SECURITY ALERT - SPOOF ATTEMPT DETECTED 🚨",
                f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Person: {person_description}",
                "This person attempted to use fake/spoofed face"
            ]
            
            # Calculate text position
            y_start = 30
            for i, text in enumerate(texts):
                y_pos = y_start + (i * 30)
                # Add black background for text readability
                (text_w, text_h), _ = cv2.getTextSize(text, font, 0.7, 2)
                cv2.rectangle(annotated_frame, (10, y_pos - text_h - 5), 
                             (10 + text_w, y_pos + 5), (0, 0, 0), -1)
                # Add white text
                cv2.putText(annotated_frame, text, (10, y_pos), font, 0.7, (255, 255, 255), 2)
            
            # Resize if image is too large
            max_size = self.config.get('max_image_size', 1024000)  # 1MB
            quality = 95
            
            while True:
                success = cv2.imwrite(image_path, annotated_frame, 
                                    [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not success:
                    logger.error("Failed to save spoof image")
                    return None
                
                file_size = os.path.getsize(image_path)
                if file_size <= max_size or quality <= 30:
                    break
                
                quality -= 10  # Reduce quality to decrease file size
            
            logger.info(f"Spoof attempt image saved: {image_path} (size: {file_size} bytes)")
            return image_path
            
        except Exception as e:
            logger.error(f"Failed to save spoof image: {e}")
            return None
    
    def create_email_content(self, person_description="Unknown", timestamp=None):
        """Create email content for spoof alert with exact live location"""
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get complete location information
        location_data = None
        location_summary = "Face Attendance System (Location not available)"
        location_details = "Location details not available"
        
        if self.location_service:
            try:
                location_data = self.location_service.get_complete_location_info()
                location_summary = self.location_service.get_location_summary()
                location_details = location_data.get('formatted_location', location_details)
            except Exception as e:
                logger.warning(f"Failed to get location info: {e}")
        
        subject = self.config.get('subject_template', 'Security Alert - Spoof Attempt')
        
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .alert {{ background-color: #ffebee; border: 2px solid #f44336; padding: 20px; border-radius: 5px; }}
                .header {{ color: #d32f2f; font-size: 24px; font-weight: bold; margin-bottom: 15px; }}
                .details {{ background-color: #f5f5f5; padding: 15px; border-radius: 3px; margin: 15px 0; }}
                .location {{ background-color: #e3f2fd; border: 2px solid #2196f3; padding: 15px; border-radius: 5px; margin: 15px 0; }}
                .warning {{ color: #ff5722; font-weight: bold; }}
                .footer {{ margin-top: 20px; color: #666; font-size: 12px; }}
                .location-header {{ color: #1976d2; font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
            </style>
        </head>
        <body>
            <div class="alert">
                <div class="header">🚨 SECURITY ALERT - Spoof Attempt Detected 🚨</div>
                
                <p class="warning">A person has attempted to gain unauthorized access using a spoofed/fake face.</p>
                
                <div class="details">
                    <h3>Incident Details:</h3>
                    <ul>
                        <li><strong>Timestamp:</strong> {timestamp}</li>
                        <li><strong>Person Description:</strong> {person_description}</li>
                        <li><strong>Event Type:</strong> Spoof/Fake Face Detection</li>
                        <li><strong>System Response:</strong> Access Denied, Alert Triggered</li>
                    </ul>
                </div>
                
                <div class="location">
                    <div class="location-header">📍 EXACT LIVE LOCATION DETAILS</div>
                    <div style="white-space: pre-line; font-family: monospace; background-color: white; padding: 10px; border-radius: 3px;">
{location_details}
                    </div>
                    <p><strong>Location Summary:</strong> {location_summary}</p>
                </div>
                
                <h3>Action Taken:</h3>
                <ul>
                    <li>✅ Access attempt blocked</li>
                    <li>✅ Security siren activated</li>
                    <li>✅ Incident logged with location</li>
                    <li>✅ Image captured and attached</li>
                    <li>✅ Email alert sent to security team</li>
                    <li>✅ Live location information gathered</li>
                </ul>
                
                <h3>Immediate Response Required:</h3>
                <ul>
                    <li>🔍 Review the attached image</li>
                    <li>📍 Dispatch security to the exact location above</li>
                    <li>📋 Check security logs for additional details</li>
                    <li>🎥 Review camera footage if available</li>
                    <li>⚠️ Investigate if this is a repeated attempt</li>
                    <li>🔒 Consider additional security measures if needed</li>
                </ul>
                
                <div class="footer">
                    <p><strong>This is an automated security alert from the Face Attendance System.</strong></p>
                    <p>Generated at: {timestamp}</p>
                    <p>Location verified at: {timestamp}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        text_body = f"""
🚨 SECURITY ALERT - Spoof Attempt Detected 🚨

A person has attempted to gain unauthorized access using a spoofed/fake face.

INCIDENT DETAILS:
- Timestamp: {timestamp}
- Person Description: {person_description}
- Event Type: Spoof/Fake Face Detection
- System Response: Access Denied, Alert Triggered

📍 EXACT LIVE LOCATION DETAILS:
{location_details}

Location Summary: {location_summary}

ACTION TAKEN:
✅ Access attempt blocked
✅ Security siren activated
✅ Incident logged with location
✅ Image captured and attached
✅ Email alert sent to security team
✅ Live location information gathered

IMMEDIATE RESPONSE REQUIRED:
🔍 Review the attached image
📍 Dispatch security to the exact location above
📋 Check security logs for additional details
🎥 Review camera footage if available
⚠️ Investigate if this is a repeated attempt
🔒 Consider additional security measures if needed

This is an automated security alert from the Face Attendance System.
Generated at: {timestamp}
Location verified at: {timestamp}
        """
        
        return subject, html_body, text_body
    
    def send_spoof_alert_email(self, frame, person_description="Unknown"):
        """Send email alert for spoof attempt with image attachment"""
        def send_email_thread():
            try:
                if not self.enabled:
                    logger.info("Email notifications disabled, skipping spoof alert email")
                    return
                
                if not self.validate_config():
                    logger.error("Invalid email configuration, cannot send alert")
                    return
                
                logger.info(f"Sending spoof alert email for: {person_description}")
                
                # Save spoof image
                image_path = self.save_spoof_image(frame, person_description)
                
                # Create email content
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                subject, html_body, text_body = self.create_email_content(person_description, timestamp)
                
                # Create email message
                msg = MIMEMultipart('alternative')
                msg['From'] = self.config['sender_email']
                msg['To'] = ', '.join(self.config['recipient_emails'])
                msg['Subject'] = subject
                
                # Add text and HTML parts
                text_part = MIMEText(text_body, 'plain', 'utf-8')
                html_part = MIMEText(html_body, 'html', 'utf-8')
                msg.attach(text_part)
                msg.attach(html_part)
                
                # Attach image if available
                if image_path and os.path.exists(image_path):
                    try:
                        with open(image_path, 'rb') as f:
                            img_data = f.read()
                        
                        attachment = MIMEBase('application', 'octet-stream')
                        attachment.set_payload(img_data)
                        encoders.encode_base64(attachment)
                        
                        filename = f"spoof_attempt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        attachment.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {filename}'
                        )
                        msg.attach(attachment)
                        logger.info(f"Image attached to email: {filename}")
                        
                        # Clean up temporary image file
                        os.remove(image_path)
                        
                    except Exception as e:
                        logger.error(f"Failed to attach image: {e}")
                
                # Send email
                with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                    server.set_debuglevel(0)  # Set to 1 for debug output
                    
                    if self.config.get('use_tls', True):
                        server.starttls()
                    
                    server.login(self.config['sender_email'], self.config['sender_password'])
                    server.send_message(msg)
                
                logger.info(f"Spoof alert email sent successfully to {len(self.config['recipient_emails'])} recipients")
                
            except Exception as e:
                logger.error(f"Failed to send spoof alert email: {e}")
                print(f"❌ Failed to send email alert: {e}")
        
        # Send email in background thread to avoid blocking
        email_thread = threading.Thread(target=send_email_thread, daemon=True)
        email_thread.start()
    
    def test_email_notification(self):
        """Test email notification system"""
        if not self.enabled:
            print("❌ Email notifications are disabled")
            print(f"Edit {self.config_file} and set 'enabled': true to test")
            return False
        
        if not self.validate_config():
            print("❌ Email configuration is invalid")
            return False
        
        try:
            print("📧 Testing email notification...")
            
            # Create test image
            test_frame = cv2.imread('test_frame.jpg') if os.path.exists('test_frame.jpg') else None
            if test_frame is None:
                # Create a simple test image
                import numpy as np
                test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.rectangle(test_frame, (100, 100), (540, 380), (0, 0, 255), 3)
                cv2.putText(test_frame, "TEST SPOOF ALERT", (150, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            self.send_spoof_alert_email(test_frame, "Test Person (Email Test)")
            print("✅ Test email queued for sending")
            return True
            
        except Exception as e:
            print(f"❌ Email test failed: {e}")
            return False


if __name__ == "__main__":
    # Test the email notification manager
    import sys
    
    logging.basicConfig(level=logging.INFO)
    email_manager = EmailNotificationManager()
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        email_manager.test_email_notification()
    else:
        print("Email Notification Manager initialized.")
        print("Run with 'test' argument to test email functionality.")
