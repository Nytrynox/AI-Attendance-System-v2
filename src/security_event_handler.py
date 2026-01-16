#!/usr/bin/env python3
"""
Security Event Handler for Face Attendance System
Coordinates sound alerts, email notifications, and logging for security events
"""

import os
import logging
import threading
import time
from datetime import datetime
import cv2

from src.sound_manager import SoundManager
from src.email_notification import EmailNotificationManager

logger = logging.getLogger(__name__)


class SecurityEventHandler:
    """Handles security events with appropriate alerts and notifications"""
    
    def __init__(self):
        """Initialize security event handler"""
        self.sound_manager = SoundManager()
        self.email_manager = EmailNotificationManager()
        
        # Event tracking to prevent spam
        self.last_events = {
            'success': 0,
            'unknown': 0,
            'spoof': 0,
            'siren': 0
        }
        
        # Minimum time between same event types (seconds)
        self.event_cooldowns = {
            'success': 1.0,      # Allow success sounds more frequently
            'unknown': 2.0,      # Moderate cooldown for unknown users
            'spoof': 3.0,        # Longer cooldown for spoof alerts
            'siren': 5.0         # Longest cooldown for siren (security alert)
        }
        
        # Spoof attempt tracking
        self.spoof_attempts = []
        self.max_spoof_history = 100
        
        logger.info("Security Event Handler initialized")
    
    def can_trigger_event(self, event_type):
        """Check if enough time has passed to trigger this event type"""
        current_time = time.time()
        last_time = self.last_events.get(event_type, 0)
        cooldown = self.event_cooldowns.get(event_type, 2.0)
        
        return (current_time - last_time) >= cooldown
    
    def update_last_event(self, event_type):
        """Update the last event time for cooldown tracking"""
        self.last_events[event_type] = int(time.time())
    
    def handle_successful_attendance(self, user_id, user_name, confidence=None):
        """Handle successful attendance marking"""
        if not self.can_trigger_event('success'):
            logger.debug("Success event in cooldown, skipping sound")
            return
        
        try:
            # Play success sound
            self.sound_manager.play_success_sound()
            self.update_last_event('success')
            
            # Log the successful attendance
            confidence_str = f" (confidence: {confidence:.2f})" if confidence else ""
            logger.info(f"✅ Successful attendance: {user_name} ({user_id}){confidence_str}")
            
            print(f"✅ Welcome, {user_name}! Attendance recorded successfully.")
            
        except Exception as e:
            logger.error(f"Error handling successful attendance: {e}")
    
    def handle_unknown_user(self, confidence=None):
        """Handle unknown user detection"""
        if not self.can_trigger_event('unknown'):
            logger.debug("Unknown user event in cooldown, skipping sound")
            return
        
        try:
            # Play unknown user sound
            self.sound_manager.play_unknown_sound()
            self.update_last_event('unknown')
            
            # Log the unknown user
            confidence_str = f" (confidence: {confidence:.2f})" if confidence else ""
            logger.info(f"❓ Unknown user detected{confidence_str}")
            
            print("❓ Unknown person detected. Please register first.")
            
        except Exception as e:
            logger.error(f"Error handling unknown user: {e}")
    
    def handle_spoof_attempt(self, frame, person_description="Unknown Person", 
                           prediction_score=None, spoof_type="Generic"):
        """Enhanced handler for spoof/fake face detection with image capture"""
        if not self.can_trigger_event('spoof'):
            logger.debug("Spoof event in cooldown, skipping alerts")
            return
        
        try:
            # Record spoof attempt
            spoof_record = {
                'timestamp': datetime.now(),
                'person_description': person_description,
                'prediction_score': prediction_score,
                'spoof_type': spoof_type
            }
            self.spoof_attempts.append(spoof_record)
            
            # Keep only recent attempts
            if len(self.spoof_attempts) > self.max_spoof_history:
                self.spoof_attempts.pop(0)
            
            # Play spoof warning sound WITH image capture
            self.sound_manager.play_spoof_sound(frame)
            self.update_last_event('spoof')
            
            # Log the spoof attempt
            score_str = f" (score: {prediction_score:.2f})" if prediction_score else ""
            logger.warning(f"🚫 SPOOF ATTEMPT: {person_description} - {spoof_type}{score_str}")
            
            # Send email alert with captured image (enhanced with better description)
            if frame is not None:
                email_description = f"{person_description}"
                details = []
                if spoof_type != "Generic":
                    details.append(f"Type: {spoof_type}")
                if prediction_score is not None:
                    details.append(f"Confidence: {prediction_score:.2f}")
                
                if details:
                    email_description += f" ({', '.join(details)})"
                
                # Send email with detailed information
                self.email_manager.send_spoof_alert_email(frame, email_description)
            
            # Enhanced console output
            print(f"🚫 SPOOF DETECTED: {person_description}")
            print(f"   Type: {spoof_type}")
            print(f"   Access denied! Security alert triggered.")
            if prediction_score:
                print(f"   Detection confidence: {prediction_score:.2f}")
            
        except Exception as e:
            logger.error(f"Error handling spoof attempt: {e}")
    
    def handle_security_alert(self, frame, person_description="Unknown Person", 
                            alert_reason="Multiple spoof attempts"):
        """Enhanced handler for high-priority security alert (plays siren and sends urgent email)"""
        if not self.can_trigger_event('siren'):
            logger.debug("Security alert in cooldown, skipping siren")
            return
        
        try:
            # Play security siren WITH image capture
            self.sound_manager.play_siren_sound(frame)
            self.update_last_event('siren')
            
            # Log security alert
            logger.critical(f"🚨 SECURITY ALERT: {person_description} - {alert_reason}")
            
            # Send urgent email alert with enhanced description
            if frame is not None:
                urgent_description = f"🚨 URGENT - {person_description} (Reason: {alert_reason})"
                self.email_manager.send_spoof_alert_email(frame, urgent_description)
            
            # Enhanced console output
            print(f"🚨 SECURITY ALERT: {alert_reason}")
            print(f"   Person: {person_description}")
            print(f"   🔊 Siren activated! 📧 Email alert sent!")
            print(f"   📸 Security image captured and saved")
            
        except Exception as e:
            logger.error(f"Error handling security alert: {e}")
    
    def handle_comprehensive_spoof_detection(self, frame, person_description="Unknown Person",
                                           prediction_score=None, spoof_type="Generic"):
        """Enhanced comprehensive handler with advanced spoof attempt tracking"""
        try:
            # Handle the spoof attempt with image capture
            self.handle_spoof_attempt(frame, person_description, prediction_score, spoof_type)
            
            # Check for repeated attempts and escalate if necessary
            if self.check_for_repeated_attempts(person_description):
                alert_reason = f"Multiple spoof attempts detected"
                if person_description != "Unknown Person":
                    alert_reason = f"Repeated spoof attempts from {person_description}"
                
                # Wait a moment before triggering siren to allow spoof sound to finish
                def delayed_security_alert():
                    time.sleep(2.5)  # Wait for spoof sound to complete
                    self.handle_security_alert(frame, person_description, alert_reason)
                
                # Trigger delayed security alert
                import threading
                alert_thread = threading.Thread(target=delayed_security_alert, daemon=True)
                alert_thread.start()
                
                # Enhanced console warning
                print(f"⚠️  ESCALATION: {alert_reason}")
                print(f"   🚨 Triggering security siren in 2.5 seconds...")
            
        except Exception as e:
            logger.error(f"Error in comprehensive spoof detection: {e}")
    
    def check_for_repeated_attempts(self, person_description="Unknown Person", time_window=300):
        """Check if there have been repeated spoof attempts and trigger security alert"""
        current_time = datetime.now()
        
        # Count recent spoof attempts
        recent_attempts = [
            attempt for attempt in self.spoof_attempts
            if (current_time - attempt['timestamp']).total_seconds() <= time_window
        ]
        
        # Check for multiple attempts from same person or in general
        person_attempts = [
            attempt for attempt in recent_attempts
            if person_description.lower() in attempt['person_description'].lower()
        ]
        
        # Trigger security alert for repeated attempts
        if len(person_attempts) >= 2:  # 2 or more attempts from same person
            return True
        elif len(recent_attempts) >= 3:  # 3 or more total attempts in time window
            return True
        
        return False
    
    def get_spoof_attempt_summary(self, time_window=3600):  # Default: last hour
        """Get summary of recent spoof attempts"""
        current_time = datetime.now()
        
        recent_attempts = [
            attempt for attempt in self.spoof_attempts
            if (current_time - attempt['timestamp']).total_seconds() <= time_window
        ]
        
        return {
            'total_attempts': len(recent_attempts),
            'unique_persons': len(set(attempt['person_description'] for attempt in recent_attempts)),
            'attempts': recent_attempts
        }
    
    def test_all_alerts(self):
        """Test all alert types"""
        print("🧪 Testing Security Event Handler...")
        
        # Create test frame
        import numpy as np
        test_frame = cv2.rectangle(
            np.zeros((480, 640, 3), dtype=np.uint8),  # type: ignore
            (100, 100), (540, 380), (0, 255, 0), 3
        )
        cv2.putText(test_frame, "TEST ALERT", (200, 250),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Test sequence
        tests = [
            ("Success Alert", lambda: self.handle_successful_attendance("TEST001", "Test User", 0.95)),
            ("Unknown User Alert", lambda: self.handle_unknown_user(0.45)),
            ("Spoof Alert", lambda: self.handle_spoof_attempt(test_frame, "Test Spoofer", 0.2, "Phone Screen")),
            ("Security Alert (Siren)", lambda: self.handle_security_alert(test_frame, "Repeat Offender", "Testing siren"))
        ]
        
        for test_name, test_func in tests:
            print(f"\n🔊 {test_name}...")
            test_func()
            time.sleep(4)  # Wait between tests
        
        print("\n✅ All alert tests completed!")


if __name__ == "__main__":
    # Test the security event handler
    import sys
    
    logging.basicConfig(level=logging.INFO)
    handler = SecurityEventHandler()
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        handler.test_all_alerts()
    else:
        print("Security Event Handler initialized.")
        print("Run with 'test' argument to test all alerts.")
