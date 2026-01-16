#!/usr/bin/env python3
"""
Face Recognizer Module
=====================

This module provides face recognition functionality using deep learning models
and face encoding techniques. It supports user registration, face matching,
and confidence scoring for the facial attendance system.
"""

import os
import logging
import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any
import face_recognition
from datetime import datetime

# Setup module logger
logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    Face recognition system using face_recognition library and deep learning models.
    
    This class handles:
    - Face encoding extraction
    - User face database management
    - Face matching and recognition
    - Confidence scoring
    """
    
    def __init__(self, model_path: Optional[str] = None, database_path: str = "data/registered_users"):
        """
        Initialize the face recognizer.
        
        Args:
            model_path (str, optional): Path to face recognition model (for future use)
            database_path (str): Path to user database directory
        """
        self.model_path = model_path
        self.database_path = database_path
        self.model = None
        
        # Face database - stores user_id -> encoding mappings
        self.known_face_encodings = []
        self.known_ids = []
        self.known_names = []
        
        # Face recognition settings
        self.recognition_threshold = 0.6  # Lower is stricter
        self.face_locations_model = "hog"  # "hog" or "cnn"
        
        # Auto-reload tracking
        self.last_reload_time = 0
        self.database_mod_time = 0
        
        # Initialize model and database
        if model_path and os.path.exists(model_path):
            self._load_model()
        
        # Note: User data loading is deferred to avoid blocking during initialization
        # Call reload_user_data() explicitly when needed
        
        logger.info(f"Face recognizer initialized with model path: {model_path}")
        # Load existing registered user data for recognition
        self.reload_user_data()
    
    def _load_model(self):
        """Load the face recognition model (placeholder for future deep learning models)."""
        try:
            # For now, we use face_recognition library which doesn't need explicit model loading
            # This method is kept for future implementation of custom models
            logger.info("Model loading placeholder - using face_recognition library")
            self.model = "face_recognition_library"
        except Exception as e:
            logger.error(f"Failed to load face recognition model: {e}")
            raise
    
    def reload_user_data(self):
        """
        Reload user face encodings from the database.
        This method is called to refresh the face recognizer with latest user data.
        """
        try:
            self.known_face_encodings.clear()
            self.known_ids.clear()
            self.known_names.clear()
            
            logger.info("Cleared all known face encodings")
            
            if not os.path.exists(self.database_path):
                logger.warning(f"Database path does not exist: {self.database_path}")
                return
            
            # Update database modification time
            self.database_mod_time = os.path.getmtime(self.database_path)
            self.last_reload_time = datetime.now().timestamp()
            
            # Load users from database
            try:
                from src.utils.data_utils import load_registered_users
                users = load_registered_users()
                
                if not users:
                    # This is normal for a new system
                    logger.info("No registered users found in the database. System is ready for new registrations.")
                    
                    # Reset the warning flag when reloading empty database to allow one warning per session
                    FaceRecognizer._no_users_warned = False
                    
                    # Create a placeholder file to indicate the database has been checked
                    placeholder_path = os.path.join(self.database_path, ".database_verified")
                    try:
                        os.makedirs(os.path.dirname(placeholder_path), exist_ok=True)
                        with open(placeholder_path, "w") as f:
                            f.write(f"Database verified at {datetime.now().isoformat()}")
                    except Exception:
                        pass
                    return
                
                for user_id, name, encoding in users:
                    self.known_face_encodings.append(encoding)
                    self.known_ids.append(user_id)
                    self.known_names.append(name)
                    logger.info(f"Loaded encoding for user {user_id}")
                
                logger.info(f"Reloaded {len(self.known_face_encodings)} user face encodings")
            except ImportError:
                logger.warning("Could not import data_utils - running without user database")
            
        except Exception as e:
            logger.error(f"Failed to reload user data: {e}")
            # Don't raise here - let the system continue without user data
    
    def check_and_reload_if_needed(self):
        """
        Check for reload notifications and database changes, then reload user data if needed.
        This enables real-time updates when new users are registered.
        """
        should_reload = False
        
        try:
            from src.utils.data_utils import check_reload_notification
            # Check for explicit reload notifications
            if check_reload_notification():
                logger.info("Reload notification detected - refreshing user database")
                should_reload = True
            
            # Also check database directory modification time
            current_time = datetime.now().timestamp()
            if os.path.exists(self.database_path):
                db_mod_time = os.path.getmtime(self.database_path)
                if db_mod_time > self.database_mod_time:
                    logger.info("Database directory modification detected - refreshing user database")
                    self.database_mod_time = db_mod_time
                    should_reload = True
            
            # Periodic reload check (every 30 seconds as fallback)
            if current_time - self.last_reload_time > 30:
                logger.debug("Periodic reload check")
                should_reload = True
            
            if should_reload:
                self.reload_user_data()
                self.last_reload_time = current_time
                
        except Exception as e:
            logger.warning(f"Failed to check reload notification: {e}")
    
    def extract_face_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract face encoding/embedding from a face image.
        
        Args:
            face_image (np.ndarray): Face image (cropped face)
            
        Returns:
            np.ndarray: Face encoding vector
            
        Raises:
            ValueError: If no face is detected in the image
        """
        try:
            # Convert BGR to RGB if needed
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = face_image
            
            # Find face locations
            face_locations = face_recognition.face_locations(rgb_image, model=self.face_locations_model)
            
            if not face_locations:
                # If no face detected in cropped image, try to use the whole image
                logger.warning("No face detected in provided image, using whole image")
                face_locations = [(0, rgb_image.shape[1], rgb_image.shape[0], 0)]
            
            # Extract face encodings
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if not face_encodings:
                raise ValueError("Could not extract face encoding from image")
            
            # Return the first encoding
            return face_encodings[0]
            
        except Exception as e:
            logger.error(f"Failed to extract face embedding: {e}")
            raise
    
    def register_face(self, face_image: np.ndarray, user_id: str, name: Optional[str] = None) -> bool:
        """
        Register a new face in the database.
        
        Args:
            face_image (np.ndarray): Face image
            user_id (str): Unique user identifier
            name (str, optional): User's name
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            # Extract face encoding
            encoding = self.extract_face_embedding(face_image)
            
            # Add to in-memory database
            if user_id in self.known_ids:
                # Update existing user
                index = self.known_ids.index(user_id)
                self.known_face_encodings[index] = encoding
                if name:
                    self.known_names[index] = name
            else:
                # Add new user
                self.known_face_encodings.append(encoding)
                self.known_ids.append(user_id)
                self.known_names.append(name or user_id)
            
            logger.info(f"Registered face for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register face for user {user_id}: {e}")
            return False
    
    # Class variable to track if we've warned about no users
    _no_users_warned = False
    
    def recognize_face(self, face_image: np.ndarray) -> Tuple[str, str, float]:
        """
        Recognize a face and return user identification with confidence.
        
        Args:
            face_image (np.ndarray): Face image to recognize
            
        Returns:
            Tuple[str, str, float]: (user_id, name, confidence)
                - user_id: Identified user ID or "Unknown"
                - name: User's name or "Unknown"
                - confidence: Recognition confidence (0.0 to 1.0)
        """
        try:
            # Check for new user registrations and reload if needed
            self.check_and_reload_if_needed()
            if not self.known_face_encodings:
                # Only log once per session to avoid spamming logs
                if not FaceRecognizer._no_users_warned:
                    logger.warning("No registered users in database - recognition will return 'Unknown' until users are registered")
                    FaceRecognizer._no_users_warned = True
                else:
                    # Use debug level for subsequent checks
                    logger.debug("No registered users in database")
                return "Unknown", "Unknown", 0.0
            
            # Extract face encoding from input image
            unknown_encoding = self.extract_face_embedding(face_image)
            
            # Compare with known face encodings
            face_distances = face_recognition.face_distance(self.known_face_encodings, unknown_encoding)
            
            # Find the best match
            min_distance = np.min(face_distances)
            best_match_index = np.argmin(face_distances)
            
            # Convert distance to confidence (lower distance = higher confidence)
            confidence = 1.0 - min_distance
            
            # Check if the match is good enough
            if min_distance <= self.recognition_threshold:
                user_id = self.known_ids[best_match_index]
                name = self.known_names[best_match_index]
                logger.debug(f"Face recognized: {user_id} with confidence {confidence:.3f}")
                return user_id, name, float(confidence)
            else:
                logger.debug(f"Face not recognized, best distance: {min_distance:.3f}")
                return "Unknown", "Unknown", float(confidence)
                
        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            return "Unknown", "Unknown", 0.0
    
    def calculate_similarity(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """
        Calculate similarity between two face encodings.
        
        Args:
            encoding1 (np.ndarray): First face encoding
            encoding2 (np.ndarray): Second face encoding
            
        Returns:
            float: Similarity score (0.0 to 1.0, higher is more similar)
        """
        try:
            # Calculate Euclidean distance
            distance = np.linalg.norm(encoding1 - encoding2)
            # Convert to similarity (higher values mean more similar)
            similarity = 1.0 - distance
            return float(max(0.0, min(1.0, similarity)))  # Clamp between 0 and 1
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def get_user_count(self) -> int:
        """
        Get the number of registered users.
        
        Returns:
            int: Number of registered users
        """
        return len(self.known_ids)
    
    def get_user_list(self) -> List[Tuple[str, str]]:
        """
        Get list of all registered users.
        
        Returns:
            List[Tuple[str, str]]: List of (user_id, name) tuples
        """
        return list(zip(self.known_ids, self.known_names))
    
    def remove_user(self, user_id: str) -> bool:
        """
        Remove a user from the face recognition database.
        
        Args:
            user_id (str): User ID to remove
            
        Returns:
            bool: True if user was removed, False if not found
        """
        try:
            if user_id in self.known_ids:
                index = self.known_ids.index(user_id)
                del self.known_face_encodings[index]
                del self.known_ids[index]
                del self.known_names[index]
                logger.info(f"Removed user {user_id} from recognition database")
                return True
            else:
                logger.warning(f"User {user_id} not found in database")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove user {user_id}: {e}")
            return False
    
    def update_recognition_threshold(self, threshold: float):
        """
        Update the recognition threshold.
        
        Args:
            threshold (float): New threshold (0.0 to 1.0, lower is stricter)
        """
        if 0.0 <= threshold <= 1.0:
            self.recognition_threshold = threshold
            logger.info(f"Recognition threshold updated to {threshold}")
        else:
            logger.warning(f"Invalid threshold value: {threshold}. Must be between 0.0 and 1.0")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get recognition system statistics.
        
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        return {
            "registered_users": len(self.known_ids),
            "recognition_threshold": self.recognition_threshold,
            "face_locations_model": self.face_locations_model,
            "model_path": self.model_path,
            "database_path": self.database_path
        }
    
    def __repr__(self) -> str:
        """String representation of the FaceRecognizer."""
        return f"FaceRecognizer(users={len(self.known_ids)}, threshold={self.recognition_threshold})"
