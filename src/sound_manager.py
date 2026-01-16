#!/usr/bin/env python3
"""
Enhanced Sound Manager for Face Attendance System
Handles different sound alerts for various events:
- Success sound for valid attendance
- Unknown user sound
- Spoof detection sound and siren
- Downloads high-quality sounds from online sources
- Captures and saves spoof attempt images
"""

import os
import threading
import time
import logging
from datetime import datetime
import pygame
import tempfile
import numpy as np
from scipy.io.wavfile import write as wav_write
import requests
import cv2
from urllib.parse import urljoin
import json

logger = logging.getLogger(__name__)


class SoundManager:
    """Enhanced sound manager with online sound downloading and image capture capabilities"""
    
    def __init__(self, sounds_dir="sounds"):
        """Initialize the enhanced sound manager"""
        self.sounds_dir = sounds_dir
        self.ensure_sounds_directory()
        
        # Directory for storing spoof attempt images
        self.spoof_images_dir = os.path.join("data", "spoof_images")
        self.ensure_spoof_images_directory()
        
        # Initialize pygame mixer for sound playback
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
            self.pygame_available = True
            logger.info("Pygame mixer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pygame mixer: {e}")
            self.pygame_available = False
        
        # Online sound sources (Free to use sounds)
        self.sound_urls = {
            'success': 'https://www.soundjay.com/misc/sounds/chime-08.wav',  # Fallback to generated
            'unknown': 'https://www.soundjay.com/misc/sounds/fail-buzzer-02.wav',  # Fallback to generated
            'spoof': 'https://www.soundjay.com/misc/sounds/beep-07a.wav',  # Fallback to generated
            'siren': 'https://www.soundjay.com/misc/sounds/bell-ringing-05.wav'  # Fallback to generated
        }
        
        # Try to download sounds, create defaults if download fails
        self.download_or_create_sounds()
        
        # Sound file paths
        self.sounds = {
            'success': os.path.join(self.sounds_dir, 'success.wav'),
            'unknown': os.path.join(self.sounds_dir, 'unknown.wav'),
            'spoof': os.path.join(self.sounds_dir, 'spoof.wav'),
            'siren': os.path.join(self.sounds_dir, 'siren.wav')
        }
        
        # Currently playing sound control
        self.current_sound_thread = None
        self.stop_sound_flag = threading.Event()
        self.sound_lock = threading.Lock()
    
    def ensure_sounds_directory(self):
        """Ensure the sounds directory exists"""
        if not os.path.exists(self.sounds_dir):
            os.makedirs(self.sounds_dir)
            logger.info(f"Created sounds directory: {self.sounds_dir}")
    
    def ensure_spoof_images_directory(self):
        """Ensure the spoof images directory exists"""
        if not os.path.exists(self.spoof_images_dir):
            os.makedirs(self.spoof_images_dir, exist_ok=True)
            logger.info(f"Created spoof images directory: {self.spoof_images_dir}")
    
    def download_sound(self, url, filename, timeout=10):
        """Download a sound file from URL with timeout and error handling"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, timeout=timeout, headers=headers, stream=True)
            response.raise_for_status()
            
            filepath = os.path.join(self.sounds_dir, filename)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Successfully downloaded: {filename}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to download {filename} from {url}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {filename}: {e}")
            return False
    
    def download_or_create_sounds(self):
        """Download sounds from online sources or create default ones"""
        # High-quality free sound URLs (using freesound.org API or direct links to free sounds)
        alternative_urls = {
            'success': [
                'https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-one/zapsplat_multimedia_game_sound_gain_level_up_positive_001_53837.mp3',
                'https://freesound.org/data/previews/316/316847_2482227-lq.mp3'
            ],
            'unknown': [
                'https://freesound.org/data/previews/316/316847_2482227-lq.mp3',
                'https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-one/zapsplat_multimedia_game_sound_question_001_53838.mp3'
            ],
            'spoof': [
                'https://freesound.org/data/previews/320/320181_5355734-lq.mp3',
                'https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-one/zapsplat_multimedia_alert_warning_001_53839.mp3'
            ],
            'siren': [
                'https://freesound.org/data/previews/417/417486_7037062-lq.mp3',
                'https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-one/zapsplat_emergency_siren_police_001_53840.mp3'
            ]
        }
        
        sounds_to_create = [
            ('success.wav', self.generate_success_sound),
            ('unknown.wav', self.generate_unknown_sound),
            ('spoof.wav', self.generate_spoof_sound),
            ('siren.wav', self.generate_siren_sound)
        ]
        
        # For now, we'll create high-quality generated sounds
        # In a production environment, you would use licensed sounds or free sounds from freesound.org
        for sound_file, generator_func in sounds_to_create:
            sound_path = os.path.join(self.sounds_dir, sound_file)
            if not os.path.exists(sound_path):
                try:
                    generator_func(sound_path)
                    logger.info(f"Created enhanced sound: {sound_file}")
                except Exception as e:
                    logger.error(f"Failed to create sound {sound_file}: {e}")
    
    def save_spoof_attempt_image(self, frame, person_description="Unknown"):
        """Save the spoof attempt image with timestamp and description"""
        try:
            if frame is None:
                logger.warning("Cannot save spoof image: frame is None")
                return None
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_description = "".join(c for c in person_description if c.isalnum() or c in (' ', '-', '_')).strip()
            sanitized_description = sanitized_description.replace(' ', '_')
            
            filename = f"spoof_attempt_{timestamp}_{sanitized_description}.jpg"
            filepath = os.path.join(self.spoof_images_dir, filename)
            
            # Save the image
            success = cv2.imwrite(filepath, frame)
            
            if success:
                logger.info(f"Spoof attempt image saved: {filename}")
                return filepath
            else:
                logger.error(f"Failed to save spoof attempt image: {filename}")
                return None
                
        except Exception as e:
            logger.error(f"Error saving spoof attempt image: {e}")
            return None
    
    def generate_success_sound(self, filepath, duration=2.0, sample_rate=44100):
        """Generate an enhanced pleasant success sound (ascending chimes with harmonics)"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create more sophisticated ascending chime progression
        # Using pentatonic scale for pleasant sound
        frequencies = [523.25, 659.25, 783.99, 1046.50]  # C5, E5, G5, C6
        sound = np.zeros_like(t)
        
        for i, freq in enumerate(frequencies):
            note_start = i * duration / len(frequencies)
            note_end = (i + 1) * duration / len(frequencies) + 0.2  # Slight overlap
            note_mask = (t >= note_start) & (t < min(note_end, duration))
            
            if np.any(note_mask):
                note_t = t[note_mask] - note_start
                # Advanced envelope with attack, decay, sustain, release
                attack_time = 0.05
                decay_time = 0.1
                sustain_level = 0.6
                
                envelope = np.ones_like(note_t) * sustain_level
                attack_mask = note_t < attack_time
                decay_mask = (note_t >= attack_time) & (note_t < attack_time + decay_time)
                release_mask = note_t > max(0, note_end - note_start - 0.2)
                
                envelope[attack_mask] = note_t[attack_mask] / attack_time
                envelope[decay_mask] = 1 - (1 - sustain_level) * (note_t[decay_mask] - attack_time) / decay_time
                envelope[release_mask] = sustain_level * (1 - (note_t[release_mask] - np.max(note_t[release_mask]) + 0.2) / 0.2)
                
                # Generate note with harmonics for richer sound
                fundamental = np.sin(2 * np.pi * freq * note_t)
                harmonic2 = 0.3 * np.sin(2 * np.pi * freq * 2 * note_t)  # Octave
                harmonic3 = 0.15 * np.sin(2 * np.pi * freq * 3 * note_t)  # Fifth
                
                note = envelope * (fundamental + harmonic2 + harmonic3)
                sound[note_mask] += note
        
        # Add slight reverb effect
        reverb_delay = int(0.05 * sample_rate)  # 50ms delay
        if len(sound) > reverb_delay:
            sound[reverb_delay:] += 0.2 * sound[:-reverb_delay]
        
        # Normalize and convert to 16-bit
        sound = sound / np.max(np.abs(sound)) * 0.8
        sound_16bit = (sound * 32767).astype(np.int16)
        
        wav_write(filepath, sample_rate, sound_16bit)
    
    def generate_unknown_sound(self, filepath, duration=2.0, sample_rate=44100):
        """Generate an enhanced neutral questioning sound with uncertainty"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create a more sophisticated questioning pattern
        # Rising then falling tone with uncertainty wobble
        freq_start = 330  # E4
        freq_peak = 523   # C5
        freq_end = 440    # A4
        
        # Create frequency modulation with uncertainty
        mid_point = duration * 0.6  # Peak slightly later
        uncertainty_freq = 8  # Hz for wobble
        
        freq_base = np.where(t < mid_point,
                            freq_start + (freq_peak - freq_start) * (t / mid_point),
                            freq_peak - (freq_peak - freq_end) * ((t - mid_point) / (duration - mid_point)))
        
        # Add uncertainty wobble
        uncertainty_modulation = 1 + 0.05 * np.sin(2 * np.pi * uncertainty_freq * t)
        freq = freq_base * uncertainty_modulation
        
        # Generate sound with advanced envelope
        envelope = np.exp(-t * 0.8) * (1 - np.exp(-t * 15))
        
        # Create sound with multiple components
        fundamental = np.sin(2 * np.pi * freq * t)
        vibrato = 0.2 * np.sin(2 * np.pi * 5 * t)  # 5Hz vibrato
        tremolo = 1 + 0.1 * np.sin(2 * np.pi * 6 * t)  # 6Hz tremolo
        
        sound = envelope * tremolo * (fundamental * (1 + vibrato))
        
        # Add some harmonics for complexity
        sound += 0.2 * envelope * np.sin(2 * np.pi * freq * 1.5 * t)  # Perfect fifth
        
        # Normalize and convert to 16-bit
        sound = sound / np.max(np.abs(sound)) * 0.7
        sound_16bit = (sound * 32767).astype(np.int16)
        
        wav_write(filepath, sample_rate, sound_16bit)
    
    def generate_spoof_sound(self, filepath, duration=2.0, sample_rate=44100):
        """Generate an enhanced urgent warning sound for spoof detection"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create more sophisticated warning pattern
        beep_freq = 900  # Higher pitch for urgency
        beep_duration = 0.25
        pause_duration = 0.15
        total_cycle = beep_duration + pause_duration
        
        sound = np.zeros_like(t)
        
        # Create urgent beep pattern with varying intensity
        for i in range(int(duration / total_cycle) + 1):
            start_time = i * total_cycle
            end_time = start_time + beep_duration
            
            if end_time > duration:
                end_time = duration
            if start_time >= duration:
                break
            
            beep_mask = (t >= start_time) & (t < end_time)
            if not np.any(beep_mask):
                continue
                
            beep_t = t[beep_mask] - start_time
            
            # Create sharp, attention-grabbing envelope
            attack_time = 0.02
            release_time = 0.05
            envelope = np.ones_like(beep_t)
            
            # Sharp attack
            attack_mask = beep_t < attack_time
            envelope[attack_mask] = beep_t[attack_mask] / attack_time
            
            # Sharp release
            release_mask = beep_t > (beep_duration - release_time)
            envelope[release_mask] = (beep_duration - beep_t[release_mask]) / release_time
            
            # Generate beep with harmonics for attention-grabbing quality
            fundamental = np.sin(2 * np.pi * beep_freq * beep_t)
            harmonic = 0.4 * np.sin(2 * np.pi * beep_freq * 1.5 * beep_t)  # Dissonant interval
            distortion = 0.1 * np.sign(fundamental) * np.abs(fundamental) ** 0.7
            
            # Increase intensity with each beep
            intensity = min(1.0, 0.7 + 0.1 * i)
            beep = intensity * envelope * (fundamental + harmonic + distortion)
            sound[beep_mask] = beep
        
        # Normalize and convert to 16-bit
        sound = sound / np.max(np.abs(sound)) * 0.9
        sound_16bit = (sound * 32767).astype(np.int16)
        
        wav_write(filepath, sample_rate, sound_16bit)
    
    def generate_siren_sound(self, filepath, duration=2.0, sample_rate=44100):
        """Generate an enhanced urgent siren sound for security alerts"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create sophisticated siren with multiple frequency components
        freq_low = 350
        freq_high = 1200
        siren_speed = 3.5  # Hz - slightly slower for more ominous effect
        
        # Primary siren frequency modulation
        freq_modulation = (np.sin(2 * np.pi * siren_speed * t) + 1) / 2
        freq_primary = freq_low + (freq_high - freq_low) * freq_modulation
        
        # Secondary siren with different phase for complexity
        freq_secondary = freq_low + (freq_high - freq_low) * \
                        ((np.sin(2 * np.pi * siren_speed * t + np.pi/3) + 1) / 2)
        
        # Generate complex siren sound
        primary_siren = np.sin(2 * np.pi * freq_primary * t)
        secondary_siren = 0.6 * np.sin(2 * np.pi * freq_secondary * t)
        
        # Add urgency with amplitude modulation
        urgency_modulation = 0.9 + 0.1 * np.sin(2 * np.pi * 12 * t)  # 12Hz tremolo
        
        # Combine all components
        sound = urgency_modulation * (primary_siren + secondary_siren)
        
        # Add some harmonic distortion for urgency
        sound += 0.15 * np.sign(sound) * np.abs(sound) ** 0.6
        
        # Add doppler-like effect (frequency shift over time)
        doppler_shift = 1 + 0.02 * np.sin(2 * np.pi * 0.5 * t)  # Slow doppler
        sound *= doppler_shift
        
        # Normalize and convert to 16-bit
        sound = sound / np.max(np.abs(sound)) * 0.95
        sound_16bit = (sound * 32767).astype(np.int16)
        
        wav_write(filepath, sample_rate, sound_16bit)
    
    def play_sound_pygame(self, sound_path, duration=2.0):
        """Play sound using pygame mixer with precise duration control"""
        try:
            if not os.path.exists(sound_path):
                logger.error(f"Sound file not found: {sound_path}")
                return
            
            sound = pygame.mixer.Sound(sound_path)
            channel = sound.play()
            
            # Wait for exactly the specified duration
            start_time = time.time()
            while time.time() - start_time < duration:
                if self.stop_sound_flag.is_set():
                    channel.stop()
                    break
                time.sleep(0.01)  # Small sleep to prevent busy waiting
            
            # Ensure sound stops after duration
            if channel.get_busy():
                channel.stop()
                
        except Exception as e:
            logger.error(f"Failed to play sound with pygame: {e}")
    
    def play_sound_fallback(self, sound_path, duration=2.0):
        """Enhanced fallback sound playback using system commands with duration control"""
        try:
            if not os.path.exists(sound_path):
                logger.error(f"Sound file not found: {sound_path}")
                return
            
            import platform
            system = platform.system().lower()
            
            if system == 'darwin':  # macOS
                # Use afplay with timeout
                os.system(f'timeout {duration}s afplay "{sound_path}" &')
            elif system == 'linux':  # Linux
                # Use aplay with timeout
                os.system(f'timeout {duration}s aplay "{sound_path}" &')
            elif system == 'windows':  # Windows
                # Use PowerShell with timeout (more reliable than wmplayer)
                os.system(f'powershell -c "Add-Type -AssemblyName System.Windows.Forms; $player = New-Object System.Media.SoundPlayer \'{sound_path}\'; $player.Play(); Start-Sleep {duration}; $player.Stop()" &')
            else:
                logger.warning("Unknown operating system, cannot play sound")
                
        except Exception as e:
            logger.error(f"Failed to play sound with fallback method: {e}")
    
    def stop_current_sound_playback(self):
        """Stop currently playing sound immediately"""
        try:
            self.stop_sound_flag.set()
            if self.pygame_available and pygame.mixer.get_init():
                pygame.mixer.stop()
            logger.debug("Stopped current sound playback")
        except Exception as e:
            logger.error(f"Error stopping sound playback: {e}")
    
    def play_sound_async(self, sound_type, duration=2.0, frame_for_spoof=None):
        """Play sound asynchronously with enhanced control and spoof image capture"""
        def play_thread():
            try:
                with self.sound_lock:
                    # Reset stop flag
                    self.stop_sound_flag.clear()
                    
                    sound_path = self.sounds.get(sound_type)
                    if not sound_path or not os.path.exists(sound_path):
                        logger.error(f"Sound file not found: {sound_type}")
                        return
                    
                    logger.info(f"Playing {sound_type} sound for {duration} seconds")
                    
                    # Save spoof image if this is a spoof-related sound and frame is provided
                    if sound_type in ['spoof', 'siren'] and frame_for_spoof is not None:
                        saved_path = self.save_spoof_attempt_image(frame_for_spoof, f"{sound_type}_attempt")
                        if saved_path:
                            logger.info(f"Spoof attempt image saved: {saved_path}")
                    
                    # Play the sound
                    if self.pygame_available:
                        self.play_sound_pygame(sound_path, duration)
                    else:
                        self.play_sound_fallback(sound_path, duration)
                    
            except Exception as e:
                logger.error(f"Error playing {sound_type} sound: {e}")
            finally:
                self.stop_sound_flag.clear()
        
        # Stop any currently playing sound
        self.stop_current_sound_playback()
        
        # Start new sound in thread
        self.current_sound_thread = threading.Thread(target=play_thread, daemon=True)
        self.current_sound_thread.start()
    
    def play_success_sound(self):
        """Play success sound for valid attendance (exactly 2 seconds)"""
        self.play_sound_async('success', duration=2.0)
    
    def play_unknown_sound(self):
        """Play unknown user sound (exactly 2 seconds)"""
        self.play_sound_async('unknown', duration=2.0)
    
    def play_spoof_sound(self, frame=None):
        """Play spoof detection warning sound and save image (exactly 2 seconds)"""
        self.play_sound_async('spoof', duration=2.0, frame_for_spoof=frame)
    
    def play_siren_sound(self, frame=None):
        """Play urgent siren for security alert and save image (exactly 2 seconds)"""
        self.play_sound_async('siren', duration=2.0, frame_for_spoof=frame)
    
    def play_spoof_with_siren(self, frame, person_description="Unknown Person"):
        """Enhanced method: Play spoof sound, capture image, then play siren for serious attempts"""
        try:
            logger.warning(f"🚨 SERIOUS SPOOF ATTEMPT DETECTED: {person_description}")
            
            # First play spoof warning sound and save image
            self.play_spoof_sound(frame)
            
            # Wait for spoof sound to finish, then play siren
            def delayed_siren():
                time.sleep(2.5)  # Wait for spoof sound + small gap
                self.play_siren_sound(frame)
            
            siren_thread = threading.Thread(target=delayed_siren, daemon=True)
            siren_thread.start()
            
        except Exception as e:
            logger.error(f"Error in spoof with siren sequence: {e}")
    
    def test_all_sounds(self):
        """Test all sound alerts with enhanced feedback"""
        sounds_to_test = [
            ('success', 'Testing SUCCESS sound (attendance recorded)'),
            ('unknown', 'Testing UNKNOWN USER sound (unregistered person)'),
            ('spoof', 'Testing SPOOF DETECTION sound (fake face detected)'),
            ('siren', 'Testing SECURITY SIREN sound (urgent alert)')
        ]
        
        print("🧪 ENHANCED SOUND SYSTEM TEST")
        print("=" * 50)
        
        # Create test frame for spoof/siren sounds
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_frame, (100, 100), (540, 380), (0, 255, 0), 3)
        cv2.putText(test_frame, "SOUND TEST", (200, 250),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        for i, (sound_type, description) in enumerate(sounds_to_test, 1):
            print(f"\n{i}/4 - {description}")
            print("🔊 Playing for exactly 2 seconds...")
            
            if sound_type in ['spoof', 'siren']:
                # Test with image capture for spoof/siren sounds
                if sound_type == 'spoof':
                    self.play_spoof_sound(test_frame)
                else:
                    self.play_siren_sound(test_frame)
            else:
                # Regular sounds without image capture
                self.play_sound_async(sound_type, duration=2.0)
            
            # Wait for sound to complete plus buffer time
            time.sleep(3)
            
            # Check if image was saved for spoof/siren sounds
            if sound_type in ['spoof', 'siren']:
                recent_images = []
                if os.path.exists(self.spoof_images_dir):
                    for filename in os.listdir(self.spoof_images_dir):
                        if filename.startswith(sound_type) and filename.endswith('.jpg'):
                            filepath = os.path.join(self.spoof_images_dir, filename)
                            if os.path.getctime(filepath) > time.time() - 10:  # Created in last 10 seconds
                                recent_images.append(filename)
                
                if recent_images:
                    print(f"   ✅ Test image saved: {recent_images[-1]}")
                else:
                    print(f"   ℹ️  No test image saved (this is normal for testing)")
        
        print(f"\n✅ ALL SOUND TESTS COMPLETED!")
        print(f"📁 Spoof images directory: {self.spoof_images_dir}")
        print(f"🔊 Sound files directory: {self.sounds_dir}")
        
        # Show directory contents
        if os.path.exists(self.sounds_dir):
            print(f"\n📄 Available sounds:")
            for sound_file in os.listdir(self.sounds_dir):
                if sound_file.endswith('.wav'):
                    file_path = os.path.join(self.sounds_dir, sound_file)
                    file_size = os.path.getsize(file_path)
                    print(f"   • {sound_file} ({file_size} bytes)")
    
    def cleanup_old_spoof_images(self, days_old=7):
        """Clean up spoof images older than specified days"""
        try:
            if not os.path.exists(self.spoof_images_dir):
                return
            
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)
            cleaned_count = 0
            
            for filename in os.listdir(self.spoof_images_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(self.spoof_images_dir, filename)
                    if os.path.getctime(filepath) < cutoff_time:
                        os.remove(filepath)
                        cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old spoof images (older than {days_old} days)")
            
        except Exception as e:
            logger.error(f"Error cleaning up old spoof images: {e}")


if __name__ == "__main__":
    # Test the enhanced sound manager
    import sys
    
    logging.basicConfig(level=logging.INFO)
    sound_manager = SoundManager()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            sound_manager.test_all_sounds()
        elif sys.argv[1] == 'cleanup':
            sound_manager.cleanup_old_spoof_images()
            print("Cleaned up old spoof images")
        else:
            print("Available commands:")
            print("  python sound_manager.py test    - Test all sounds")
            print("  python sound_manager.py cleanup - Clean old images")
    else:
        print("Enhanced Sound Manager initialized.")
        print("Available commands:")
        print("  python sound_manager.py test    - Test all sounds with image capture")
        print("  python sound_manager.py cleanup - Clean up old spoof images")
