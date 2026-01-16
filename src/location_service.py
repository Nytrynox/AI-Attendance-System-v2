#!/usr/bin/env python3
"""
Location Service for Face Attendance System
Provides exact live location information for email notifications
"""

import requests
import json
import logging
import socket
from datetime import datetime
import platform
import os

logger = logging.getLogger(__name__)


class LocationService:
    """Manages location information for security alerts"""
    
    def __init__(self, config_file="location_config.json"):
        """Initialize location service"""
        self.config_file = config_file
        self.config = self.load_or_create_config()
        self.enabled = self.config.get('enabled', True)
        
        if self.enabled:
            logger.info("Location service enabled")
        else:
            logger.info("Location service disabled")
    
    def load_or_create_config(self):
        """Load location configuration or create default config"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.info("Location configuration loaded")
                return config
            except Exception as e:
                logger.error(f"Failed to load location config: {e}")
                return self.create_default_config()
        else:
            return self.create_default_config()
    
    def create_default_config(self):
        """Create default location configuration"""
        default_config = {
            "enabled": True,
            "use_ip_location": True,
            "use_manual_location": True,
            "manual_location": {
                "building_name": "Main Office Building",
                "floor": "Ground Floor",
                "room": "Reception Area",
                "address": "123 Business Street, City, State 12345",
                "coordinates": {
                    "latitude": 0.0,
                    "longitude": 0.0
                },
                "contact_info": {
                    "security_phone": "+1-XXX-XXX-XXXX",
                    "office_phone": "+1-XXX-XXX-XXXX"
                }
            },
            "fallback_location": "Face Attendance System - Location Unknown",
            "location_services": {
                "ipapi_enabled": True,
                "ipgeolocation_enabled": False,
                "ipgeolocation_api_key": ""
            },
            "notes": [
                "Set manual_location details for your specific installation",
                "IP-based location provides approximate location based on internet connection",
                "For precise location, update manual_location coordinates",
                "You can get coordinates from Google Maps or GPS devices"
            ]
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            logger.info(f"Created default location config: {self.config_file}")
            print(f"\n📍 LOCATION CONFIGURATION CREATED")
            print(f"Please edit {self.config_file} with your exact location details")
        except Exception as e:
            logger.error(f"Failed to create location config: {e}")
        
        return default_config
    
    def get_ip_location(self):
        """Get location based on IP address"""
        try:
            if not self.config.get('location_services', {}).get('ipapi_enabled', True):
                return None
            
            # Try ip-api.com (free service, no API key required)
            response = requests.get('http://ip-api.com/json/', timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success':
                    location_info = {
                        'country': data.get('country', 'Unknown'),
                        'region': data.get('regionName', 'Unknown'),
                        'city': data.get('city', 'Unknown'),
                        'zip': data.get('zip', 'Unknown'),
                        'latitude': data.get('lat', 0.0),
                        'longitude': data.get('lon', 0.0),
                        'isp': data.get('isp', 'Unknown'),
                        'timezone': data.get('timezone', 'Unknown'),
                        'source': 'IP-based location'
                    }
                    logger.info(f"IP location obtained: {location_info['city']}, {location_info['region']}")
                    return location_info
            
        except requests.RequestException as e:
            logger.warning(f"Failed to get IP location: {e}")
        except Exception as e:
            logger.error(f"IP location error: {e}")
        
        return None
    
    def get_system_info(self):
        """Get system information"""
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            system_info = {
                'hostname': hostname,
                'local_ip': local_ip,
                'platform': platform.platform(),
                'system': platform.system(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version()
            }
            return system_info
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {}
    
    def get_manual_location(self):
        """Get manually configured location"""
        try:
            manual_config = self.config.get('manual_location', {})
            if manual_config and manual_config.get('building_name'):
                location_info = {
                    'building_name': manual_config.get('building_name', 'Unknown Building'),
                    'floor': manual_config.get('floor', 'Unknown Floor'),
                    'room': manual_config.get('room', 'Unknown Room'),
                    'address': manual_config.get('address', 'Address not configured'),
                    'coordinates': manual_config.get('coordinates', {'latitude': 0.0, 'longitude': 0.0}),
                    'contact_info': manual_config.get('contact_info', {}),
                    'source': 'Manual configuration'
                }
                logger.info(f"Manual location: {location_info['building_name']} - {location_info['room']}")
                return location_info
        except Exception as e:
            logger.error(f"Failed to get manual location: {e}")
        
        return None
    
    def get_complete_location_info(self):
        """Get complete location information combining all sources"""
        if not self.enabled:
            return {'location': self.config.get('fallback_location', 'Face Attendance System')}
        
        try:
            location_data = {
                'timestamp': datetime.now().isoformat(),
                'manual_location': None,
                'ip_location': None,
                'system_info': None,
                'formatted_location': ''
            }
            
            # Get manual location (most accurate)
            if self.config.get('use_manual_location', True):
                location_data['manual_location'] = self.get_manual_location()
            
            # Get IP-based location
            if self.config.get('use_ip_location', True):
                location_data['ip_location'] = self.get_ip_location()
            
            # Get system information
            location_data['system_info'] = self.get_system_info()
            
            # Create formatted location string
            location_data['formatted_location'] = self.format_location_for_email(location_data)
            
            return location_data
            
        except Exception as e:
            logger.error(f"Failed to get complete location info: {e}")
            return {'location': self.config.get('fallback_location', 'Face Attendance System')}
    
    def format_location_for_email(self, location_data):
        """Format location data for email display"""
        try:
            formatted_parts = []
            
            # Manual location (highest priority)
            manual = location_data.get('manual_location')
            if manual:
                manual_text = f"🏢 **EXACT LOCATION:**\n"
                manual_text += f"   • Building: {manual['building_name']}\n"
                manual_text += f"   • Floor: {manual['floor']}\n"
                manual_text += f"   • Room: {manual['room']}\n"
                manual_text += f"   • Address: {manual['address']}\n"
                
                coords = manual.get('coordinates', {})
                if coords.get('latitude') and coords.get('longitude'):
                    manual_text += f"   • GPS: {coords['latitude']}, {coords['longitude']}\n"
                    # Create Google Maps link
                    maps_url = f"https://www.google.com/maps?q={coords['latitude']},{coords['longitude']}"
                    manual_text += f"   • Maps: {maps_url}\n"
                
                contact = manual.get('contact_info', {})
                if contact:
                    manual_text += f"   • Security: {contact.get('security_phone', 'N/A')}\n"
                    manual_text += f"   • Office: {contact.get('office_phone', 'N/A')}\n"
                
                formatted_parts.append(manual_text)
            
            # IP location (approximate)
            ip_loc = location_data.get('ip_location')
            if ip_loc:
                ip_text = f"🌐 **NETWORK LOCATION:**\n"
                ip_text += f"   • City: {ip_loc['city']}, {ip_loc['region']}\n"
                ip_text += f"   • Country: {ip_loc['country']}\n"
                ip_text += f"   • ZIP: {ip_loc['zip']}\n"
                ip_text += f"   • ISP: {ip_loc['isp']}\n"
                ip_text += f"   • Timezone: {ip_loc['timezone']}\n"
                if ip_loc.get('latitude') and ip_loc.get('longitude'):
                    ip_text += f"   • Approx GPS: {ip_loc['latitude']}, {ip_loc['longitude']}\n"
                
                formatted_parts.append(ip_text)
            
            # System information
            system = location_data.get('system_info')
            if system:
                system_text = f"💻 **SYSTEM INFO:**\n"
                system_text += f"   • Hostname: {system.get('hostname', 'Unknown')}\n"
                system_text += f"   • Local IP: {system.get('local_ip', 'Unknown')}\n"
                system_text += f"   • Platform: {system.get('system', 'Unknown')}\n"
                
                formatted_parts.append(system_text)
            
            # Combine all parts
            if formatted_parts:
                return '\n'.join(formatted_parts)
            else:
                return "📍 Location: Face Attendance System (Location details not configured)"
                
        except Exception as e:
            logger.error(f"Failed to format location: {e}")
            return "📍 Location: Face Attendance System (Location formatting error)"
    
    def get_location_summary(self):
        """Get a brief location summary for quick reference"""
        try:
            location_data = self.get_complete_location_info()
            
            # Priority: Manual location > IP location > fallback
            manual = location_data.get('manual_location')
            if manual and manual.get('building_name'):
                return f"{manual['building_name']} - {manual['room']} ({manual['address']})"
            
            ip_loc = location_data.get('ip_location')
            if ip_loc and ip_loc.get('city'):
                return f"{ip_loc['city']}, {ip_loc['region']}, {ip_loc['country']} (IP-based)"
            
            return self.config.get('fallback_location', 'Face Attendance System')
            
        except Exception as e:
            logger.error(f"Failed to get location summary: {e}")
            return "Face Attendance System (Location error)"
    
    def test_location_service(self):
        """Test the location service"""
        print("\n" + "="*60)
        print("📍 TESTING LOCATION SERVICE")
        print("="*60)
        
        try:
            # Test complete location info
            print("🔍 Getting complete location information...")
            location_data = self.get_complete_location_info()
            
            print("\n📋 COMPLETE LOCATION DATA:")
            print("-" * 40)
            print(location_data['formatted_location'])
            
            print("\n📝 LOCATION SUMMARY:")
            print("-" * 40)
            summary = self.get_location_summary()
            print(f"Summary: {summary}")
            
            print("\n✅ Location service test completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Location service test failed: {e}")
            return False


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    location_service = LocationService()
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        location_service.test_location_service()
    else:
        print("Location Service initialized.")
        print("Run with 'test' argument to test location functionality.")
        
        # Show quick location summary
        summary = location_service.get_location_summary()
        print(f"Current Location: {summary}")
