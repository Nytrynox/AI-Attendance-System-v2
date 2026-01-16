# src/gui/camera_selection_window.py

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import threading
import time
import requests
import socket
from PIL import Image, ImageTk


class CameraSelectionWindow:
    """Camera selection window for choosing between laptop camera and mobile camera (DroidCam)"""
    
    def __init__(self, master, callback=None):
        self.master = master
        self.callback = callback
        self.master.title("Camera Selection - Face Attendance System")
        self.master.geometry("900x700")
        self.master.resizable(True, True)
        
        # Camera variables
        self.selected_camera_type = None
        self.selected_camera_source = None
        self.test_cap = None
        self.preview_running = False
        self.preview_thread = None
        
        # DroidCam settings
        self.droidcam_ip = tk.StringVar(value="192.168.1.100")
        self.droidcam_port = tk.StringVar(value="4747")
        
        # Available cameras
        self.available_cameras = []
        
        self.create_ui()
        self.scan_available_cameras()
        
    def create_ui(self):
        """Create the camera selection interface"""
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, text="Select Camera Source", 
                              font=("Arial", 18, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Create notebook for different camera types
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Laptop Camera Tab
        self.laptop_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.laptop_frame, text="💻 Laptop Camera")
        
        # Mobile Camera Tab
        self.mobile_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.mobile_frame, text="📱 Mobile Camera (DroidCam)")
        
        # Create laptop camera interface
        self.create_laptop_camera_interface()
        
        # Create mobile camera interface
        self.create_mobile_camera_interface()
        
        # Preview frame
        preview_frame = tk.LabelFrame(main_frame, text="Camera Preview", padx=10, pady=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.preview_label = tk.Label(preview_frame, text="No camera selected", 
                                     width=80, height=20, bg="black", fg="white")
        self.preview_label.pack(expand=True)
        
        # Control buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.test_btn = tk.Button(button_frame, text="Test Selected Camera", 
                                 command=self.test_camera, bg="lightblue", 
                                 font=("Arial", 12))
        self.test_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_test_btn = tk.Button(button_frame, text="Stop Test", 
                                      command=self.stop_camera_test, bg="lightcoral", 
                                      font=("Arial", 12), state=tk.DISABLED)
        self.stop_test_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.use_camera_btn = tk.Button(button_frame, text="Use This Camera", 
                                       command=self.use_selected_camera, bg="lightgreen", 
                                       font=("Arial", 12, "bold"), state=tk.DISABLED)
        self.use_camera_btn.pack(side=tk.RIGHT)
        
        # Instructions
        instructions = """
Instructions:
1. Select either Laptop Camera or Mobile Camera tab
2. For Laptop Camera: Choose from detected cameras
3. For Mobile Camera: Install DroidCam on your phone and enter the IP address
4. Click 'Test Selected Camera' to preview
5. Click 'Use This Camera' to proceed with attendance system
        """
        
        info_label = tk.Label(main_frame, text=instructions, justify=tk.LEFT, 
                             font=("Arial", 9), fg="gray")
        info_label.pack(pady=(10, 0))
        
    def create_laptop_camera_interface(self):
        """Create interface for laptop camera selection"""
        info_frame = tk.Frame(self.laptop_frame)
        info_frame.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(info_frame, text="Available Laptop Cameras:", 
                font=("Arial", 12, "bold")).pack(anchor=tk.W)
        
        tk.Label(info_frame, text="Select a camera from the detected devices below:", 
                font=("Arial", 10)).pack(anchor=tk.W, pady=(5, 10))
        
        # Camera list frame
        list_frame = tk.Frame(self.laptop_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Scrollable listbox for cameras
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.camera_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                        font=("Arial", 11), height=8)
        self.camera_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.camera_listbox.bind('<<ListboxSelect>>', self.on_laptop_camera_select)
        
        scrollbar.config(command=self.camera_listbox.yview)
        
        # Refresh button
        refresh_btn = tk.Button(self.laptop_frame, text="🔄 Refresh Cameras", 
                               command=self.scan_available_cameras, bg="lightyellow")
        refresh_btn.pack(pady=10)
        
    def create_mobile_camera_interface(self):
        """Create interface for mobile camera (DroidCam) setup"""
        info_frame = tk.Frame(self.mobile_frame)
        info_frame.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(info_frame, text="Mobile Camera Setup (DroidCam):", 
                font=("Arial", 12, "bold")).pack(anchor=tk.W)
        
        instructions = """
Steps to use mobile camera:
1. Install 'DroidCam' app on your mobile phone
2. Connect your phone to the same WiFi network as this computer
3. Open DroidCam app and note the IP address shown
4. Enter the IP address below and click 'Test Selected Camera'
        """
        
        tk.Label(info_frame, text=instructions, justify=tk.LEFT, 
                font=("Arial", 10)).pack(anchor=tk.W, pady=(5, 15))
        
        # IP and Port configuration
        config_frame = tk.LabelFrame(self.mobile_frame, text="DroidCam Configuration", 
                                    padx=15, pady=15)
        config_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # IP Address
        ip_frame = tk.Frame(config_frame)
        ip_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(ip_frame, text="IP Address:", width=12, anchor=tk.W).pack(side=tk.LEFT)
        ip_entry = tk.Entry(ip_frame, textvariable=self.droidcam_ip, font=("Arial", 11))
        ip_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Port
        port_frame = tk.Frame(config_frame)
        port_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(port_frame, text="Port:", width=12, anchor=tk.W).pack(side=tk.LEFT)
        port_entry = tk.Entry(port_frame, textvariable=self.droidcam_port, font=("Arial", 11))
        port_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Test connection button
        test_connection_btn = tk.Button(config_frame, text="🔗 Test Connection", 
                                       command=self.test_droidcam_connection, 
                                       bg="lightcyan")
        test_connection_btn.pack(pady=10)
        
        # Auto-detect button
        auto_detect_btn = tk.Button(self.mobile_frame, text="🔍 Auto-detect DroidCam", 
                                   command=self.auto_detect_droidcam, bg="lightgreen")
        auto_detect_btn.pack(pady=10)
        
    def scan_available_cameras(self):
        """Scan for available laptop cameras"""
        self.camera_listbox.delete(0, tk.END)
        self.available_cameras = []
        
        self.camera_listbox.insert(tk.END, "Scanning for cameras...")
        self.master.update()
        
        # Test camera indices 0-5
        for i in range(6):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.available_cameras.append(i)
                cap.release()
            except Exception as e:
                print(f"Error testing camera {i}: {e}")
        
        # Update listbox
        self.camera_listbox.delete(0, tk.END)
        
        if self.available_cameras:
            for i, camera_idx in enumerate(self.available_cameras):
                self.camera_listbox.insert(tk.END, f"Camera {camera_idx} - Default Camera" if camera_idx == 0 else f"Camera {camera_idx} - External Camera")
        else:
            self.camera_listbox.insert(tk.END, "No cameras detected")
            
    def on_laptop_camera_select(self, event):
        """Handle laptop camera selection"""
        selection = self.camera_listbox.curselection()
        if selection and self.available_cameras:
            camera_idx = self.available_cameras[selection[0]]
            self.selected_camera_type = "laptop"
            self.selected_camera_source = camera_idx
            print(f"Selected laptop camera: {camera_idx}")
            
    def test_droidcam_connection(self):
        """Test DroidCam connection"""
        ip = self.droidcam_ip.get().strip()
        port = self.droidcam_port.get().strip()
        
        if not ip or not port:
            messagebox.showerror("Error", "Please enter both IP address and port")
            return
            
        try:
            # Test HTTP connection to DroidCam
            url = f"http://{ip}:{port}/video"
            response = requests.get(url, timeout=5, stream=True)
            
            if response.status_code == 200:
                messagebox.showinfo("Success", "DroidCam connection successful!")
                self.selected_camera_type = "mobile"
                self.selected_camera_source = url
                return True
            else:
                messagebox.showerror("Error", f"DroidCam connection failed. Status: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            messagebox.showerror("Error", f"Cannot connect to DroidCam:\n{str(e)}")
            return False
            
    def auto_detect_droidcam(self):
        """Auto-detect DroidCam on local network"""
        messagebox.showinfo("Auto-detect", "Scanning local network for DroidCam...\nThis may take a few moments.")
        
        def scan_network():
            # Get local network range
            try:
                # Get local IP
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
                
                # Get network base (e.g., 192.168.1.)
                network_base = '.'.join(local_ip.split('.')[:-1]) + '.'
                  # Scan common IPs
                for i in range(1, 255):
                    ip = network_base + str(i)
                    try:
                        url = f"http://{ip}:4747/video"
                        response = requests.get(url, timeout=1)
                        if response.status_code == 200:
                            self.droidcam_ip.set(ip)
                            messagebox.showinfo("Found", f"DroidCam found at {ip}:4747")
                            return
                    except Exception:
                        continue
                
                messagebox.showwarning("Not Found", "No DroidCam devices found on local network")
                
            except Exception as e:
                messagebox.showerror("Error", f"Network scan failed: {str(e)}")
        
        # Run scan in background thread
        threading.Thread(target=scan_network, daemon=True).start()
        
    def test_camera(self):
        """Test the selected camera"""
        if not self.selected_camera_type or not self.selected_camera_source:
            messagebox.showerror("Error", "Please select a camera first")
            return
            
        if self.preview_running:
            return
            
        try:
            if self.selected_camera_type == "laptop":
                self.test_cap = cv2.VideoCapture(self.selected_camera_source)
                if not self.test_cap.isOpened():
                    messagebox.showerror("Error", "Failed to open laptop camera")
                    return
            elif self.selected_camera_type == "mobile":
                self.test_cap = cv2.VideoCapture(self.selected_camera_source)
                if not self.test_cap.isOpened():
                    messagebox.showerror("Error", "Failed to connect to mobile camera")
                    return
            
            self.preview_running = True
            self.test_btn.config(state=tk.DISABLED)
            self.stop_test_btn.config(state=tk.NORMAL)
            self.use_camera_btn.config(state=tk.NORMAL)
            
            # Start preview in separate thread
            self.preview_thread = threading.Thread(target=self.camera_preview_loop, daemon=True)
            self.preview_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to test camera: {str(e)}")
            
    def camera_preview_loop(self):
        """Camera preview loop"""
        while self.preview_running and self.test_cap:
            try:
                ret, frame = self.test_cap.read()
                if ret:
                    # Resize frame for preview
                    frame = cv2.resize(frame, (640, 480))
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PhotoImage
                    image = Image.fromarray(frame_rgb)
                    photo = ImageTk.PhotoImage(image)
                    
                    # Update preview label
                    self.preview_label.config(image=photo, text="")
                    setattr(self.preview_label, 'image', photo)  # Keep reference to prevent GC
                else:
                    break
                    
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"Preview error: {e}")
                break
                
        self.stop_camera_test()
        
    def stop_camera_test(self):
        """Stop camera test"""
        self.preview_running = False
        
        if self.test_cap:
            self.test_cap.release()
            self.test_cap = None
            
        self.preview_label.config(image="", text="Camera test stopped")
        setattr(self.preview_label, 'image', None)  # Clear reference
        
        self.test_btn.config(state=tk.NORMAL)
        self.stop_test_btn.config(state=tk.DISABLED)
        
    def use_selected_camera(self):
        """Use the selected camera for attendance system"""
        if not self.selected_camera_type or not self.selected_camera_source:
            messagebox.showerror("Error", "Please select and test a camera first")
            return
            
        self.stop_camera_test()
        
        if self.callback:
            camera_info = {
                'type': self.selected_camera_type,
                'source': self.selected_camera_source,
                'description': f"{self.selected_camera_type.title()} Camera ({self.selected_camera_source})"
            }
            self.callback(camera_info)
            
        self.master.destroy()
        
    def on_closing(self):
        """Handle window closing"""
        self.stop_camera_test()
        self.master.destroy()


def main():
    """Test the camera selection window"""
    def camera_selected(camera_info):
        print(f"Selected camera: {camera_info}")
        
    root = tk.Tk()
    app = CameraSelectionWindow(root, callback=camera_selected)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
