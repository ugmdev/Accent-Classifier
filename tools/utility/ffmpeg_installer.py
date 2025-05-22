"""
Helper script to install FFmpeg automatically.
This can be run independently or imported from other scripts.
"""

import os
import sys
import platform
import subprocess
import tempfile
import shutil
import zipfile
import tarfile
import requests
from pathlib import Path
import time

def check_ffmpeg_installed():
    """Check if FFmpeg is installed and in PATH"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.SubprocessError):
        return False

def install_ffmpeg_mac():
    """Install FFmpeg on macOS using Homebrew"""
    print("Attempting to install FFmpeg using Homebrew...")
    try:
        # Check if Homebrew is installed
        brew_check = subprocess.run(['brew', '--version'], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
        
        if brew_check.returncode != 0:
            print("Homebrew not found. Installing Homebrew first...")
            install_cmd = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
            os.system(install_cmd)
            print("Please follow any instructions from Homebrew to complete setup.")
            print("Once Homebrew is set up, run this script again to install FFmpeg.")
            return False
            
        # Install FFmpeg using Homebrew
        print("Installing FFmpeg...")
        subprocess.run(['brew', 'install', 'ffmpeg'], check=True)
        print("FFmpeg installed successfully!")
        return True
        
    except Exception as e:
        print(f"Error installing FFmpeg: {str(e)}")
        print("\nManual installation instructions:")
        print("1. Install Homebrew: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        print("2. Install FFmpeg: brew install ffmpeg")
        return False

def install_ffmpeg_linux():
    """Install FFmpeg on Linux using apt"""
    print("Attempting to install FFmpeg using apt...")
    try:
        # Update package list
        subprocess.run(['sudo', 'apt-get', 'update'], check=True)
        
        # Install FFmpeg
        subprocess.run(['sudo', 'apt-get', 'install', '-y', 'ffmpeg'], check=True)
        print("FFmpeg installed successfully!")
        return True
        
    except Exception as e:
        print(f"Error installing FFmpeg: {str(e)}")
        print("\nManual installation instructions:")
        print("1. Update packages: sudo apt-get update")
        print("2. Install FFmpeg: sudo apt-get install -y ffmpeg")
        return False

def download_ffmpeg_windows():
    """Download FFmpeg for Windows"""
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "ffmpeg.zip")
        
        # URL for FFmpeg builds
        ffmpeg_url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        
        print(f"Downloading FFmpeg from {ffmpeg_url}...")
        response = requests.get(ffmpeg_url, stream=True)
        response.raise_for_status()
        
        # Download with progress reporting
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress = min(int(50 * downloaded / total_size), 50)
                    sys.stdout.write("\r[%s%s] %d%%" % ('=' * progress, ' ' * (50 - progress), int(100 * downloaded / total_size)))
                    sys.stdout.flush()
        
        print("\nDownload complete! Extracting files...")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find the bin directory
        bin_dir = None
        for root, dirs, files in os.walk(temp_dir):
            if 'ffmpeg.exe' in files:
                bin_dir = root
                break
        
        if not bin_dir:
            print("Could not find FFmpeg executable in the downloaded package.")
            return None
            
        return bin_dir
        
    except Exception as e:
        print(f"Error downloading FFmpeg: {str(e)}")
        return None

def install_ffmpeg_windows(bin_dir):
    """Install FFmpeg on Windows"""
    try:
        # Get user's home directory
        home_dir = str(Path.home())
        ffmpeg_dir = os.path.join(home_dir, "ffmpeg")
        
        # Create directory if it doesn't exist
        os.makedirs(ffmpeg_dir, exist_ok=True)
        
        print(f"Copying FFmpeg files to {ffmpeg_dir}...")
        
        # Copy FFmpeg files
        for file in os.listdir(bin_dir):
            if file.endswith('.exe'):
                shutil.copy2(os.path.join(bin_dir, file), ffmpeg_dir)
        
        # Add to PATH (for current session)
        os.environ["PATH"] += os.pathsep + ffmpeg_dir
        
        # For permanent PATH addition, we need to modify Windows registry or use setx
        # This will only affect new command prompts, not the current one
        subprocess.run(['setx', 'PATH', f"%PATH%;{ffmpeg_dir}"], check=True)
        
        print("FFmpeg installed successfully!")
        print(f"FFmpeg has been installed to: {ffmpeg_dir}")
        print("This directory has been added to your PATH environment variable.")
        print("You may need to restart your command prompt or application for the changes to take effect.")
        return True
        
    except Exception as e:
        print(f"Error installing FFmpeg: {str(e)}")
        if bin_dir:
            print(f"\nManual installation instructions:")
            print(f"1. Copy all .exe files from {bin_dir} to a directory like C:\\ffmpeg\\bin")
            print(f"2. Add that directory to your PATH environment variable")
        return False

def install_ffmpeg():
    """Detect OS and install FFmpeg accordingly"""
    system = platform.system().lower()
    
    print(f"Detected operating system: {system}")
    
    if check_ffmpeg_installed():
        print("FFmpeg is already installed and in PATH!")
        return True
    
    if system == "darwin":
        return install_ffmpeg_mac()
    elif system == "linux":
        return install_ffmpeg_linux()
    elif system == "windows":
        bin_dir = download_ffmpeg_windows()
        if bin_dir:
            return install_ffmpeg_windows(bin_dir)
        return False
    else:
        print(f"Unsupported operating system: {system}")
        print("Please install FFmpeg manually from https://ffmpeg.org/download.html")
        return False

if __name__ == "__main__":
    print("===== FFmpeg Installer =====")
    success = install_ffmpeg()
    
    if success:
        print("\nVerifying installation...")
        time.sleep(1)  # Give a moment for PATH to update
        
        if check_ffmpeg_installed():
            print("✅ FFmpeg is now installed and in PATH!")
            try:
                result = subprocess.run(['ffmpeg', '-version'], 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE,
                                       text=True)
                version = result.stdout.split('\n')[0]
                print(f"Version information: {version}")
            except:
                pass
        else:
            print("⚠️ FFmpeg is installed but not in PATH.")
            print("You may need to restart your terminal/command prompt or computer.")
    else:
        print("\n❌ FFmpeg installation failed or was canceled.")
        print("Please try installing FFmpeg manually from https://ffmpeg.org/download.html")
