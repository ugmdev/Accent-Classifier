#!/usr/bin/env python3
"""
Direct libsndfile installer for Accent Classifier
Downloads prebuilt libsndfile binaries for your system and installs them where SoundFile can find them.
"""

import os
import sys
import platform
import tempfile
import shutil
import subprocess
from pathlib import Path
import urllib.request
import tarfile
import zipfile
import importlib.util

def print_colored(message, color):
    """Print colored messages to the terminal."""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'end': '\033[0m'
    }
    print(f"{colors.get(color, '')}{message}{colors['end']}")

def find_soundfile_dir():
    """Find the soundfile package directory."""
    try:
        spec = importlib.util.find_spec('soundfile')
        if spec is None:
            return None
        
        package_dir = Path(spec.origin).parent
        print_colored(f"Found soundfile package at: {package_dir}", "green")
        return package_dir
    except Exception as e:
        print_colored(f"Error finding soundfile package: {e}", "red")
        return None

def download_file(url, local_filename):
    """Download a file with progress reporting."""
    print_colored(f"Downloading {url}...", "blue")
    
    try:
        with urllib.request.urlopen(url) as response:
            file_size = int(response.info().get('Content-Length', 0))
            downloaded = 0
            chunk_size = 8192
            
            with open(local_filename, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if file_size > 0:
                        progress = int(50 * downloaded / file_size)
                        sys.stdout.write("\r[%s%s] %d%%" % ('=' * progress, ' ' * (50 - progress), int(100 * downloaded / file_size)))
                        sys.stdout.flush()
            
            print("\nDownload complete!")
            return True
    except Exception as e:
        print_colored(f"Download failed: {e}", "red")
        return False

def extract_archive(archive_path, extract_dir):
    """Extract an archive (zip or tar)."""
    try:
        print_colored(f"Extracting {archive_path}...", "blue")
        
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_dir)
        elif archive_path.endswith('.tar.xz'):
            with tarfile.open(archive_path, 'r:xz') as tar_ref:
                tar_ref.extractall(extract_dir)
        elif archive_path.endswith('.tar'):
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            print_colored(f"Unsupported archive format: {archive_path}", "red")
            return False
        
        print_colored("Extraction complete!", "green")
        return True
    except Exception as e:
        print_colored(f"Extraction failed: {e}", "red")
        return False

def find_library_in_dir(directory, pattern):
    """Find a library file in a directory tree."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if pattern in file:
                return os.path.join(root, file)
    return None

def install_libsndfile():
    """Download and install libsndfile for the current platform."""
    system = platform.system()
    machine = platform.machine()
    
    print_colored(f"Installing libsndfile for {system} {machine}", "blue")
    
    # Create temp directory for downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Find download URL for current platform
        if system == "Darwin":  # macOS
            print_colored("Detected macOS", "blue")
            
            if "arm64" in machine:
                print_colored("Detected Apple Silicon (arm64)", "blue")
                # For Apple Silicon, we might need to use a different approach
                # First try using Homebrew
                try:
                    result = subprocess.run(['brew', 'install', 'libsndfile'], 
                                           check=True, capture_output=True, text=True)
                    print_colored("Installed libsndfile via Homebrew", "green")
                    return True
                except:
                    print_colored("Failed to install via Homebrew, trying direct download...", "yellow")
            
            # Try downloading prebuilt binary
            url = "https://github.com/libsndfile/libsndfile/releases/download/1.1.0/libsndfile-1.1.0-macos.zip"
            download_path = temp_dir / "libsndfile.zip"
            
            if not download_file(url, download_path):
                return False
            
            # Extract archive
            extract_dir = temp_dir / "extracted"
            extract_dir.mkdir(exist_ok=True)
            
            if not extract_archive(download_path, extract_dir):
                return False
            
            # Find the library
            lib_path = find_library_in_dir(extract_dir, "libsndfile.dylib")
            if not lib_path:
                print_colored("Could not find libsndfile.dylib in the extracted files", "red")
                return False
            
        elif system == "Linux":
            print_colored("Detected Linux", "blue")
            
            # Determine architecture
            if "x86_64" in machine:
                arch = "x86_64"
            elif "aarch64" in machine or "arm64" in machine:
                arch = "arm64"
            else:
                arch = "x86"
            
            # Try downloading prebuilt binary
            url = f"https://github.com/libsndfile/libsndfile/releases/download/1.1.0/libsndfile-1.1.0-linux-{arch}.tar.gz"
            download_path = temp_dir / "libsndfile.tar.gz"
            
            if not download_file(url, download_path):
                return False
            
            # Extract archive
            extract_dir = temp_dir / "extracted"
            extract_dir.mkdir(exist_ok=True)
            
            if not extract_archive(download_path, extract_dir):
                return False
            
            # Find the library
            lib_path = find_library_in_dir(extract_dir, "libsndfile.so")
            if not lib_path:
                print_colored("Could not find libsndfile.so in the extracted files", "red")
                return False
            
        elif system == "Windows":
            print_colored("Detected Windows", "blue")
            
            # Determine architecture
            if "AMD64" in machine or "x86_64" in machine:
                arch = "win64"
            else:
                arch = "win32"
            
            # Try downloading prebuilt binary
            url = f"https://github.com/libsndfile/libsndfile/releases/download/1.1.0/libsndfile-1.1.0-{arch}.zip"
            download_path = temp_dir / "libsndfile.zip"
            
            if not download_file(url, download_path):
                return False
            
            # Extract archive
            extract_dir = temp_dir / "extracted"
            extract_dir.mkdir(exist_ok=True)
            
            if not extract_archive(download_path, extract_dir):
                return False
            
            # Find the library
            lib_path = find_library_in_dir(extract_dir, "sndfile.dll")
            if not lib_path:
                print_colored("Could not find sndfile.dll in the extracted files", "red")
                return False
            
        else:
            print_colored(f"Unsupported system: {system}", "red")
            return False
        
        # Find soundfile package directory
        sf_dir = find_soundfile_dir()
        if not sf_dir:
            print_colored("Failed to find soundfile package directory", "red")
            try:
                # Try installing soundfile
                subprocess.run([sys.executable, "-m", "pip", "install", "soundfile"], check=True)
                sf_dir = find_soundfile_dir()
                if not sf_dir:
                    raise Exception("Still couldn't find soundfile package directory after installation")
            except Exception as e:
                print_colored(f"Error installing soundfile: {e}", "red")
                return False
        
        # Make sure _soundfile_data directory exists
        data_dir = sf_dir / "_soundfile_data"
        data_dir.mkdir(exist_ok=True)
        
        # Copy library to soundfile data directory
        target_name = os.path.basename(lib_path)
        if system == "Darwin":
            target_name = "libsndfile.dylib"
        elif system == "Linux":
            target_name = "libsndfile.so"
        elif system == "Windows":
            target_name = "sndfile.dll"
        
        target_path = data_dir / target_name
        
        try:
            shutil.copy2(lib_path, target_path)
            print_colored(f"Copied {lib_path} to {target_path}", "green")
            
            # On macOS and Linux, make sure the library is executable
            if system != "Windows":
                target_path.chmod(0o755)
            
            print_colored("libsndfile installation complete!", "green")
            return True
        except Exception as e:
            print_colored(f"Failed to copy library: {e}", "red")
            return False

def test_soundfile():
    """Test if soundfile is working properly."""
    print_colored("\nTesting soundfile installation...", "blue")
    
    try:
        # Try to import soundfile
        import soundfile
        print_colored(f"soundfile version: {soundfile.__version__}", "green")
        
        # Create a test audio file
        import numpy as np
        sample_rate = 16000
        duration = 0.5  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        test_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Save test file
        test_file = "test_soundfile.wav"
        soundfile.write(test_file, test_data, sample_rate)
        print_colored(f"Created test file: {test_file}", "green")
        
        # Read test file
        data, sr = soundfile.read(test_file)
        print_colored(f"Read test file: shape={data.shape}, sample_rate={sr}", "green")
        
        # Clean up
        os.unlink(test_file)
        
        print_colored("soundfile test passed!", "green")
        return True
    except Exception as e:
        print_colored(f"soundfile test failed: {e}", "red")
        return False

def main():
    print_colored("LibSndFile Direct Installer", "blue")
    print_colored("=========================", "blue")
    
    # Try to import soundfile first to see if it works
    try:
        import soundfile
        print_colored("soundfile is already installed", "green")
        
        # Test if it works
        if test_soundfile():
            print_colored("soundfile is working correctly!", "green")
            print_colored("No need to install libsndfile.", "green")
            return 0
        else:
            print_colored("soundfile is installed but not working correctly.", "yellow")
            print_colored("Will attempt to install libsndfile...", "yellow")
    except ImportError:
        print_colored("soundfile is not installed. Will install it...", "yellow")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "soundfile"], check=True)
            print_colored("soundfile installed successfully", "green")
        except Exception as e:
            print_colored(f"Failed to install soundfile: {e}", "red")
            return 1
    
    # Install libsndfile
    if install_libsndfile():
        # Test if it works now
        if test_soundfile():
            print_colored("libsndfile installed and working correctly!", "green")
            return 0
        else:
            print_colored("libsndfile installed but soundfile still not working correctly.", "red")
            print_colored("Please try using a different audio backend in the app.", "yellow")
            return 1
    else:
        print_colored("Failed to install libsndfile.", "red")
        print_colored("Please try using a different audio backend in the app.", "yellow")
        return 1

if __name__ == "__main__":
    sys.exit(main())
