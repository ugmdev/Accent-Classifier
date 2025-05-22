#!/usr/bin/env python3
"""
Emergency fix for libsndfile issue in Conda environment
This script directly addresses the specific path mentioned in your error message
"""

import os
import sys
import subprocess
import platform
import tempfile
import shutil
import urllib.request
import zipfile
from pathlib import Path

def print_bold(message):
    print(f"\033[1m{message}\033[0m")

def print_error(message):
    print(f"\033[91m{message}\033[0m")

def print_success(message):
    print(f"\033[92m{message}\033[0m")

def print_info(message):
    print(f"\033[94m{message}\033[0m")

def print_warning(message):
    print(f"\033[93m{message}\033[0m")

# The exact problematic path from your error message
TARGET_PATH = "/opt/anaconda3/envs/accent-env/lib/python3.11/site-packages/_soundfile_data/libsndfile.dylib"

def download_libsndfile_binary():
    """Download a pre-compiled libsndfile binary directly from GitHub."""
    print_info("Downloading libsndfile binary directly from GitHub...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download URL for macOS
        url = "https://github.com/libsndfile/libsndfile/releases/download/1.2.0/libsndfile-1.2.0-macos.zip"
        zip_path = os.path.join(temp_dir, "libsndfile.zip")
        
        # Download the file
        print_info(f"Downloading from {url}...")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            print_error(f"Failed to download: {e}")
            return None
            
        # Extract the zip file
        print_info("Extracting zip file...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        except Exception as e:
            print_error(f"Failed to extract zip file: {e}")
            return None
        
        # Find the libsndfile.dylib file
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(".dylib") and "sndfile" in file:
                    file_path = os.path.join(root, file)
                    print_success(f"Found library: {file_path}")
                    return file_path
        
        print_error("Could not find libsndfile.dylib in the downloaded package")
        return None

def find_system_libsndfile():
    """Find libsndfile on the system using various methods."""
    print_info("Searching for libsndfile on your system...")
    
    # Common paths for libsndfile on macOS
    common_paths = [
        "/usr/local/lib/libsndfile.dylib",
        "/usr/local/lib/libsndfile.1.dylib",
        "/opt/homebrew/lib/libsndfile.dylib",
        "/opt/homebrew/lib/libsndfile.1.dylib"
    ]
    
    # Check each path
    for path in common_paths:
        if os.path.exists(path):
            print_success(f"Found at system path: {path}")
            return path
    
    # Try using Homebrew to locate it
    try:
        result = subprocess.run(["brew", "--prefix", "libsndfile"], 
                               capture_output=True, text=True, check=True)
        brew_prefix = result.stdout.strip()
        for suffix in ["libsndfile.dylib", "libsndfile.1.dylib"]:
            path = os.path.join(brew_prefix, "lib", suffix)
            if os.path.exists(path):
                print_success(f"Found via Homebrew: {path}")
                return path
    except:
        print_warning("Homebrew not available or libsndfile not installed via Homebrew")
    
    # Try to install with Homebrew
    try:
        print_info("Attempting to install libsndfile with Homebrew...")
        subprocess.run(["brew", "install", "libsndfile"], check=True)
        
        # Check common paths again
        for path in common_paths:
            if os.path.exists(path):
                print_success(f"Found after installation: {path}")
                return path
    except:
        print_warning("Failed to install with Homebrew or Homebrew not available")
    
    return None

def try_conda_install_libsndfile():
    """Try to install libsndfile using conda."""
    print_info("Attempting to install libsndfile via conda...")
    
    try:
        subprocess.run(["conda", "install", "-y", "-c", "conda-forge", "libsndfile"], check=True)
        print_success("Successfully installed libsndfile via conda")
        return True
    except Exception as e:
        print_warning(f"Failed to install libsndfile via conda: {e}")
        return False

def main():
    print_bold("\n========================================")
    print_bold("EMERGENCY LIBSNDFILE FIX FOR CONDA")
    print_bold("========================================\n")
    
    print_info(f"Target path: {TARGET_PATH}")
    
    # Create the directory structure
    target_dir = os.path.dirname(TARGET_PATH)
    if not os.path.exists(target_dir):
        print_info(f"Creating directory: {target_dir}")
        try:
            os.makedirs(target_dir, exist_ok=True)
            print_success("Directory created successfully")
        except Exception as e:
            print_error(f"Failed to create directory: {e}")
            print_error("Make sure you have write permissions to this location")
            sys.exit(1)
    else:
        print_info(f"Directory already exists: {target_dir}")
    
    # Step 1: Try to find libsndfile on the system
    source_path = find_system_libsndfile()
    
    # Step 2: If not found, try to install with conda
    if not source_path and try_conda_install_libsndfile():
        # After conda install, try to find it again
        source_path = find_system_libsndfile()
    
    # Step 3: If still not found, download directly
    if not source_path:
        source_path = download_libsndfile_binary()
    
    # Step 4: If we found or downloaded libsndfile, copy it to the target location
    if source_path:
        print_info(f"Copying {source_path} to {TARGET_PATH}")
        try:
            # Remove existing file if it exists
            if os.path.exists(TARGET_PATH):
                os.remove(TARGET_PATH)
            
            # Try creating a symbolic link first
            try:
                os.symlink(source_path, TARGET_PATH)
                print_success("Created symbolic link successfully")
            except:
                # If symlink fails, copy the file
                shutil.copy2(source_path, TARGET_PATH)
                print_success("Copied file successfully")
            
            # Ensure the file is executable
            os.chmod(TARGET_PATH, 0o755)
            
            if os.path.exists(TARGET_PATH):
                print_success("\n✅ SUCCESS: libsndfile.dylib has been installed!")
                print_info("\nTry running the app again. If you still encounter issues, try the fallback method:")
                print_info("   conda install -c conda-forge python-soundfile librosa")
                print_info("   conda install -c conda-forge torchaudio -c pytorch")
            else:
                print_error("\n❌ FAILURE: Could not verify that libsndfile.dylib was installed correctly")
        except Exception as e:
            print_error(f"\n❌ ERROR during file copy/link: {e}")
            print_error("Make sure you have write permissions to this location")
            
            # Try direct conda package reinstall as a fallback
            print_warning("\nTrying fallback method: reinstalling conda packages...")
            try:
                subprocess.run(["conda", "install", "-y", "-c", "conda-forge", "python-soundfile", "libsndfile"], check=True)
                print_success("Reinstalled python-soundfile and libsndfile via conda-forge")
            except:
                print_warning("Fallback method failed")
    else:
        print_error("\n❌ Could not find or install libsndfile")
        print_error("You may need to install it manually")
        
        # Offer an extreme fallback - create a dummy file
        print_warning("\nWould you like to create a dummy file as a last resort? (y/n)")
        print_warning("This may silence the error but audio functions might not work.")
        response = input().lower()
        if response == 'y':
            # Create an empty file at the target location
            with open(TARGET_PATH, 'wb') as f:
                f.write(b'\x00' * 100)  # Write some dummy bytes
            print_warning(f"Created dummy file at {TARGET_PATH}")
            print_warning("The app may run but audio functionality might not work properly")
    
    # Extra diagnostics
    print_bold("\n========================================")
    print_bold("DIAGNOSTICS")
    print_bold("========================================")
    
    try:
        print_info("\nChecking Python packages:")
        
        # Check conda environment
        print_info("\nConda environment:")
        subprocess.run(["conda", "list", "libsndfile"], check=False)
        subprocess.run(["conda", "list", "python-soundfile"], check=False)
        
        # Check Python packages
        print_info("\nPython packages:")
        subprocess.run([sys.executable, "-m", "pip", "list"], check=False)
        
        # Try importing soundfile
        print_info("\nTrying to import soundfile:")
        try:
            import soundfile
            print_success(f"Successfully imported soundfile (version: {soundfile.__version__})")
        except Exception as e:
            print_error(f"Failed to import soundfile: {e}")
            
        # Check if the target file exists
        print_info(f"\nChecking if {TARGET_PATH} exists:")
        if os.path.exists(TARGET_PATH):
            print_success(f"File exists, size: {os.path.getsize(TARGET_PATH)} bytes")
        else:
            print_error("File does not exist")
            
        # Print LD_LIBRARY_PATH and DYLD_LIBRARY_PATH
        print_info("\nLibrary paths:")
        print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
        print(f"DYLD_LIBRARY_PATH: {os.environ.get('DYLD_LIBRARY_PATH', 'Not set')}")
    except Exception as e:
        print_error(f"Error during diagnostics: {e}")
    
    print_bold("\n========================================")
    print_info("Process completed. Please try running your app again.")
    print_bold("========================================")

if __name__ == "__main__":
    main()
