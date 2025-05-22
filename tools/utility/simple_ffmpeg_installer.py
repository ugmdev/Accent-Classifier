#!/usr/bin/env python3
"""
Simple FFmpeg installer script for Accent Classifier
This is a more direct approach to installing FFmpeg, focusing on reliability
"""

import os
import sys
import platform
import subprocess
import tempfile
import shutil
from pathlib import Path

def print_message(message, message_type="info"):
    """Print a formatted message"""
    colors = {
        "info": "\033[94m",    # Blue
        "success": "\033[92m",  # Green
        "warning": "\033[93m",  # Yellow
        "error": "\033[91m",    # Red
        "end": "\033[0m"        # Reset
    }
    
    prefix = {
        "info": "INFO",
        "success": "SUCCESS",
        "warning": "WARNING",
        "error": "ERROR"
    }
    
    print(f"{colors.get(message_type, '')}{prefix.get(message_type, '')}: {message}{colors['end']}")

def run_command(command, check=True, shell=False):
    """Run a command with better error handling"""
    try:
        if shell:
            process = subprocess.run(command, shell=True, check=check, 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True)
        else:
            process = subprocess.run(command, check=check, 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True)
        return process.returncode == 0, process.stdout
    except Exception as e:
        print_message(f"Command failed: {e}", "error")
        return False, str(e)

def is_ffmpeg_installed():
    """Check if FFmpeg is installed and in PATH"""
    success, output = run_command(['ffmpeg', '-version'], check=False)
    if success:
        print_message(f"FFmpeg is already installed! Version info: {output.split('version')[1].split()[0]}", "success")
        return True
    return False

def install_ffmpeg_macos():
    """Install FFmpeg on macOS using Homebrew"""
    print_message("Installing FFmpeg on macOS...", "info")
    
    # Check if Homebrew is installed
    success, _ = run_command(['brew', '--version'], check=False)
    if not success:
        print_message("Homebrew not found. Installing Homebrew...", "info")
        brew_install_cmd = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        success, _ = run_command(brew_install_cmd, shell=True)
        if not success:
            print_message("Failed to install Homebrew. Please install it manually first.", "error")
            print_message("Visit https://brew.sh/ for instructions", "info")
            return False
        
        # After Homebrew installation, we need to make sure it's in the PATH
        print_message("Homebrew installed! Setting up PATH...", "success")
        
        # Add Homebrew to PATH for this session
        if os.path.exists("/opt/homebrew/bin/brew"):
            os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ["PATH"]
            print_message("Added /opt/homebrew/bin to PATH", "info")
        elif os.path.exists("/usr/local/bin/brew"):
            os.environ["PATH"] = "/usr/local/bin:" + os.environ["PATH"]
            print_message("Added /usr/local/bin to PATH", "info")
            
        # Verify Homebrew is now available
        success, _ = run_command(['brew', '--version'], check=False)
        if not success:
            print_message("Homebrew was installed but isn't available in PATH. Please restart your terminal and run this script again.", "error")
            return False
    
    # Install FFmpeg
    print_message("Installing FFmpeg using Homebrew...", "info")
    success, output = run_command(['brew', 'install', 'ffmpeg'])
    if success:
        print_message("FFmpeg installed successfully!", "success")
        return True
    else:
        print_message(f"Error installing FFmpeg: {output}", "error")
        return False

def install_ffmpeg_linux():
    """Install FFmpeg on Linux"""
    print_message("Installing FFmpeg on Linux...", "info")
    
    # Detect package manager
    package_managers = [
        ("apt-get", ["sudo", "apt-get", "update"], ["sudo", "apt-get", "install", "-y", "ffmpeg"]),
        ("apt", ["sudo", "apt", "update"], ["sudo", "apt", "install", "-y", "ffmpeg"]),
        ("yum", ["sudo", "yum", "check-update"], ["sudo", "yum", "install", "-y", "ffmpeg"]),
        ("dnf", ["sudo", "dnf", "check-update"], ["sudo", "dnf", "install", "-y", "ffmpeg"])
    ]
    
    for pm_name, update_cmd, install_cmd in package_managers:
        success, _ = run_command([pm_name, "--version"], check=False)
        if success:
            print_message(f"Using {pm_name} package manager", "info")
            
            # Update repositories
            print_message(f"Updating package repositories with {pm_name}...", "info")
            run_command(update_cmd, check=False)
            
            # Install FFmpeg
            print_message(f"Installing FFmpeg with {pm_name}...", "info")
            success, output = run_command(install_cmd, check=False)
            if success:
                print_message("FFmpeg installed successfully!", "success")
                return True
            else:
                print_message(f"Error installing FFmpeg with {pm_name}: {output}", "error")
    
    print_message("Could not install FFmpeg automatically. No supported package manager found or installation failed.", "error")
    print_message("Please install FFmpeg manually according to your Linux distribution's instructions.", "info")
    return False

def download_static_ffmpeg():
    """Download a pre-compiled static FFmpeg binary as a fallback"""
    print_message("Downloading pre-compiled FFmpeg binary...", "info")
    
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    # URLs for static builds
    urls = {
        "darwin": {
            "x86_64": "https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip",
            "arm64": "https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip"
        },
        "linux": {
            "x86_64": "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
            "i686": "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-i686-static.tar.xz",
            "armv7l": "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-armhf-static.tar.xz",
            "aarch64": "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz"
        },
        "windows": {
            "x86_64": "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
            "x86": "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        }
    }
    
    if system not in urls:
        print_message(f"No pre-compiled FFmpeg available for {system}", "error")
        return False
        
    if machine not in urls[system]:
        # Try a fallback for this system
        if system == "darwin" and "x86_64" in urls[system]:
            # For Mac, we can try x86_64 for Apple Silicon under Rosetta
            machine = "x86_64"
            print_message("Using x86_64 build for Apple Silicon under Rosetta", "warning")
        elif system == "linux" and "x86_64" in urls[system]:
            # For Linux, default to x86_64
            machine = "x86_64"
            print_message(f"No build for {machine}, trying x86_64 instead", "warning")
        else:
            print_message(f"No pre-compiled FFmpeg available for {system} on {machine}", "error")
            return False
    
    download_url = urls[system][machine]
    
    try:
        import urllib.request
        
        # Create a directory for FFmpeg
        ffmpeg_dir = Path.home() / ".accent_classifier" / "bin"
        ffmpeg_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a temporary file to download to
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            print_message(f"Downloading from {download_url}...", "info")
            print_message("This may take a few minutes...", "info")
            
            # Download the file with progress reporting
            def report_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, int(block_num * block_size * 100 / total_size))
                    if percent % 10 == 0:
                        print_message(f"Downloaded: {percent}%", "info")
            
            urllib.request.urlretrieve(download_url, temp_file.name, reporthook=report_progress)
            
            print_message("Download complete! Extracting...", "info")
            
            # Extract based on file type
            if download_url.endswith('.zip'):
                import zipfile
                with zipfile.ZipFile(temp_file.name, 'r') as zip_ref:
                    # Extract to a temporary directory
                    extract_dir = tempfile.mkdtemp()
                    zip_ref.extractall(extract_dir)
                    
                    # Find the ffmpeg executable
                    ffmpeg_path = None
                    for root, dirs, files in os.walk(extract_dir):
                        for file in files:
                            if file == "ffmpeg" or file == "ffmpeg.exe":
                                ffmpeg_path = os.path.join(root, file)
                                break
                        if ffmpeg_path:
                            break
                    
                    if not ffmpeg_path:
                        print_message("Could not find ffmpeg executable in the downloaded archive", "error")
                        return False
                    
                    # Copy to our bin directory
                    target_path = ffmpeg_dir / os.path.basename(ffmpeg_path)
                    shutil.copy2(ffmpeg_path, target_path)
                    os.chmod(target_path, 0o755)  # Make executable
                    
                    print_message(f"FFmpeg installed to {target_path}", "success")
                    
                    # Add to PATH
                    add_to_path(str(ffmpeg_dir))
                    return True
                
            elif download_url.endswith('.tar.xz'):
                import tarfile
                
                # Extract to a temporary directory
                extract_dir = tempfile.mkdtemp()
                
                try:
                    with tarfile.open(temp_file.name, 'r:xz') as tar:
                        tar.extractall(extract_dir)
                except:
                    # Try a different extraction method for older Python versions
                    result = run_command(['tar', '-xf', temp_file.name, '-C', extract_dir])
                    if not result[0]:
                        print_message("Failed to extract tar.xz file", "error")
                        return False
                
                # Find the ffmpeg executable
                ffmpeg_path = None
                for root, dirs, files in os.walk(extract_dir):
                    for file in files:
                        if file == "ffmpeg":
                            ffmpeg_path = os.path.join(root, file)
                            break
                    if ffmpeg_path:
                        break
                
                if not ffmpeg_path:
                    print_message("Could not find ffmpeg executable in the downloaded archive", "error")
                    return False
                
                # Copy to our bin directory
                target_path = ffmpeg_dir / "ffmpeg"
                shutil.copy2(ffmpeg_path, target_path)
                os.chmod(target_path, 0o755)  # Make executable
                
                print_message(f"FFmpeg installed to {target_path}", "success")
                
                # Add to PATH
                add_to_path(str(ffmpeg_dir))
                return True
            
            else:
                print_message(f"Unsupported file format: {download_url}", "error")
                return False
            
    except Exception as e:
        print_message(f"Error downloading FFmpeg: {e}", "error")
        return False

def add_to_path(directory):
    """Add directory to PATH environment variable"""
    print_message(f"Adding {directory} to PATH...", "info")
    
    # Add to PATH for this session
    os.environ["PATH"] = f"{directory}:{os.environ['PATH']}"
    
    # Detect shell to suggest permanent PATH addition
    shell = os.environ.get("SHELL", "")
    home = str(Path.home())
    
    if "bash" in shell:
        rc_file = os.path.join(home, ".bashrc")
        print_message(f"To permanently add FFmpeg to PATH, add this line to {rc_file}:", "info")
        print(f"export PATH=\"{directory}:$PATH\"")
    elif "zsh" in shell:
        rc_file = os.path.join(home, ".zshrc")
        print_message(f"To permanently add FFmpeg to PATH, add this line to {rc_file}:", "info")
        print(f"export PATH=\"{directory}:$PATH\"")
    elif platform.system() == "Windows":
        print_message("To permanently add FFmpeg to PATH on Windows:", "info")
        print("1. Right-click on 'This PC' or 'My Computer' and select 'Properties'")
        print("2. Click on 'Advanced system settings'")
        print("3. Click on 'Environment Variables'")
        print("4. Under 'System variables', select 'Path' and click 'Edit'")
        print("5. Click 'New' and add the directory containing ffmpeg.exe")
    else:
        print_message(f"To permanently add FFmpeg to PATH, add this directory to your shell startup file: {directory}", "info")

def install_ffmpeg():
    """Main function to install FFmpeg"""
    print_message("==== FFmpeg Installer for Accent Classifier ====", "info")
    
    # Check if FFmpeg is already installed
    if is_ffmpeg_installed():
        return True
    
    # Install based on platform
    system = platform.system()
    success = False
    
    if system == "Darwin":  # macOS
        success = install_ffmpeg_macos()
    elif system == "Linux":
        success = install_ffmpeg_linux()
    elif system == "Windows":
        print_message("Automated installation on Windows is not supported.", "warning")
        print_message("Please download FFmpeg from https://ffmpeg.org/download.html", "info")
        print_message("And add it to your PATH environment variable.", "info")
        return False
    else:
        print_message(f"Unsupported platform: {system}", "error")
        return False
    
    # If regular installation failed, try downloading a static binary
    if not success:
        print_message("Standard installation failed. Trying to download pre-compiled binary...", "warning")
        success = download_static_ffmpeg()
    
    # Verify installation
    if success:
        if is_ffmpeg_installed():
            print_message("FFmpeg installation verified and working!", "success")
            print_message("You can now return to the Accent Classifier app.", "success")
            return True
        else:
            print_message("FFmpeg was installed but is not in PATH", "warning")
            print_message("Please restart your terminal/shell and try again", "info")
            return False
    else:
        print_message("FFmpeg installation failed", "error")
        print_message("Please install FFmpeg manually from https://ffmpeg.org/download.html", "info")
        return False

def check_python_dependencies():
    """Check and install required Python dependencies"""
    try:
        import urllib.request
        import shutil
        import tempfile
        return True
    except ImportError as e:
        print_message(f"Missing Python dependency: {e}", "error")
        print_message("Please install required packages: pip install urllib3 shutil tempfile", "info")
        return False

if __name__ == "__main__":
    if check_python_dependencies():
        success = install_ffmpeg()
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
