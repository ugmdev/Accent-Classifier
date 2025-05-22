#!/usr/bin/env python3
import os
import platform
import subprocess
import sys
from pathlib import Path

def print_colored(text, color):
    """Print text in color."""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'end': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['end']}")

def check_ffmpeg():
    """Check if FFmpeg is installed and accessible."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                                capture_output=True, 
                                text=True)
        if result.returncode == 0:
            print_colored("✓ FFmpeg is already installed!", "green")
            print(f"Version info: {result.stdout.splitlines()[0]}")
            return True
    except FileNotFoundError:
        print_colored("✗ FFmpeg not found in PATH", "red")
    return False

def install_ffmpeg_mac():
    """Install FFmpeg on macOS using Homebrew."""
    print_colored("Installing FFmpeg via Homebrew...", "blue")
    
    # Check if Homebrew is installed
    try:
        subprocess.run(['brew', '--version'], check=True, capture_output=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print_colored("Homebrew not found. Installing Homebrew first...", "yellow")
        homebrew_install_cmd = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        try:
            subprocess.run(homebrew_install_cmd, shell=True, check=True)
            print_colored("Homebrew installed successfully.", "green")
        except subprocess.SubprocessError as e:
            print_colored(f"Failed to install Homebrew: {e}", "red")
            return False
    
    # Now install FFmpeg
    try:
        subprocess.run(['brew', 'install', 'ffmpeg'], check=True)
        print_colored("FFmpeg installed successfully!", "green")
        return True
    except subprocess.SubprocessError as e:
        print_colored(f"Failed to install FFmpeg: {e}", "red")
        return False

def install_ffmpeg_linux():
    """Install FFmpeg on Linux using apt or other package managers."""
    print_colored("Installing FFmpeg...", "blue")
    
    # Try apt (Debian, Ubuntu)
    try:
        print_colored("Trying apt package manager...", "yellow")
        subprocess.run(['sudo', 'apt', 'update'], check=True)
        subprocess.run(['sudo', 'apt', 'install', '-y', 'ffmpeg'], check=True)
        print_colored("FFmpeg installed successfully using apt!", "green")
        return True
    except subprocess.SubprocessError:
        print_colored("apt installation failed.", "red")
    
    # Try yum (RHEL, CentOS, Fedora)
    try:
        print_colored("Trying yum package manager...", "yellow")
        subprocess.run(['sudo', 'yum', 'install', '-y', 'ffmpeg'], check=True)
        print_colored("FFmpeg installed successfully using yum!", "green")
        return True
    except subprocess.SubprocessError:
        print_colored("yum installation failed.", "red")
        
    # Try dnf (newer Fedora)
    try:
        print_colored("Trying dnf package manager...", "yellow")
        subprocess.run(['sudo', 'dnf', 'install', '-y', 'ffmpeg'], check=True)
        print_colored("FFmpeg installed successfully using dnf!", "green")
        return True
    except subprocess.SubprocessError:
        print_colored("dnf installation failed.", "red")
    
    print_colored("Unable to install FFmpeg automatically on this Linux distribution.", "red")
    print_colored("Please install manually using your distribution's package manager.", "yellow")
    return False

def install_ffmpeg_windows():
    """Provide instructions for installing FFmpeg on Windows."""
    print_colored("Automated FFmpeg installation on Windows is not supported.", "yellow")
    print_colored("Please follow these manual steps:", "blue")
    print_colored("1. Download FFmpeg from https://ffmpeg.org/download.html", "yellow")
    print_colored("2. Extract the zip file to a location (e.g., C:\\ffmpeg)", "yellow")
    print_colored("3. Add the bin folder (e.g., C:\\ffmpeg\\bin) to your PATH environment variable", "yellow")
    print_colored("4. Restart your terminal/command prompt and run this script again to verify", "yellow")
    
    # Ask if user wants to open the download page
    response = input("Would you like to open the FFmpeg download page? (y/n): ")
    if response.lower() == 'y':
        import webbrowser
        webbrowser.open('https://ffmpeg.org/download.html')
    
    return False

def add_to_path(directory):
    """Add directory to PATH environment variable."""
    if directory not in os.environ['PATH'].split(os.pathsep):
        os.environ['PATH'] = os.pathsep.join([directory, os.environ['PATH']])
        print_colored(f"Added {directory} to PATH for this session", "green")
        print_colored("To make this permanent, you need to add it to your shell profile or system environment variables", "yellow")

def download_ffmpeg_binary():
    """Download precompiled FFmpeg binary as a last resort."""
    import tempfile
    import urllib.request
    import zipfile
    import shutil
    
    print_colored("Attempting to download precompiled FFmpeg binary...", "blue")
    
    system = platform.system().lower()
    if system == 'darwin':
        url = "https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip"
        binary_name = "ffmpeg"
    elif system == 'windows':
        url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        binary_name = "ffmpeg.exe"
    elif system == 'linux':
        if platform.machine() == 'x86_64':
            url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
        else:
            url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-i686-static.tar.xz"
        binary_name = "ffmpeg"
    else:
        print_colored(f"Unsupported system: {system}", "red")
        return False
    
    try:
        # Create app directory if it doesn't exist
        app_dir = Path.home() / ".accent_classifier"
        bin_dir = app_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the file
        with tempfile.NamedTemporaryFile(suffix=url.split('/')[-1], delete=False) as tmp_file:
            print_colored(f"Downloading from {url}...", "yellow")
            urllib.request.urlretrieve(url, tmp_file.name)
            
            # Extract
            if tmp_file.name.endswith('.zip'):
                with zipfile.ZipFile(tmp_file.name, 'r') as zip_ref:
                    # Extract to temporary directory
                    temp_extract = tempfile.mkdtemp()
                    zip_ref.extractall(temp_extract)
                    
                    # Find ffmpeg binary
                    for root, dirs, files in os.walk(temp_extract):
                        if binary_name in files:
                            source = os.path.join(root, binary_name)
                            target = os.path.join(bin_dir, binary_name)
                            shutil.copy2(source, target)
                            os.chmod(target, 0o755)  # Make executable
                            print_colored(f"Installed FFmpeg to {target}", "green")
                            add_to_path(str(bin_dir))
                            return True
            
            print_colored("Failed to extract and locate FFmpeg binary", "red")
            return False
            
    except Exception as e:
        print_colored(f"Error downloading FFmpeg: {e}", "red")
        return False

def main():
    print_colored("FFmpeg Installation Helper", "blue")
    print_colored("=======================", "blue")
    
    # Check if FFmpeg is already installed
    if check_ffmpeg():
        print_colored("Your system is ready to use the Accent Classifier!", "green")
        return
    
    # If not installed, attempt to install based on platform
    system = platform.system()
    if system == 'Darwin':  # macOS
        success = install_ffmpeg_mac()
    elif system == 'Linux':
        success = install_ffmpeg_linux()
    elif system == 'Windows':
        success = install_ffmpeg_windows()
        # Return since we only provide instructions for Windows
        return
    else:
        print_colored(f"Unsupported operating system: {system}", "red")
        success = False
    
    # If standard installation failed, try downloading binary
    if not success:
        print_colored("Standard installation failed, trying alternative method...", "yellow")
        success = download_ffmpeg_binary()
    
    # Verify installation
    if success:
        if check_ffmpeg():
            print_colored("FFmpeg installation verified. Your system is ready to use the Accent Classifier!", "green")
        else:
            print_colored("FFmpeg seems to be installed but is not in PATH.", "yellow")
            print_colored("Please restart your terminal/command prompt or add FFmpeg to your PATH manually.", "yellow")
    else:
        print_colored("Could not install FFmpeg automatically.", "red")
        print_colored("Please install it manually according to your system's instructions.", "yellow")

if __name__ == "__main__":
    main()
