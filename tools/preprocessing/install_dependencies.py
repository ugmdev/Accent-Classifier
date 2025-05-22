#!/usr/bin/env python3
"""
Dependency installer for Accent Classifier
Installs required audio libraries for proper functionality
"""

import subprocess
import sys
import os
import platform
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

def check_pip():
    """Ensure pip is available and updated."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
        return True
    except:
        print_colored("pip not found or not working. Please install pip first.", "red")
        return False

def install_package(package_name, extras=None):
    """Install a Python package with error handling."""
    package_spec = package_name
    if extras:
        package_spec = f"{package_name}[{extras}]"
    
    print_colored(f"Installing {package_spec}...", "blue")
    
    try:
        cmd = [sys.executable, "-m", "pip", "install", package_spec]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print_colored(f"✓ Successfully installed {package_spec}", "green")
        return True
    except subprocess.CalledProcessError as e:
        print_colored(f"Error installing {package_spec}: {e}", "red")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")
        return False

def install_audio_backends():
    """Install the required audio processing libraries."""
    print_colored("Installing audio backends and dependencies...", "blue")
    
    # Install SoundFile
    print_colored("Installing SoundFile and its dependencies...", "yellow")
    
    # Install libsndfile system dependency for SoundFile
    system = platform.system()
    if system == "Darwin":  # macOS
        try:
            result = subprocess.run(["brew", "install", "libsndfile"], check=False, capture_output=True, text=True)
            if result.returncode == 0:
                print_colored("✓ Installed libsndfile via Homebrew", "green")
            else:
                print_colored("Warning: Failed to install libsndfile via Homebrew", "yellow")
                print_colored("You may need to install it manually: brew install libsndfile", "yellow")
        except:
            print_colored("Homebrew not found or error running brew", "yellow")
    elif system == "Linux":
        try:
            # Try apt first (Debian/Ubuntu)
            result = subprocess.run(["sudo", "apt-get", "install", "-y", "libsndfile1"], check=False, capture_output=True, text=True)
            if result.returncode == 0:
                print_colored("✓ Installed libsndfile via apt", "green")
            else:
                # Try yum (Fedora/CentOS/RHEL)
                result = subprocess.run(["sudo", "yum", "install", "-y", "libsndfile"], check=False, capture_output=True, text=True)
                if result.returncode == 0:
                    print_colored("✓ Installed libsndfile via yum", "green")
                else:
                    print_colored("Warning: Failed to install libsndfile via system package manager", "yellow")
                    print_colored("You may need to install it manually according to your Linux distribution", "yellow")
        except:
            print_colored("Error running system package manager", "yellow")
    
    # Install Python packages
    packages = [
        ("numpy", None),
        ("scipy", None),
        ("soundfile", None),
        ("librosa", None),
        ("torch", None),
        ("torchaudio", None),
        ("transformers", None),
        ("moviepy", None),
        ("matplotlib", None),
        ("streamlit", None),
        ("whisper", None),  # This is a placeholder; we'll install it properly later
        ("SpeechRecognition", None),
    ]
    
    for package, extras in packages:
        if package == "whisper":
            # Skip normal installation for whisper
            continue
        install_package(package, extras)
    
    # Install proper whisper version
    print_colored("Installing OpenAI Whisper...", "blue")
    try:
        # Uninstall any existing whisper packages first
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "whisper", "openai-whisper"], check=False)
        
        # Install the correct one
        subprocess.run([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/whisper.git"], check=True)
        print_colored("✓ Successfully installed OpenAI Whisper", "green")
    except subprocess.CalledProcessError as e:
        print_colored(f"Error installing OpenAI Whisper: {e}", "red")
        print_colored("Trying alternative installation...", "yellow")
        install_package("openai-whisper", None)
    
    # Create a test audio file for torchaudio to use
    create_test_audio_file()
    
    print_colored("\nAudio backends installation completed!", "green")
    print_colored("You can now run the accent classifier app again.", "green")

def create_test_audio_file():
    """Create a test WAV file for audio backend testing."""
    try:
        import numpy as np
        from scipy.io import wavfile
        
        print_colored("Creating test audio file...", "blue")
        
        # Get the current script directory
        script_dir = Path(__file__).parent.absolute()
        test_file = script_dir / "test.wav"
        
        # Generate a simple sine wave
        sample_rate = 16000
        duration = 1  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        tone = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Normalize and convert to int16
        tone = (tone * 32767).astype(np.int16)
        
        # Write to file
        wavfile.write(test_file, sample_rate, tone)
        
        if test_file.exists():
            print_colored(f"✓ Test audio file created at {test_file}", "green")
        else:
            print_colored("Failed to create test audio file", "red")
    except Exception as e:
        print_colored(f"Error creating test audio file: {e}", "red")

def verify_installation():
    """Verify that all required packages are installed and working."""
    print_colored("\nVerifying installations...", "blue")
    
    # Check for all required packages
    packages_to_check = [
        "numpy", "scipy", "soundfile", "torch", "torchaudio", 
        "librosa", "transformers", "moviepy", "whisper"
    ]
    
    all_successful = True
    
    for package in packages_to_check:
        try:
            if package == "whisper":
                # Special check for OpenAI Whisper
                import whisper
                if hasattr(whisper, 'load_model'):
                    print_colored(f"✓ {package} correctly installed (OpenAI version)", "green")
                else:
                    print_colored(f"✗ {package} is installed but not the correct version", "red")
                    all_successful = False
            else:
                # Generic import check
                __import__(package)
                print_colored(f"✓ {package} installed", "green")
        except ImportError:
            print_colored(f"✗ {package} not installed or not working", "red")
            all_successful = False
    
    # Verify audio backends
    try:
        import torchaudio
        
        # Check SoundFile backend
        try:
            torchaudio.set_audio_backend("soundfile")
            print_colored("✓ torchaudio soundfile backend working", "green")
        except:
            print_colored("✗ torchaudio soundfile backend not working", "red")
            all_successful = False
        
        # Check SoX backend
        try:
            torchaudio.set_audio_backend("sox_io")
            print_colored("✓ torchaudio sox_io backend working", "green")
        except:
            print_colored("✗ torchaudio sox_io backend not working", "red")
            # This is not critical as we have multiple fallbacks
            print_colored("(This is not critical as other backends are available)", "yellow")
    except Exception as e:
        print_colored(f"Error checking torchaudio backends: {e}", "red")
        all_successful = False
    
    if all_successful:
        print_colored("\n✓ All packages verified successfully!", "green")
    else:
        print_colored("\n⚠ Some packages could not be verified.", "yellow")
        print_colored("The app may still work with fallback methods, but for optimal performance, please address the issues above.", "yellow")
    
    return all_successful

def main():
    print_colored("Audio Dependencies Installer for Accent Classifier", "blue")
    print_colored("==============================================", "blue")
    
    if not check_pip():
        sys.exit(1)
    
    install_audio_backends()
    verify_installation()
    
    print_colored("\nSetup complete! You can now run the Accent Classifier app.", "green")

if __name__ == "__main__":
    main()
