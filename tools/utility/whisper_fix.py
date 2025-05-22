#!/usr/bin/env python3
"""
This script installs and tests the correct version of Whisper.
"""

import subprocess
import sys
import os

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

def fix_whisper_installation():
    print_colored("Fixing Whisper Installation", "blue")
    print_colored("========================", "blue")
    
    # Check if Whisper is installed properly
    try:
        import whisper
        try:
            # Try to access load_model (which is from OpenAI's whisper)
            if hasattr(whisper, 'load_model'):
                print_colored("✓ OpenAI Whisper is already correctly installed!", "green")
                return True
            else:
                print_colored("✗ Wrong Whisper package is installed (missing load_model method)", "red")
        except AttributeError:
            print_colored("✗ Wrong Whisper package is installed (missing load_model method)", "red")
    except ImportError:
        print_colored("✗ Whisper is not installed", "red")
    
    print_colored("Installing the correct OpenAI Whisper package...", "yellow")
    
    # Uninstall any existing whisper packages
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "whisper", "openai-whisper", "whisper-openai"], 
                      check=False)
        print_colored("✓ Removed existing whisper packages", "green")
    except Exception as e:
        print_colored(f"Error removing existing packages: {e}", "red")
    
    # Install from GitHub
    print_colored("Installing OpenAI Whisper from GitHub...", "yellow")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/whisper.git"], 
                      check=True)
        print_colored("✓ Successfully installed OpenAI Whisper!", "green")
    except subprocess.SubprocessError as e:
        print_colored(f"Failed to install from GitHub: {e}", "red")
        print_colored("Trying alternative installation method...", "yellow")
        
        # Try alternative installation method
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "openai-whisper"], 
                          check=True)
            print_colored("✓ Successfully installed OpenAI Whisper via alternative method!", "green")
        except subprocess.SubprocessError as e:
            print_colored(f"Alternative installation failed: {e}", "red")
            return False
    
    # Verify installation
    try:
        import whisper
        if hasattr(whisper, 'load_model'):
            print_colored("✓ Whisper installation verified!", "green")
            print_colored("\nYou can now restart the Accent Classifier app and it should work correctly.", "green")
            return True
        else:
            print_colored("✗ Whisper installation verification failed.", "red")
            return False
    except ImportError:
        print_colored("✗ Failed to import whisper after installation.", "red")
        return False

if __name__ == "__main__":
    success = fix_whisper_installation()
    if not success:
        print_colored("\nWhisper installation failed. Please report this issue.", "red")
        sys.exit(1)
