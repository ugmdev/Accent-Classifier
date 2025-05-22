#!/usr/bin/env python3
"""
Fix script for Accent Classifier
Diagnoses and fixes common issues with the app
"""

import os
import sys
import subprocess
import platform
import importlib
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    line = "=" * len(text)
    print(f"\n{line}")
    print(text)
    print(f"{line}\n")

def print_result(text, success=True):
    """Print a result with colored status"""
    status = "✅" if success else "❌"
    color = "\033[92m" if success else "\033[91m"
    end_color = "\033[0m"
    print(f"{color}{status} {text}{end_color}")

def check_command(command):
    """Check if a command is available"""
    try:
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except:
        return False

def check_python_module(module_name):
    """Check if a Python module is installed"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False
    
def fix_audio_data():
    """Create a test audio file for testing"""
    try:
        import numpy as np
        from scipy.io import wavfile
        
        # Get the current directory
        current_dir = Path(__file__).parent.absolute()
        test_file = current_dir / "test.wav"
        
        # Generate a simple sine wave
        sample_rate = 16000
        duration = 1  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        tone = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Normalize and convert to int16
        tone = (tone * 32767).astype(np.int16)
        
        # Write to file
        wavfile.write(test_file, sample_rate, tone)
        
        return test_file.exists()
    except Exception as e:
        print(f"Error creating test audio: {e}")
        return False

def run_diagnostic():
    """Run a diagnostic on the accent classifier setup"""
    print_header("Accent Classifier Diagnostic")
    
    # Check Python version
    python_version = sys.version.split()[0]
    print_result(f"Python version: {python_version}", True)
    
    # Check operating system
    system = platform.system()
    print_result(f"Operating system: {system}", True)
    
    # Check essential dependencies
    dependencies = {
        "streamlit": "Streamlit UI framework",
        "numpy": "Numerical processing",
        "torch": "PyTorch for ML models",
        "moviepy": "Video processing",
        "transformers": "Transformer models",
        "scipy": "Scientific computing"
    }
    
    print_header("Checking Essential Dependencies")
    all_deps_ok = True
    for dep, desc in dependencies.items():
        has_dep = check_python_module(dep)
        print_result(f"{dep} ({desc})", has_dep)
        if not has_dep:
            all_deps_ok = False
    
    # Check optional dependencies
    opt_dependencies = {
        "whisper": "OpenAI Whisper for transcription",
        "librosa": "Advanced audio processing",
        "soundfile": "Audio file loading",
        "matplotlib": "Plotting and visualization",
        "torchaudio": "PyTorch audio utilities"
    }
    
    print_header("Checking Optional Dependencies")
    missing_opt_deps = []
    for dep, desc in opt_dependencies.items():
        has_dep = check_python_module(dep)
        print_result(f"{dep} ({desc})", has_dep)
        if not has_dep:
            missing_opt_deps.append(dep)
    
    # Check FFmpeg
    print_header("Checking External Dependencies")
    has_ffmpeg = check_command(["ffmpeg", "-version"])
    print_result("FFmpeg (audio/video processing)", has_ffmpeg)
    
    # Check test audio file
    print_header("Checking Audio Test Data")
    has_test_audio = os.path.exists(os.path.join(os.path.dirname(__file__), "test.wav"))
    if not has_test_audio:
        print_result("Creating test audio file", False)
        has_test_audio = fix_audio_data()
    print_result("Test audio file", has_test_audio)
    
    # Summary and next steps
    print_header("Summary")
    
    if all_deps_ok and has_ffmpeg and len(missing_opt_deps) == 0:
        print_result("All dependencies are installed and working properly!", True)
        print("\nYou can now run the accent classifier with:")
        print("   streamlit run app.py")
    else:
        if not all_deps_ok:
            print_result("Missing essential dependencies", False)
            print("\nTo install essential dependencies, run:")
            print("   pip install streamlit numpy torch moviepy transformers scipy")
        
        if not has_ffmpeg:
            print_result("FFmpeg is not installed or not in PATH", False)
            print("\nTo install FFmpeg, run:")
            print("   python simple_ffmpeg_installer.py")
            print("\nIf you can't install FFmpeg, you can use the no-FFmpeg version:")
            print("   streamlit run accent_classifier_no_ffmpeg.py")
        
        if missing_opt_deps:
            print_result(f"Missing optional dependencies: {', '.join(missing_opt_deps)}", False)
            print("\nTo install optional dependencies, run:")
            print(f"   pip install {' '.join(missing_opt_deps)}")
            
            if "whisper" in missing_opt_deps:
                print("\nFor whisper specifically, run:")
                print("   python whisper_fix.py")
    
    return all_deps_ok and has_ffmpeg and len(missing_opt_deps) == 0

def fix_setup():
    """Try to fix common setup issues"""
    print_header("Fixing Accent Classifier Setup")
    
    # Fix essential dependencies
    print("Installing essential dependencies...")
    essential_cmd = [sys.executable, "-m", "pip", "install", "streamlit", "numpy", "torch", "moviepy", "transformers", "scipy"]
    subprocess.run(essential_cmd, check=False)
    
    # Fix optional dependencies
    print("\nInstalling optional dependencies...")
    optional_cmd = [sys.executable, "-m", "pip", "install", "librosa", "soundfile", "matplotlib", "torchaudio"]
    subprocess.run(optional_cmd, check=False)
    
    # Fix Whisper
    try:
        print("\nInstalling OpenAI Whisper...")
        uninstall_cmd = [sys.executable, "-m", "pip", "uninstall", "-y", "whisper", "whisper-openai", "openai-whisper"]
        subprocess.run(uninstall_cmd, check=False)
        install_cmd = [sys.executable, "-m", "pip", "install", "git+https://github.com/openai/whisper.git"]
        subprocess.run(install_cmd, check=False)
    except:
        print("Failed to install Whisper. You can run whisper_fix.py later.")
    
    # Create test audio file
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "test.wav")):
        print("\nCreating test audio file...")
        fix_audio_data()
    
    # Recommend FFmpeg installation
    print("\nFor FFmpeg installation, please run:")
    print("   python simple_ffmpeg_installer.py")
    
    print("\nSetup fixes applied. Run the diagnostic again to verify:")
    print("   python fix_accent_classifier.py --diagnose")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--diagnose":
        run_diagnostic()
    else:
        fix_setup()
        print("\nWould you like to install FFmpeg now? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            print("\nRunning FFmpeg installer...")
            subprocess.run([sys.executable, "simple_ffmpeg_installer.py"], check=False)
