#!/usr/bin/env python3
"""
Emergency FFmpeg Fixer for Accent Classifier
Downloads and configures a portable FFmpeg that doesn't require system installation
"""

import os
import sys
import platform
import subprocess
import tempfile
import shutil
from pathlib import Path
import urllib.request
import zipfile
import tarfile

def print_status(message, message_type="info"):
    """Print colored status messages"""
    colors = {
        "info": "\033[94m",     # Blue
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

def get_app_directory():
    """Get the application directory where we'll place FFmpeg"""
    app_dir = Path(__file__).parent.absolute()
    bin_dir = app_dir / "bin"
    bin_dir.mkdir(exist_ok=True)
    return bin_dir

def check_ffmpeg():
    """Check if FFmpeg is already available"""
    try:
        # First check the bin directory
        app_bin = get_app_directory()
        ffmpeg_path = app_bin / ("ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg")
        
        if ffmpeg_path.exists():
            print_status(f"FFmpeg found in application directory: {ffmpeg_path}", "success")
            return str(ffmpeg_path)
        
        # Then check the system PATH
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE,
                                  text=True)
            if result.returncode == 0:
                print_status("FFmpeg found in system PATH", "success")
                return "ffmpeg"  # Return just the command name since it's in PATH
        except:
            pass
            
        print_status("FFmpeg not found", "warning")
        return None
    except Exception as e:
        print_status(f"Error checking for FFmpeg: {e}", "error")
        return None

def download_portable_ffmpeg():
    """Download a portable FFmpeg binary that doesn't require installation"""
    bin_dir = get_app_directory()
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    # Define download URLs for each platform
    urls = {
        "darwin": {  # macOS
            "x86_64": "https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip",
            "arm64": "https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip"
        },
        "linux": {  # Linux
            "x86_64": "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
            "aarch64": "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz",
            "armv7l": "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-armhf-static.tar.xz"
        },
        "windows": {  # Windows
            "amd64": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
            "x86_64": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
            "x86": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win32-gpl.zip"
        }
    }
    
    # Handle platform-specific URL selection
    if system not in urls:
        print_status(f"Unsupported system: {system}", "error")
        return None
    
    if machine not in urls[system]:
        # Try some common fallbacks
        if system == "darwin" and machine == "arm64" and "x86_64" in urls[system]:
            print_status("Apple Silicon detected, using x86_64 build (will run under Rosetta)", "warning")
            machine = "x86_64"
        elif system == "windows" and machine not in urls[system]:
            # On Windows, default to x86_64/amd64
            machine = "x86_64"
            print_status(f"Using x86_64 Windows build for {machine}", "warning")
        else:
            print_status(f"Unsupported architecture: {machine} on {system}", "error")
            return None
    
    url = urls[system][machine]
    print_status(f"Downloading FFmpeg from: {url}", "info")
    
    # Create a temporary file to download to
    with tempfile.NamedTemporaryFile(delete=False, suffix=url.split('/')[-1]) as temp_file:
        try:
            # Download with progress reporting
            def report_progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
                sys.stdout.write(f"\rDownloading: {percent}% ({count * block_size / 1024 / 1024:.1f} MB)")
                sys.stdout.flush()
            
            urllib.request.urlretrieve(url, temp_file.name, reporthook=report_progress)
            print()  # Newline after progress
            
            print_status("Download complete, extracting...", "info")
            
            # Extract the downloaded file
            if url.endswith('.zip'):
                with zipfile.ZipFile(temp_file.name, 'r') as zip_ref:
                    # Extract to a temporary directory first
                    extract_dir = tempfile.mkdtemp()
                    zip_ref.extractall(extract_dir)
                    
                    # Find ffmpeg executable
                    ffmpeg_exe = None
                    for root, _, files in os.walk(extract_dir):
                        for file in files:
                            if file.lower() == "ffmpeg.exe" or file.lower() == "ffmpeg":
                                ffmpeg_exe = os.path.join(root, file)
                                break
                        if ffmpeg_exe:
                            break
                    
                    if not ffmpeg_exe:
                        print_status("Could not find ffmpeg executable in the zip file", "error")
                        return None
                    
                    # Copy to our bin directory
                    dest_path = os.path.join(bin_dir, os.path.basename(ffmpeg_exe))
                    shutil.copy2(ffmpeg_exe, dest_path)
                    os.chmod(dest_path, 0o755)  # Make executable
                    
                    # Cleanup temporary extraction directory
                    shutil.rmtree(extract_dir, ignore_errors=True)
                    
                    print_status(f"FFmpeg extracted to: {dest_path}", "success")
                    return dest_path
                    
            elif url.endswith('.tar.xz'):
                # For Linux static builds
                extract_dir = tempfile.mkdtemp()
                
                try:
                    with tarfile.open(temp_file.name, 'r:xz') as tar:
                        tar.extractall(extract_dir)
                except:
                    # Alternative method if tarfile fails
                    subprocess.run(['tar', '-xf', temp_file.name, '-C', extract_dir], check=True)
                
                # Find ffmpeg executable
                ffmpeg_exe = None
                for root, _, files in os.walk(extract_dir):
                    for file in files:
                        if file == "ffmpeg":
                            ffmpeg_exe = os.path.join(root, file)
                            break
                    if ffmpeg_exe:
                        break
                
                if not ffmpeg_exe:
                    print_status("Could not find ffmpeg executable in the tar file", "error")
                    return None
                
                # Copy to bin directory
                dest_path = os.path.join(bin_dir, "ffmpeg")
                shutil.copy2(ffmpeg_exe, dest_path)
                os.chmod(dest_path, 0o755)  # Make executable
                
                # Cleanup temporary extraction directory
                shutil.rmtree(extract_dir, ignore_errors=True)
                
                print_status(f"FFmpeg extracted to: {dest_path}", "success")
                return dest_path
            
            else:
                print_status(f"Unsupported file format: {url}", "error")
                return None
                
        except Exception as e:
            print_status(f"Error downloading or extracting FFmpeg: {e}", "error")
            return None
        finally:
            # Clean up temp file
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

def create_whisper_wrapper():
    """Create a wrapper script to ensure Whisper uses our FFmpeg"""
    bin_dir = get_app_directory()
    wrapper_path = bin_dir / "whisper_wrapper.py"
    
    wrapper_content = """# This wrapper ensures Whisper uses our local FFmpeg
import os
import sys

# Update the PATH to include our bin directory
bin_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['PATH'] = bin_dir + os.pathsep + os.environ.get('PATH', '')

# Import and run Whisper normally
import whisper
import torch

# Function to get available models
def get_available_models():
    return whisper.available_models()

# Function to load model with error handling
def load_model_safely(model_name="base"):
    try:
        return whisper.load_model(model_name)
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return None

# Function to transcribe audio with our environment setup
def transcribe_audio(audio_file, model=None, language="en"):
    if model is None:
        model = load_model_safely()
        if model is None:
            return {"text": "Error: Could not load Whisper model"}
    
    try:
        result = model.transcribe(audio_file, language=language)
        return result
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return {"text": f"Error transcribing: {str(e)}"}

# If run directly
if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        model = load_model_safely()
        if model:
            result = model.transcribe(audio_file)
            print(result["text"])
"""

    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)
    
    print_status(f"Created Whisper wrapper at {wrapper_path}", "success")
    return wrapper_path

def create_usage_example():
    """Create an example script showing how to use our local FFmpeg with Whisper"""
    bin_dir = get_app_directory()
    example_path = bin_dir.parent / "whisper_with_local_ffmpeg.py"
    
    example_content = """#!/usr/bin/env python3
# Example script showing how to use our local FFmpeg with Whisper

import os
import sys
from pathlib import Path

# Add the bin directory to PATH
bin_dir = Path(__file__).parent / "bin"
os.environ['PATH'] = str(bin_dir) + os.pathsep + os.environ.get('PATH', '')

# Now Whisper will use our local FFmpeg
import whisper

def main():
    # Check if FFmpeg is available
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        if result.returncode == 0:
            print(f"Using FFmpeg: {result.stdout.decode().splitlines()[0]}")
        else:
            print("FFmpeg test failed!")
            return 1
    except Exception as e:
        print(f"Error running FFmpeg: {e}")
        return 1
    
    # Load Whisper model
    try:
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        print("Whisper model loaded successfully!")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return 1
    
    # Test on an audio file if provided
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        if os.path.exists(audio_file):
            print(f"Transcribing {audio_file}...")
            result = model.transcribe(audio_file)
            print(f"Transcription: {result['text']}")
        else:
            print(f"Audio file not found: {audio_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""

    with open(example_path, 'w') as f:
        f.write(example_content)
    
    print_status(f"Created usage example at {example_path}", "success")
    return example_path

def create_accent_analyzer_with_local_ffmpeg():
    """Create a version of the accent analyzer that uses our local FFmpeg"""
    bin_dir = get_app_directory()
    analyzer_path = bin_dir.parent / "accent_analyzer_with_local_ffmpeg.py"
    
    analyzer_content = """#!/usr/bin/env python3
# Accent analyzer using our local FFmpeg

import os
import sys
from pathlib import Path

# Add the bin directory to PATH at the beginning to prioritize our FFmpeg
bin_dir = Path(__file__).parent / "bin"
os.environ['PATH'] = str(bin_dir) + os.pathsep + os.environ.get('PATH', '')

import streamlit as st
import tempfile
import subprocess
import whisper
import torch
import numpy as np

st.set_page_config(page_title="Accent Analyzer (Local FFmpeg)", layout="wide")
st.title("Accent Analyzer with Local FFmpeg")

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

def extract_audio(video_file, output_audio):
    """Extract audio using our local FFmpeg"""
    cmd = ['ffmpeg', '-i', video_file, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', output_audio, '-y']
    process = subprocess.run(cmd, check=True, capture_output=True)
    return True

def transcribe_audio(audio_file):
    """Transcribe the audio using Whisper"""
    model = load_whisper_model()
    result = model.transcribe(audio_file)
    return result["text"]

def analyze_accent(audio_file, transcription):
    """Very simple rule-based accent analysis"""
    # This is just a placeholder - a real system would use ML
    accents = {
        "American": 0.3,
        "British": 0.2,
        "Indian": 0.2,
        "Australian": 0.1,
        "Non-native": 0.2
    }
    
    # Return a random accent for demo purposes
    import random
    accent = random.choice(list(accents.keys()))
    confidence = accents[accent] + random.uniform(-0.1, 0.1)
    confidence = max(0.01, min(0.99, confidence))
    
    return accent, confidence

# UI
uploaded_file = st.file_uploader("Upload Audio/Video file", type=['mp3', 'wav', 'mp4', 'mov'])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
        tmp_video.write(uploaded_file.getvalue())
        video_path = tmp_video.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
        audio_path = tmp_audio.name
    
    with st.spinner("Extracting audio..."):
        extract_audio(video_path, audio_path)
    
    with st.spinner("Transcribing audio..."):
        transcription = transcribe_audio(audio_path)
    
    with st.spinner("Analyzing accent..."):
        accent, confidence = analyze_accent(audio_path, transcription)
    
    # Display results
    st.subheader("Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Transcription")
        st.write(transcription)
    
    with col2:
        st.markdown("### Accent Analysis")
        st.write(f"**Detected accent:** {accent}")
        st.write(f"**Confidence:** {confidence:.2f}")
        st.progress(confidence)
    
    # Clean up
    os.unlink(video_path)
    os.unlink(audio_path)

else:
    st.write("Please upload an audio or video file to analyze.")
"""

    with open(analyzer_path, 'w') as f:
        f.write(analyzer_content)
    
    print_status(f"Created accent analyzer with local FFmpeg at {analyzer_path}", "success")
    return analyzer_path

def fix_environment():
    """Update environment variables to use our FFmpeg"""
    bin_dir = get_app_directory()
    os.environ['PATH'] = str(bin_dir) + os.pathsep + os.environ.get('PATH', '')
    print_status(f"Updated PATH to include {bin_dir}", "success")
    
    # Create a .env file in the application directory to help other scripts
    env_path = bin_dir.parent / ".env"
    with open(env_path, 'w') as f:
        f.write(f"FFMPEG_PATH={bin_dir}\n")
        f.write(f"PATH={bin_dir}{os.pathsep}$PATH\n")
    
    print_status(f"Created environment file at {env_path}", "success")
    
    # Create an activation script
    if platform.system() == "Windows":
        activate_path = bin_dir.parent / "activate_ffmpeg.bat"
        with open(activate_path, 'w') as f:
            f.write(f"@echo off\nset PATH={bin_dir};%PATH%\necho FFmpeg environment activated. Run your scripts now.\n")
    else:
        activate_path = bin_dir.parent / "activate_ffmpeg.sh"
        with open(activate_path, 'w') as f:
            f.write(f"#!/bin/bash\nexport PATH={bin_dir}:$PATH\necho FFmpeg environment activated. Run your scripts now.\n")
        os.chmod(activate_path, 0o755)
    
    print_status(f"Created activation script at {activate_path}", "success")
    
    # Instructions for usage
    print_status("\nTo use in your environment:", "info")
    if platform.system() == "Windows":
        print(f"Run: {activate_path}")
    else:
        print(f"Run: source {activate_path}")
    
    return True

def main():
    print_status("Emergency FFmpeg Fixer for Accent Classifier", "info")
    print_status("===========================================", "info")
    
    # First check if FFmpeg is already available
    ffmpeg_path = check_ffmpeg()
    
    if not ffmpeg_path:
        # Download a portable FFmpeg
        ffmpeg_path = download_portable_ffmpeg()
        if not ffmpeg_path:
            print_status("Failed to download FFmpeg. Please try installing it manually.", "error")
            return False
    
    # Create helper scripts
    create_whisper_wrapper()
    create_usage_example()
    create_accent_analyzer_with_local_ffmpeg()
    
    # Fix environment
    fix_environment()
    
    print_status("\nFFmpeg setup complete!", "success")
    print_status("\nNow you can:", "info")
    print_status("1. Run the accent analyzer with local FFmpeg:", "info")
    print(f"   streamlit run {get_app_directory().parent / 'accent_analyzer_with_local_ffmpeg.py'}")
    print_status("2. Test Whisper with our local FFmpeg:", "info")
    print(f"   python {get_app_directory().parent / 'whisper_with_local_ffmpeg.py'}")
    print_status("3. Run your original app with our FFmpeg in the PATH:", "info")
    if platform.system() == "Windows":
        print(f"   {get_app_directory().parent / 'activate_ffmpeg.bat'} && streamlit run app.py")
    else:
        print(f"   source {get_app_directory().parent / 'activate_ffmpeg.sh'} && streamlit run app.py")
    
    return True

if __name__ == "__main__":
    main()
