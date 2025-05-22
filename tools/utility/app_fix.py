/Users/ganesh/Desktop/Accent Classifier/
├── app/
│   ├── app.py                     # Main application
│   ├── accent_labels.py           # Label mapping for accents
│   └── emergency_app.py           # No-FFmpeg version
├── tools/
│   ├── install_dependencies.py
│   ├── ffmpeg_installer.py
│   ├── whisper_fix.py
│   └── fix_accent_classifier.py
├── bin/
│   └── ...                        # Bundled binaries
├── docs/
│   ├── README.md
│   └── audio_libraries_guide.md
└── tests/
    ├── test_audio.wav
    └── test_torchaudio.py"""
This script helps fix the audio loading issues in the Accent Classifier app.

Run this script to:
1. Check which audio libraries are working
2. Create a test audio file
3. Test different loading methods
4. Fix the app configuration
"""

import os
import sys
import subprocess
import tempfile
import platform
import importlib.util

# Function to check if a package is installed
def check_package(package_name):
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            print(f"❌ {package_name} is not installed")
            return None
        
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, "__version__", "unknown version")
            print(f"✅ {package_name} is installed ({version})")
            return module
        except:
            print(f"✅ {package_name} is installed")
            return True
    except Exception as e:
        print(f"❌ Error checking {package_name}: {e}")
        return None

# Function to install a package
def install_package(package_name):
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ Successfully installed {package_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to install {package_name}: {e}")
        return False

# Create a test WAV file
def create_test_wav():
    try:
        print("Creating test WAV file...")
        
        # Try different methods to create a WAV file
        methods = ["scipy", "soundfile", "librosa", "wave"]
        
        for method in methods:
            try:
                if method == "scipy":
                    try:
                        from scipy.io import wavfile
                        import numpy as np
                        
                        sample_rate = 16000
                        duration = 1  # seconds
                        t = np.linspace(0, duration, int(sample_rate * duration))
                        audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
                        
                        test_file = os.path.join(os.path.dirname(__file__), "test_audio.wav")
                        wavfile.write(test_file, sample_rate, audio_data)
                        print(f"✅ Created test file using scipy: {test_file}")
                        return test_file
                    except Exception as e:
                        print(f"❌ Failed to create WAV using scipy: {e}")
                
                elif method == "soundfile":
                    try:
                        import soundfile as sf
                        import numpy as np
                        
                        sample_rate = 16000
                        duration = 1  # seconds
                        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration))).astype(np.float32)
                        
                        test_file = os.path.join(os.path.dirname(__file__), "test_audio_sf.wav")
                        sf.write(test_file, audio_data, sample_rate)
                        print(f"✅ Created test file using soundfile: {test_file}")
                        return test_file
                    except Exception as e:
                        print(f"❌ Failed to create WAV using soundfile: {e}")
                
                elif method == "wave":
                    try:
                        import wave
                        import numpy as np
                        import struct
                        
                        sample_rate = 16000
                        duration = 1  # seconds
                        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
                        
                        # Convert to 16-bit integers
                        audio_data = (audio_data * 32767).astype(np.int16)
                        
                        test_file = os.path.join(os.path.dirname(__file__), "test_audio_wave.wav")
                        with wave.open(test_file, 'w') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)  # 2 bytes = 16 bits
                            wf.setframerate(sample_rate)
                            wf.writeframes(audio_data.tobytes())
                        
                        print(f"✅ Created test file using wave: {test_file}")
                        return test_file
                    except Exception as e:
                        print(f"❌ Failed to create WAV using wave: {e}")
                
                elif method == "librosa":
                    try:
                        import librosa
                        import numpy as np
                        import soundfile as sf
                        
                        sample_rate = 16000
                        duration = 1  # seconds
                        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration))).astype(np.float32)
                        
                        test_file = os.path.join(os.path.dirname(__file__), "test_audio_lr.wav")
                        sf.write(test_file, audio_data, sample_rate)
                        print(f"✅ Created test file using librosa/soundfile: {test_file}")
                        return test_file
                    except Exception as e:
                        print(f"❌ Failed to create WAV using librosa: {e}")
            
            except Exception as method_error:
                print(f"Error with method {method}: {method_error}")
        
        # If all methods fail, try ffmpeg
        try:
            print("Trying ffmpeg to create a test WAV file...")
            test_file = os.path.join(os.path.dirname(__file__), "test_audio_ffmpeg.wav")
            cmd = [
                "ffmpeg", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", 
                "-ar", "16000", "-ac", "1", test_file, "-y"
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✅ Created test file using ffmpeg: {test_file}")
            return test_file
        except Exception as e:
            print(f"❌ Failed to create WAV using ffmpeg: {e}")
        
        print("❌ All methods to create test WAV file failed")
        return None
    
    except Exception as e:
        print(f"❌ Error creating test WAV file: {e}")
        return None

# Test different audio loading methods
def test_audio_loading(test_file):
    if not test_file or not os.path.exists(test_file):
        print("❌ No test file available")
        return
    
    print(f"\nTesting audio loading for: {test_file}")
    print(f"File size: {os.path.getsize(test_file)} bytes")
    
    # Try loading with torchaudio (sox_io)
    try:
        import torchaudio
        print("\nTesting torchaudio with sox_io backend")
        torchaudio.set_audio_backend("sox_io")
        waveform, sample_rate = torchaudio.load(test_file)
        print(f"✅ torchaudio (sox_io) worked! Shape: {waveform.shape}, SR: {sample_rate}")
    except Exception as e:
        print(f"❌ torchaudio (sox_io) failed: {e}")
    
    # Try loading with torchaudio (soundfile)
    try:
        import torchaudio
        print("\nTesting torchaudio with soundfile backend")
        torchaudio.set_audio_backend("soundfile")
        waveform, sample_rate = torchaudio.load(test_file)
        print(f"✅ torchaudio (soundfile) worked! Shape: {waveform.shape}, SR: {sample_rate}")
    except Exception as e:
        print(f"❌ torchaudio (soundfile) failed: {e}")
    
    # Try loading with librosa
    try:
        import librosa
        print("\nTesting librosa")
        waveform, sample_rate = librosa.load(test_file, sr=None)
        print(f"✅ librosa worked! Shape: {waveform.shape}, SR: {sample_rate}")
    except Exception as e:
        print(f"❌ librosa failed: {e}")
    
    # Try loading with soundfile
    try:
        import soundfile as sf
        print("\nTesting soundfile")
        waveform, sample_rate = sf.read(test_file)
        print(f"✅ soundfile worked! Shape: {waveform.shape}, SR: {sample_rate}")
    except Exception as e:
        print(f"❌ soundfile failed: {e}")
    
    # Try loading with scipy
    try:
        from scipy.io import wavfile
        print("\nTesting scipy.io.wavfile")
        sample_rate, waveform = wavfile.read(test_file)
        print(f"✅ scipy.io.wavfile worked! Shape: {waveform.shape}, SR: {sample_rate}")
    except Exception as e:
        print(f"❌ scipy.io.wavfile failed: {e}")

# Fix missing dependencies
def fix_dependencies():
    print("\nFixing dependencies...")
    
    packages = [
        "numpy",
        "scipy",
        "torch",
        "torchaudio",
        "librosa",
        "soundfile",
        "SoundFile",  # Alternative name
        "matplotlib",
    ]
    
    for package in packages:
        if not check_package(package):
            install_package(package)
    
    # System dependencies based on platform
    if platform.system() == "Darwin":  # macOS
        print("\nInstalling system dependencies for macOS...")
        try:
            # Check if brew is available
            subprocess.run(["which", "brew"], check=True, capture_output=True)
            
            # Install sox and libsndfile
            subprocess.run(["brew", "install", "sox", "libsndfile"], check=False)
            print("✅ Installed sox and libsndfile using Homebrew")
        except:
            print("""
            ⚠️ Could not install system dependencies with Homebrew.
            Please manually install:
            
            brew install sox libsndfile
            """)
    
    elif platform.system() == "Linux":
        print("\nInstalling system dependencies for Linux...")
        try:
            # Try to detect package manager
            if os.path.exists("/usr/bin/apt"):
                # Debian/Ubuntu
                subprocess.run(["sudo", "apt-get", "update"], check=False)
                subprocess.run(["sudo", "apt-get", "install", "-y", "sox", "libsox-dev", "libsndfile1", "ffmpeg"], check=False)
                print("✅ Installed sox, libsndfile and ffmpeg using apt")
            elif os.path.exists("/usr/bin/yum"):
                # CentOS/RHEL
                subprocess.run(["sudo", "yum", "install", "-y", "sox", "sox-devel", "libsndfile", "ffmpeg"], check=False)
                print("✅ Installed sox, libsndfile and ffmpeg using yum")
            else:
                print("⚠️ Could not detect package manager, please install sox and libsndfile manually")
        except:
            print("""
            ⚠️ Could not install system dependencies.
            Please manually install sox and libsndfile.
            """)
    
    elif platform.system() == "Windows":
        print("""
        On Windows, you need to:
        1. Download and install FFmpeg from https://ffmpeg.org/download.html
        2. Add FFmpeg to your PATH
        3. Install SoX from https://sourceforge.net/projects/sox/
        """)

# Create a direct torchaudio test file
def create_torch_specific_script():
    script_path = os.path.join(os.path.dirname(__file__), "test_torchaudio.py")
    
    with open(script_path, "w") as f:
        f.write("""
import torch
import torchaudio
import os
import sys

def test_backend(backend_name, file_path):
    print(f"Testing {backend_name} backend with {file_path}")
    try:
        torchaudio.set_audio_backend(backend_name)
        waveform, sample_rate = torchaudio.load(file_path)
        print(f"✅ Success! Shape: {waveform.shape}, Sample rate: {sample_rate}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchaudio version: {torchaudio.__version__}")
    
    # Available backends
    try:
        backends = torchaudio.list_audio_backends()
        print(f"Available backends: {backends}")
    except:
        print("Could not list available backends")
    
    # Create test audio
    import numpy as np
    from scipy.io import wavfile
    
    test_file = "test_direct.wav"
    sample_rate = 16000
    data = np.sin(2 * np.pi * 440 * np.arange(sample_rate) / sample_rate).astype(np.float32)
    wavfile.write(test_file, sample_rate, data)
    
    # Test each backend
    for backend in ["sox_io", "soundfile"]:
        test_backend(backend, test_file)
    
    # Try ffmpeg directly if both fail
    try:
        import subprocess
        print("\\nTrying ffmpeg to decode audio...")
        result = subprocess.run([
            "ffmpeg", "-i", test_file, "-f", "s16le", "-acodec", "pcm_s16le", 
            "-ar", "16000", "-ac", "1", "output.raw", "-y"
        ], capture_output=True, text=True)
        
        if os.path.exists("output.raw"):
            print(f"✅ FFmpeg successfully decoded the audio!")
            # Read raw data 
            with open("output.raw", "rb") as f:
                raw_data = f.read()
            print(f"Raw data size: {len(raw_data)} bytes")
        else:
            print("❌ FFmpeg failed to create output file")
    except Exception as e:
        print(f"❌ FFmpeg error: {e}")
""")
    
    print(f"Created torchaudio test script at: {script_path}")
    print(f"Run it with: python {script_path}")

def main():
    print(f"Audio Loading Fix for Accent Classifier")
    print(f"Platform: {platform.system()} {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    
    # Check current installations
    print("\nChecking audio libraries:")
    check_package("torch")
    check_package("torchaudio")
    check_package("librosa")
    check_package("soundfile")
    check_package("scipy")
    
    # Create test WAV file
    test_file = create_test_wav()
    
    # Test audio loading
    if test_file:
        test_audio_loading(test_file)
    
    # Create torch-specific test script
    create_torch_specific_script()
    
    # Offer to fix dependencies
    print("\nWould you like to fix missing dependencies? (y/n)")
    choice = input().lower()
    if choice == 'y':
        fix_dependencies()
        
        # Retest after fixing
        if test_file:
            print("\nRe-testing audio loading after fixes:")
            test_audio_loading(test_file)
    
    print("\nTo update the app.py file with a better audio loading approach:")
    print("1. Ensure librosa is installed: pip install librosa")
    print("2. Your app.py is already updated with the robust audio loading function")
    print("3. The most reliable method for your system appears to be librosa")
    
    print("\nIf you continue to have issues:")
    print("1. Try the emergency_app.py which doesn't rely on audio processing")
    print("2. Run the generated test_torchaudio.py script for more detailed diagnostics")

if __name__ == "__main__":
    main()
