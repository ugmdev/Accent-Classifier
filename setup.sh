#!/bin/bash
# Setup script for Accent Classifier using Conda
# Installs all required dependencies including system libraries

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== Accent Classifier Setup (Conda Version) =====${NC}"
echo -e "${BLUE}This script will install all required dependencies using Conda${NC}"

# Check if conda is installed and in PATH
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Conda not found in PATH.${NC}"
    echo -e "${YELLOW}Please make sure Conda is installed and properly set up.${NC}"
    echo -e "${YELLOW}You can download Miniconda from: https://docs.conda.io/en/latest/miniconda.html${NC}"
    exit 1
fi

# Check if running in conda environment already
if [ -z "$CONDA_PREFIX" ]; then
    echo -e "${YELLOW}Not running in a conda environment. Please activate your base conda environment first.${NC}"
    echo -e "${YELLOW}Run: conda activate${NC}"
    exit 1
fi

# Detect operating system for system dependencies
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo -e "${YELLOW}Detected macOS operating system${NC}"
    
    # Check if Homebrew is installed (for system packages)
    if ! command -v brew &> /dev/null; then
        echo -e "${YELLOW}Homebrew not found. Installing Homebrew for system packages...${NC}"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH if needed
        if [[ $(uname -m) == 'arm64' ]]; then
            echo -e "${YELLOW}Adding Homebrew to PATH for Apple Silicon...${NC}"
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
    fi
    
    echo -e "${GREEN}Installing system dependencies with Homebrew...${NC}"
    brew install ffmpeg libsndfile sox
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo -e "${YELLOW}Detected Linux operating system${NC}"
    
    # Try to determine distribution
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
    elif [ -f /etc/debian_version ]; then
        DISTRO="debian"
    elif [ -f /etc/redhat-release ]; then
        DISTRO="redhat"
    else
        DISTRO="unknown"
    fi
    
    echo -e "${YELLOW}Detected distribution: $DISTRO${NC}"
    
    case $DISTRO in
        ubuntu|debian)
            echo -e "${GREEN}Installing system dependencies with apt...${NC}"
            sudo apt-get update
            sudo apt-get install -y ffmpeg libsndfile1 libsndfile1-dev libsox-dev sox
            ;;
        fedora)
            echo -e "${GREEN}Installing system dependencies with dnf...${NC}"
            sudo dnf install -y ffmpeg libsndfile libsndfile-devel sox-devel sox
            ;;
        centos|rhel)
            echo -e "${GREEN}Installing system dependencies with yum...${NC}"
            sudo yum install -y ffmpeg libsndfile libsndfile-devel sox-devel sox
            ;;
        *)
            echo -e "${RED}Unsupported Linux distribution. Please install dependencies manually:${NC}"
            echo -e "${YELLOW}Required packages: ffmpeg libsndfile sox${NC}"
            ;;
    esac
    
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    echo -e "${YELLOW}Detected Windows operating system${NC}"
    echo -e "${YELLOW}Installing system dependencies through conda...${NC}"
    # Many system dependencies can be installed via conda on Windows
    conda install -y -c conda-forge ffmpeg libsndfile sox
else
    echo -e "${RED}Unsupported operating system: $OSTYPE${NC}"
    exit 1
fi

# Create conda environment for Accent Classifier
ENV_NAME="accent_classifier"
echo -e "${YELLOW}Creating conda environment: $ENV_NAME${NC}"

# Check if the environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo -e "${YELLOW}Environment $ENV_NAME already exists. Updating it...${NC}"
    conda env update -n $ENV_NAME --file - <<EOF
name: $ENV_NAME
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - numpy>=1.23.0
  - ffmpeg
  - sox
  - libsndfile
  - pytorch>=2.0.0
  - torchaudio>=2.0.0
  - matplotlib>=3.6.0
  - scipy>=1.10.0
  # Additional conda packages
  - librosa>=0.10.0
  - streamlit>=1.25.0
  - pip:
    - openai-whisper>=20230314
    - transformers>=4.28.0
    - moviepy>=1.0.3
    - soundfile>=0.10.0
    - pysoundfile>=0.9.0
    - requests>=2.28.0
EOF
else
    # Create a new environment
    conda create -y -n $ENV_NAME python=3.10
    conda env update -n $ENV_NAME --file - <<EOF
name: $ENV_NAME
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - numpy>=1.23.0
  - ffmpeg
  - sox
  - libsndfile
  - pytorch>=2.0.0
  - torchaudio>=2.0.0
  - matplotlib>=3.6.0
  - scipy>=1.10.0
  # Additional conda packages
  - librosa>=0.10.0
  - streamlit>=1.25.0
  - pip:
    - openai-whisper>=20230314
    - transformers>=4.28.0
    - moviepy>=1.0.3
    - soundfile>=0.10.0
    - pysoundfile>=0.9.0
    - requests>=2.28.0
EOF
fi

echo -e "${GREEN}Conda environment $ENV_NAME created/updated successfully!${NC}"

# Create a simple test script to verify audio setup
echo -e "${YELLOW}Creating audio test script...${NC}"
cat > test_audio_setup.py << EOL
#!/usr/bin/env python3
"""
Test script to verify that audio libraries are working correctly.
This tests all audio loading methods used by the Accent Classifier.
"""

import os
import sys
import numpy as np
import tempfile

def generate_test_audio():
    """Generate a simple test audio file"""
    print("Generating test audio file...")
    
    # Create a simple sine wave
    sample_rate = 16000
    duration = 1  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Save using scipy (most basic, should always work)
    temp_dir = tempfile.gettempdir()
    test_file = os.path.join(temp_dir, "test_audio.wav")
    
    from scipy.io import wavfile
    wavfile.write(test_file, sample_rate, audio)
    print(f"Created test file: {test_file}")
    
    return test_file, audio, sample_rate

def test_soundfile(test_file):
    """Test loading with SoundFile"""
    try:
        import soundfile as sf
        print("\nTesting SoundFile:")
        data, sr = sf.read(test_file)
        print(f"✓ SoundFile works! Shape: {data.shape}, SR: {sr}")
        return True
    except Exception as e:
        print(f"✗ SoundFile failed: {str(e)}")
        return False

def test_torchaudio(test_file):
    """Test loading with torchaudio"""
    try:
        import torchaudio
        import torch
        
        # Test sox_io backend
        print("\nTesting torchaudio with sox_io backend:")
        try:
            torchaudio.set_audio_backend("sox_io")
            waveform, sr = torchaudio.load(test_file)
            print(f"✓ torchaudio (sox_io) works! Shape: {waveform.shape}, SR: {sr}")
        except Exception as e:
            print(f"✗ torchaudio (sox_io) failed: {str(e)}")
        
        # Test soundfile backend
        print("\nTesting torchaudio with soundfile backend:")
        try:
            torchaudio.set_audio_backend("soundfile")
            waveform, sr = torchaudio.load(test_file)
            print(f"✓ torchaudio (soundfile) works! Shape: {waveform.shape}, SR: {sr}")
            return True
        except Exception as e:
            print(f"✗ torchaudio (soundfile) failed: {str(e)}")
            return False
    except Exception as e:
        print(f"✗ torchaudio import failed: {str(e)}")
        return False

def test_librosa(test_file):
    """Test loading with librosa"""
    try:
        import librosa
        print("\nTesting librosa:")
        waveform, sr = librosa.load(test_file, sr=None)
        print(f"✓ librosa works! Shape: {waveform.shape}, SR: {sr}")
        return True
    except Exception as e:
        print(f"✗ librosa failed: {str(e)}")
        return False

def test_ffmpeg():
    """Test if FFmpeg is available"""
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                               capture_output=True, 
                               text=True)
        print("\nTesting FFmpeg:")
        if result.returncode == 0:
            print(f"✓ FFmpeg is installed: {result.stdout.splitlines()[0]}")
            return True
        else:
            print("✗ FFmpeg command failed")
            return False
    except Exception as e:
        print(f"✗ FFmpeg test failed: {str(e)}")
        return False

def test_whisper():
    """Test if whisper is properly installed"""
    try:
        import whisper
        print("\nTesting Whisper:")
        if hasattr(whisper, 'load_model'):
            print(f"✓ Whisper is properly installed")
            return True
        else:
            print("✗ Whisper is installed but lacks the load_model function")
            return False
    except Exception as e:
        print(f"✗ Whisper test failed: {str(e)}")
        return False

def test_transformers():
    """Test if transformers is properly installed"""
    try:
        from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
        print("\nTesting Transformers:")
        print(f"✓ Transformers is properly installed")
        return True
    except Exception as e:
        print(f"✗ Transformers test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("Accent Classifier Audio Setup Test")
    print("=================================\n")
    
    # Generate test audio
    test_file, _, _ = generate_test_audio()
    
    # Run all tests
    tests = [
        ("SoundFile", lambda: test_soundfile(test_file)),
        ("torchaudio", lambda: test_torchaudio(test_file)),
        ("librosa", lambda: test_librosa(test_file)),
        ("FFmpeg", test_ffmpeg),
        ("Whisper", test_whisper),
        ("Transformers", test_transformers)
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
    
    # Print summary
    print("\nTest Summary:")
    print("=============")
    all_passed = True
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        status_color = "\033[92m" if result else "\033[91m"  # Green for pass, red for fail
        print(f"{status_color}{status}\033[0m - {name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n\033[92mAll tests passed! Your audio setup is working correctly.\033[0m")
        return 0
    else:
        print("\n\033[93mSome tests failed. The Accent Classifier app might still work using fallback methods.\033[0m")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOL

# Make script executable
chmod +x test_audio_setup.py

# Create a simple run script
echo -e "${YELLOW}Creating run script...${NC}"
cat > run_classifier.sh << EOL
#!/bin/bash
# Script to run the Accent Classifier

# Activate conda environment
eval "\$(conda shell.bash hook)"
conda activate $ENV_NAME

# Run the Streamlit app
cd "\$(dirname "\$0")"
streamlit run app/app.py
EOL

# Make run script executable
chmod +x run_classifier.sh

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${YELLOW}To activate the conda environment, run:${NC}"
echo -e "${BLUE}conda activate $ENV_NAME${NC}"
echo -e "${YELLOW}To verify your audio setup, run:${NC}"
echo -e "${BLUE}conda activate $ENV_NAME && python test_audio_setup.py${NC}"
echo -e "${YELLOW}To run the Accent Classifier, run:${NC}"
echo -e "${BLUE}./run_classifier.sh${NC}"
echo -e "${YELLOW}or:${NC}"
echo -e "${BLUE}conda activate $ENV_NAME && streamlit run app/app.py${NC}"
