"""
Test script for torchaudio backends.
"""

import torch
import torchaudio
import os
import sys
import numpy as np
from scipy.io import wavfile

def test_backend(backend_name, file_path):
    """Test a specific torchaudio backend with the given audio file."""
    print(f"Testing {backend_name} backend with {file_path}")
    try:
        torchaudio.set_audio_backend(backend_name)
        waveform, sample_rate = torchaudio.load(file_path)
        print(f"✅ Success! Shape: {waveform.shape}, Sample rate: {sample_rate}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def main():
    """Main test function."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchaudio version: {torchaudio.__version__}")
    
    # Available backends
    try:
        backends = torchaudio.list_audio_backends()
        print(f"Available backends: {backends}")
    except:
        print("Could not list available backends")
    
    # Create test audio
    test_file = os.path.join(os.path.dirname(__file__), "test_direct.wav")
    sample_rate = 16000
    data = np.sin(2 * np.pi * 440 * np.arange(sample_rate) / sample_rate).astype(np.float32)
    wavfile.write(test_file, sample_rate, data)
    print(f"Created test file: {test_file}")
    
    # Test existing audio file
    test_audio = os.path.join(os.path.dirname(__file__), "test_audio.wav")
    if os.path.exists(test_audio):
        print(f"Found existing test audio: {test_audio}")
    else:
        test_audio = test_file
    
    # Test each backend
    for backend in ["sox_io", "soundfile"]:
        test_backend(backend, test_audio)
    
    # Try ffmpeg directly if both fail
    try:
        import subprocess
        print("\nTrying ffmpeg to decode audio...")
        output_raw = os.path.join(os.path.dirname(__file__), "output.raw")
        result = subprocess.run([
            "ffmpeg", "-i", test_audio, "-f", "s16le", "-acodec", "pcm_s16le", 
            "-ar", "16000", "-ac", "1", output_raw, "-y"
        ], capture_output=True, text=True)
        
        if os.path.exists(output_raw):
            print(f"✅ FFmpeg successfully decoded the audio!")
            # Read raw data 
            with open(output_raw, "rb") as f:
                raw_data = f.read()
            print(f"Raw data size: {len(raw_data)} bytes")
        else:
            print("❌ FFmpeg failed to create output file")
    except Exception as e:
        print(f"❌ FFmpeg error: {e}")

if __name__ == "__main__":
    main()
