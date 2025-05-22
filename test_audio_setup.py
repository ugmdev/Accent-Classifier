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
        ("FFmpeg", test_ffmpeg)
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
