# Understanding Audio Libraries in Python

This guide explains the different audio libraries used in the Accent Classifier application and how they work together.

## Librosa

Librosa is a Python library for audio and music analysis. It provides the building blocks necessary to create music information retrieval systems.

### How Librosa Works:

1. **Audio Loading**: Librosa uses soundfile or audioread as backends to load audio files into NumPy arrays.
   ```python
   import librosa
   # Load an audio file (returns a numpy array and sample rate)
   waveform, sample_rate = librosa.load('audio_file.wav', sr=None)  # sr=None preserves original sample rate
   ```

2. **Key Features**:
   - Audio processing (resampling, trimming silence)
   - Feature extraction (spectrograms, MFCCs, chroma)
   - Music analysis (beat detection, onset detection)
   - Visualization tools

3. **Advantages**:
   - Highly robust to different audio formats
   - Built-in fallbacks when primary loading methods fail
   - Works across different platforms without many dependencies

## Torchaudio and Its Backends

Torchaudio is PyTorch's audio processing library that provides GPU compatibility and integrates with the PyTorch ecosystem.

### Available Backends:

1. **sox_io**: Uses the Sox (Sound eXchange) utility
   - Needs libsox installed on your system
   - Fast and efficient for basic audio operations

2. **soundfile**: Uses the SoundFile library
   - Better compatibility with various file formats
   - More reliable on some systems

### Backend Selection:

```python
import torchaudio
# Set backend explicitly
torchaudio.set_audio_backend("soundfile")  # or "sox_io"
```

## Why Your Error Occurs

The error `"Couldn't find appropriate backend to handle uri"` typically happens when:

1. **Missing Dependencies**: The system doesn't have the required libraries installed
2. **File Format Issues**: The WAV file might have an unusual encoding or sample rate
3. **Path Issues**: Special characters or permissions problems with the file path

## Solutions in Our App

Our app implements multiple fallback methods:

1. First tries torchaudio with sox_io backend
2. Then tries torchaudio with soundfile backend 
3. Falls back to librosa (most reliable but slower)
4. As last resort, uses scipy.io.wavfile or direct ffmpeg conversion

## Installing Dependencies on Mac

```bash
# Install sox
brew install sox

# Install soundfile dependencies
brew install libsndfile

# Install Python packages
pip install torchaudio torch soundfile pysoundfile librosa
```

## Installing Dependencies on Linux

```bash
# Install sox and soundfile dependencies
sudo apt-get update
sudo apt-get install -y sox libsox-dev libsndfile1 ffmpeg

# Install Python packages
pip install torchaudio torch soundfile pysoundfile librosa
```

## Manual Testing of Audio Loading

You can test if your audio libraries are working correctly with this script:

```python
import os
import torch
import torchaudio
import librosa
import soundfile as sf
from scipy.io import wavfile

# Create a simple test audio file
import numpy as np
sample_rate = 16000
duration = 1  # seconds
t = np.linspace(0, duration, int(sample_rate * duration))
# Generate a 440 Hz sine wave
audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)

# Save as WAV using scipy
test_file = "test_audio.wav"
wavfile.write(test_file, sample_rate, audio_data)
print(f"Created test file: {test_file}")

# Try different loading methods
methods = {
    "torchaudio (sox_io)": lambda: torchaudio.set_audio_backend("sox_io") or torchaudio.load(test_file),
    "torchaudio (soundfile)": lambda: torchaudio.set_audio_backend("soundfile") or torchaudio.load(test_file),
    "librosa": lambda: librosa.load(test_file, sr=None),
    "soundfile": lambda: sf.read(test_file),
    "scipy.io.wavfile": lambda: wavfile.read(test_file)
}

for name, method in methods.items():
    try:
        result = method()
        print(f"✅ {name}: Success!")
    except Exception as e:
        print(f"❌ {name}: Failed - {str(e)}")
```

## What to Use in Your Code

For maximum reliability, use the multi-fallback approach as implemented in our application's `load_audio_robust` function, which tries multiple methods in sequence.
