# English Accent Classifier

This application analyzes speech from video files to classify the speaker's English accent. It combines speech transcription using OpenAI's Whisper model with accent classification using a fine-tuned Wav2Vec2 model.

## Features

- Supports video uploads or URL inputs (including YouTube, Google Drive, etc.)
- Transcribes speech using Whisper
- Classifies the speaker's accent
- Provides confidence scores and alternative possibilities

## Installation

### Prerequisites

- Python 3.7 or later
- FFmpeg (required for audio processing)

### Setup Instructions

1. **Clone this repository:**
   ```bash
   git clone <repository-url>
   cd Accent-Classifier
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   python whisper_fix.py
   ```

3. **Install FFmpeg:**
   - Mac: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)

## Usage

Run the application:
```bash
streamlit run app.py
```

Then access the web interface at http://localhost:8501

## Troubleshooting

If you encounter audio processing errors:

1. Make sure FFmpeg is installed and available in your system PATH
2. Check that the audio codec in your video is supported
3. Try different video formats or sources
4. Run the whisper fix script: `python whisper_fix.py`
5. Install librosa: `pip install librosa`

## Dependencies

- streamlit: Web interface
- openai-whisper: Speech transcription
- transformers & wav2vec2: Accent classification
- moviepy/ffmpeg: Audio extraction
- librosa: Alternative audio processing
