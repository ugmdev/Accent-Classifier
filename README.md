# English Accent Classifier

![Accent Classifier](https://img.shields.io/badge/AI-Accent%20Classification-blue)
![Python](https://img.shields.io/badge/Python-3.10-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red)

A machine learning application that classifies English accents from video content. The system extracts audio from videos, transcribes speech, and uses advanced neural networks to identify the speaker's accent.

## 🎯 Features

- **Video Processing**: Extract audio from videos via URL
- **Speech Recognition**: Transcribe speech with OpenAI's Whisper model
- **Accent Classification**: Identify accents with confidence scores
- **Interactive UI**: User-friendly Streamlit interface with visualizations

## 📊 Supported Accents

The classifier can identify multiple English accents, including:
- American English
- British English
- Indian English
- Australian English
- Canadian English
- Non-native English variations

## 🔧 Installation

### Prerequisites
- Python 3.10 or higher
- Conda package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/accent-classifier.git
cd accent-classifier

# Run the setup script to create conda environment
bash setup.sh

# Activate the environment
conda activate accent_classifier
```

## 🚀 Usage

```bash
# Start the application
streamlit run app/app.py
```

Then open your browser and go to http://localhost:8501

### Using the Application

1. Enter a video URL in the input field
2. Click "Analyze Accent"
3. View the transcription and accent classification results

## 💻 Technical Implementation

The application uses a pipeline of several components:

1. **Video Processing**: Downloads videos and extracts audio using FFmpeg
2. **Speech Recognition**: Transcribes audio using Whisper ASR
3. **Feature Extraction**: Processes audio with Wav2Vec2 feature extraction
4. **Accent Classification**: Identifies accents using a fine-tuned model
5. **Result Visualization**: Presents results with confidence scores

## 📈 Improving Model Performance

For better accent classification results, you can:

1. **Use a Better Base Model**: Replace the default model with XLS-R for improved accuracy
   ```python
   MODEL_NAME="facebook/wav2vec2-xls-r-300m" streamlit run app/app.py
   ```

2. **Fine-tune on Specific Accents**: Use the tools in the tools directory to fine-tune the model

## 🧰 Project Structure

```
accent-classifier/
├── app/               # Application code
│   ├── app.py         # Main Streamlit application
│   └── accent_labels.py  # Accent label mapping utilities
├── tools/             # Utility scripts organized by functionality
│   ├── audio/         # Audio processing utilities
│   ├── models/        # Model training and fine-tuning tools
│   └── ...
├── setup.sh           # Environment setup script
└── requirements.txt   # Python dependencies
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for accent models
- [Streamlit](https://streamlit.io/) for the web interface
