# English Accent Classifier

## 📌 Overview

The English Accent Classifier is an advanced audio processing system that uses machine learning to identify different English accents from speech. Built with an agent-based architecture, it combines powerful deep learning models with sophisticated audio processing techniques to deliver accurate accent classification.

## 🌟 Features

- **Multi-accent Detection**: Identifies 6 distinct English accents (American, British, Indian, Australian, Canadian, Non-native)
- **High-quality Audio Processing**: Advanced audio enhancement and speech extraction
- **Confidence Calibration**: Sophisticated confidence scoring based on audio quality and accent traits
- **Visual Analysis**: Waveform and spectrogram visualization
- **Video Support**: Extract audio from video files (including Google Drive links)
- **Transcription**: Full speech-to-text capability with quality assessment

## 🧠 Machine Learning Architecture

### Models Used

1. **Speech Recognition**:
   - **Model**: OpenAI's Whisper (Base variant)
   - **Purpose**: Transcribe speech to text for both display and text-based accent analysis
   - **Features**: Multi-language capability, high accuracy, noise resilience

2. **Accent Classification**:
   - **Model**: Facebook Wav2Vec2 (`facebook/wav2vec2-base`)
   - **Architecture**: Fine-tuned self-supervised learning model with a classification head
   - **Classes**: Six accent categories with custom probability calibration

### Agent-based System Design

The application uses a modular agent-based architecture:

1. **TranscriptionAgent**: Handles speech-to-text conversion
   - Audio preprocessing
   - Transcription using Whisper
   - Quality assessment
   - Multiple fallback mechanisms

2. **AccentClassifierAgent**: Performs accent classification
   - Robust audio loading
   - Audio quality enhancement
   - Speech segment extraction
   - Ensemble classification
   - Confidence calibration
   - Text-based accent feature detection

3. **AgentManager**: Coordinates the workflow
   - Manages communication between agents
   - Orchestrates the processing pipeline
   - Error handling and recovery

## 🔄 Workflow

1. **Input**: User uploads a video URL or file
2. **Video Processing**:
   - Download video (with special handling for Google Drive links)
   - Extract audio track
3. **Audio Analysis**:
   - Preprocess audio (normalization, enhancement)
   - Extract speech segments
   - Assess audio quality
4. **Accent Classification**:
   - Segment audio for ensemble prediction
   - Generate accent probabilities
   - Apply special detection for underrepresented accents (e.g., British)
   - Calibrate confidence based on multiple factors
5. **Transcription**: Convert speech to text
6. **Results Presentation**:
   - Display accent with confidence score
   - Show audio quality assessment
   - Visualize audio waveform and spectrogram
   - Present alternative accent possibilities
   - Provide detailed summary

## 🛠️ Technical Details

### Audio Processing Techniques

- **Pre-emphasis**: Enhance high frequencies crucial for accent detection
- **Noise Reduction**: Spectral subtraction for cleaner audio
- **Speech Segmentation**: Energy-based speech detection
- **Ensemble Classification**: Segment-based prediction with weighted averaging

### Confidence Calibration

The system uses a multi-factor approach to calibrate confidence:
- Audio quality metrics (SNR, dynamic range, zero-crossing rate)
- Probability distribution analysis (margin, concentration)
- Accent-specific adjustments for model biases
- Text-based accent feature verification

## Getting Started

### Prerequisites

- Python 3.8+
- FFmpeg (recommended but not required)
- CUDA-compatible GPU (recommended for faster processing)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/accent-classifier.git
cd accent-classifier
```

### 2. Run the Setup Script
```bash
bash setup.sh
```

### 3. Activate the Environment
```bash
conda activate accent-classifier
```

### 4. Launch the Application
```bash
streamlit run app/app.py
```

---

## Project Structure

```
accent-classifier/
├── app/                      
│   ├── app.py                      # Main Streamlit application
│   ├── accent_labels.py            # Accent label mapping utilities
│   └── agents/                     
│       ├── __init__.py            
│       ├── agent_manager.py        # Coordination between agents
│       ├── base_agent.py           # Base agent class
│       ├── accent_classifier_agent.py  # Accent classification logic
│       └── transcription_agent.py      # Speech transcription logic
├── tools/                    
│   ├── audio/                      # Audio processing utilities
│   ├── models/                     # Model training and fine-tuning tools
│   └── ...
├── setup.sh                        # Environment setup script
└── requirements.txt                # Python dependencies
```

---

## Acknowledgments

- OpenAI for the Whisper ASR model.
- Facebook AI for the Wav2Vec2 feature extractor.

---

## License

This project is licensed under the [MIT License](LICENSE).
