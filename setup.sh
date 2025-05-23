#!/bin/bash

# English Accent Classifier Setup Script
echo "🎙️ Setting up English Accent Classifier environment..."

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found! Please install Miniconda or Anaconda first."
    echo "💡 Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create Conda environment
echo "🔧 Creating accent-classifier conda environment..."
conda create -n accent-classifier python=3.9 -y

# Activate environment
eval "$(conda shell.bash hook)"
conda activate accent-classifier

# Install PyTorch with CPU/GPU based on system capability
if [ "$(uname)" = "Darwin" ]; then
    # macOS - use CPU version
    echo "🍎 macOS detected. Installing CPU version of PyTorch..."
    conda install -y pytorch torchaudio -c pytorch
elif command -v nvidia-smi &> /dev/null; then
    # System with NVIDIA GPU
    echo "🖥️ NVIDIA GPU detected. Installing GPU version of PyTorch..."
    conda install -y pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
else
    # System without NVIDIA GPU
    echo "💻 Installing CPU version of PyTorch..."
    conda install -y pytorch torchaudio cpuonly -c pytorch
fi

# Install Python packages from requirements.txt
echo "📦 Installing Python packages..."
pip install -r requirements.txt

# Install ffmpeg if not installed
if ! command -v ffmpeg &> /dev/null; then
    echo "🎬 Installing FFmpeg..."
    if [ "$(uname)" = "Darwin" ]; then
        # macOS
        conda install -y -c conda-forge ffmpeg
    else
        # Linux
        conda install -y -c conda-forge ffmpeg
    fi
else
    echo "✅ FFmpeg already installed."
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p app/audio_cache

# Download small test model to validate installation
echo "🔍 Validating installation by downloading test model..."
python -c "from transformers import AutoFeatureExtractor; AutoFeatureExtractor.from_pretrained('facebook/wav2vec2-base', cache_dir='app/models')"
python -c "import whisper; whisper.load_model('tiny')"

# Final instructions
echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 To activate the environment:"
echo "    conda activate accent-classifier"
echo ""
echo "🚀 To start the application:"
echo "    streamlit run app/app.py"
echo ""
echo "🌐 The app will be available at http://localhost:8501"
echo ""
echo "🔧 For custom models, use:"
echo "    MODEL_NAME=\"your-model-name\" streamlit run app/app.py"
echo ""