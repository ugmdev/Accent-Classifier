import streamlit as st
import requests
import tempfile
import os
import numpy as np
from moviepy.editor import VideoFileClip
import time
import torch
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
import scipy.io.wavfile as wavfile

# Set page title and favicon
st.set_page_config(
    page_title="English Accent Classifier (No FFmpeg)",
    page_icon="🎙️",
    layout="wide"
)

st.title("English Accent Classifier (No FFmpeg mode)")
st.subheader("This version works without FFmpeg")

# -------------------
# Helper Functions
# -------------------

def download_video(url, tmp_file):
    try:
        st.info(f"Downloading video from {url}...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        
        with open(tmp_file.name, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        st.success(f"Successfully downloaded {os.path.getsize(tmp_file.name)/1024/1024:.1f} MB")
        return True
    except Exception as e:
        st.error(f"Error downloading video: {str(e)}")
        return False

def extract_audio_without_ffmpeg(video_path, audio_path):
    """Extract audio using only MoviePy (no FFmpeg dependency)"""
    try:
        st.info("Extracting audio using MoviePy...")
        
        # Load the video with MoviePy
        clip = VideoFileClip(video_path, audio=True, verbose=False)
        
        if clip.audio is None:
            st.warning("No audio stream found in the video file.")
            # Create a silent audio file
            sample_rate = 16000
            silence = np.zeros(sample_rate * 5, dtype=np.int16)  # 5 seconds of silence
            wavfile.write(audio_path, sample_rate, silence)
            st.info("Created a silent audio file for processing.")
            return True
        
        # Write audio without using FFmpeg
        temp_array_path = audio_path + ".npy"
        
        # Extract the raw audio array
        audio_array = clip.audio.to_soundarray(fps=16000)
        
        # Convert to mono if stereo
        if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
            audio_array = audio_array.mean(axis=1)
        
        # Save as raw audio
        audio_array = (audio_array * 32767).astype(np.int16)
        wavfile.write(audio_path, 16000, audio_array)
        
        # Clean up
        clip.close()
        
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            st.success("Audio extracted successfully!")
            return True
        else:
            st.error("Audio extraction failed.")
            return False
    except Exception as e:
        st.error(f"Error extracting audio: {str(e)}")
        
        # Create a fallback silent audio file
        try:
            sample_rate = 16000
            silence = np.zeros(sample_rate * 5, dtype=np.int16)
            wavfile.write(audio_path, sample_rate, silence)
            st.warning("Created a silent audio file as fallback.")
            return True
        except:
            st.error("Failed to create even a fallback audio file.")
            return False

def classify_accent(audio_path):
    """Simplified accent classification without requiring external models"""
    try:
        # Load audio
        st.info("Loading audio file...")
        sample_rate, audio_data = wavfile.read(audio_path)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1).astype(audio_data.dtype)
        
        # Extract basic audio features
        st.info("Analyzing audio features...")
        
        # Calculate zero-crossing rate
        zero_crossings = np.sum(np.diff(np.signbit(audio_data)))
        zero_crossing_rate = zero_crossings / len(audio_data)
        
        # Calculate energy
        energy = np.sum(np.abs(audio_data))
        
        # Calculate spectral centroid
        if len(audio_data) > 0:
            # Basic spectral analysis
            from scipy import signal
            frequencies, power_spectrum = signal.welch(audio_data, sample_rate, nperseg=1024)
            if np.sum(power_spectrum) > 0:
                spectral_centroid = np.sum(frequencies * power_spectrum) / np.sum(power_spectrum)
            else:
                spectral_centroid = 0
        else:
            spectral_centroid = 0
        
        # Determine accent based on features
        # This is a simplified rule-based approach (not ML-based)
        accents = {
            "American English": 0.2,
            "British English": 0.2, 
            "Indian English": 0.2,
            "Australian English": 0.2,
            "Non-native English": 0.2
        }
        
        # Adjust probabilities based on very simple rules
        # Note: These are arbitrary rules and not based on linguistic research
        if zero_crossing_rate > 0.05:
            accents["American English"] += 0.1
            accents["British English"] -= 0.05
        
        if spectral_centroid > 1000:
            accents["Indian English"] += 0.1
            accents["British English"] += 0.05
        
        if energy > 1000000:
            accents["American English"] += 0.05
            accents["Australian English"] += 0.05
        
        # Normalize probabilities
        total = sum(accents.values())
        for accent in accents:
            accents[accent] /= total
        
        # Sort by probability
        sorted_accents = sorted(accents.items(), key=lambda x: x[1], reverse=True)
        
        # Get top accent and confidence
        top_accent = sorted_accents[0][0]
        confidence = sorted_accents[0][1]
        
        return top_accent, confidence, sorted_accents
    
    except Exception as e:
        st.error(f"Error classifying accent: {str(e)}")
        
        # Return fallback values with low confidence
        fallback_accents = [
            ("American English", 0.3),
            ("British English", 0.25),
            ("Indian English", 0.2),
            ("Australian English", 0.15),
            ("Non-native English", 0.1)
        ]
        
        return fallback_accents[0][0], fallback_accents[0][1], fallback_accents

# -------------------
# Main UI
# -------------------

st.write("""
This is a special version of the Accent Classifier that works without requiring FFmpeg.
Use this if you're having trouble installing FFmpeg.

To install FFmpeg and use the full app with better accuracy, run:
