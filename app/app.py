import streamlit as st
import requests
import tempfile
import os
import numpy as np
from moviepy.editor import VideoFileClip
import torchaudio
import torch
import time
import gc
import matplotlib.pyplot as plt
import librosa
import librosa.display
import whisper

from accent_labels import get_accent_name, AccentLabelMapper
# Import the agent classes
from agents.agent_manager import AgentManager

# Set environment variables to limit memory usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create a singleton instance of the agent manager
@st.cache_resource(show_spinner=False)
def get_agent_manager():
    return AgentManager()

# Set page title and favicon
st.set_page_config(
    page_title="English Accent Classifier",
    page_icon="🎙️",
    layout="wide"
)

# Helper functions for video handling
def download_video(url, tmp_file):
    try:
        # Special handling for Google Drive links
        if "drive.google.com" in url:
            # Extract the file ID from the Google Drive URL
            if "file/d/" in url:
                file_id = url.split("file/d/")[1].split("/")[0]
            elif "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]
            else:
                st.error("Invalid Google Drive URL format. Please use a direct sharing link.")
                return False
            
            # Create the direct download link
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            st.info(f"Using Google Drive direct download link: {url}")

        # Download headers first to check content type
        head_response = requests.head(url, allow_redirects=True)
        if head_response.status_code != 200:
            st.error(f"Couldn't access URL (status code: {head_response.status_code}). Check if the URL is valid and publicly accessible.")
            return False
            
        # Check content type if available
        content_type = head_response.headers.get('content-type', '')
        if content_type and not any(media_type in content_type.lower() for media_type in ['video', 'mp4', 'octet-stream']):
            st.warning(f"URL may not point to a video file (Content-Type: {content_type}). Will attempt download anyway.")
        
        # Proceed with download
        st.info(f"Downloading video from {url}...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        
        # Get file size if available
        file_size = int(r.headers.get('content-length', 0))
        downloaded = 0
        
        # Create a progress bar for download
        download_bar = st.progress(0)
        download_status = st.empty()
        
        with open(tmp_file.name, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if file_size > 0:
                        # Update progress bar
                        progress = min(int(downloaded / file_size * 100), 100)
                        download_bar.progress(progress)
                        download_status.text(f"Downloaded: {downloaded/1024/1024:.1f} MB of {file_size/1024/1024:.1f} MB ({progress}%)")
        
        # Clean up progress indicators
        download_status.empty()
        download_bar.empty()
        
        # Verify file size
        if os.path.getsize(tmp_file.name) < 1000:  # Less than 1KB
            st.error("Downloaded file is too small to be a valid video. Check URL again.")
            return False
            
        st.success(f"Successfully downloaded {os.path.getsize(tmp_file.name)/1024/1024:.1f} MB")
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading video: {str(e)}")
        return False
    except Exception as e:
        st.error(f"Unexpected error during download: {str(e)}")
        return False

def extract_audio_from_video(video_path, audio_path):
    try:
        # Check if the video file exists and has content
        if not os.path.exists(video_path):
            st.error(f"Video file not found at {video_path}")
            return False
            
        file_size = os.path.getsize(video_path)
        if file_size < 1000:  # Less than 1KB
            st.error(f"Video file is too small ({file_size} bytes). It may be corrupted or empty.")
            return False
            
        st.info(f"Extracting audio from video file ({file_size/1024/1024:.1f} MB)...")
        
        # Try using ffmpeg directly first
        ffmpeg_available = False
        try:
            import subprocess
            # Check if ffmpeg is available
            check_process = subprocess.run(['ffmpeg', '-version'], 
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.PIPE,
                                         text=True)
            ffmpeg_available = (check_process.returncode == 0)
            
            if ffmpeg_available:
                cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path, '-y']
                process = subprocess.run(cmd, check=True, capture_output=True)
                st.success("Audio extracted successfully")
                return True
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            st.warning("Using MoviePy for audio extraction...")
            ffmpeg_available = False
        
        # Fallback to MoviePy
        try:
            clip = VideoFileClip(video_path)
            if clip.audio is None:
                st.warning("No audio stream found in the video file.")
                # Create a silent audio file
                import numpy as np
                from scipy.io import wavfile
                sample_rate = 16000
                silence = np.zeros(sample_rate * 5, dtype=np.int16)  # 5 seconds of silence
                wavfile.write(audio_path, sample_rate, silence)
                st.info("Created a silent audio file for processing.")
                return True
                
            clip.audio.write_audiofile(audio_path, codec="pcm_s16le", verbose=False, logger=None)
            clip.close()
            
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                st.success("Audio extraction completed successfully")
                return True
            else:
                st.error("Audio file was not created properly.")
                return False
        except Exception as e:
            st.error(f"Audio extraction failed: {str(e)}")
            
    except Exception as e:
        st.error(f"Error extracting audio: {str(e)}")
        st.error("Try using a different video file or URL.")
        return False

# UI components
st.markdown("""
<style>
    /* Overall page styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }
    
    /* Text and header styling */
    .main-header {
        text-align: center;
        color: #40A0FF;
        font-size: 2.2rem;
        margin-bottom: 0.2rem;
        font-weight: 600;
    }
    .subtitle {
        text-align: center;
        color: #999;
        margin-bottom: 1.5rem;
    }
    
    /* Results container */
    .result-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* URL input styling */
    .url-container {
        padding: 0px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    /* Confidence colors */
    .confidence-high {color: #4CAF50; font-weight: bold;}
    .confidence-medium {color: #FF9800; font-weight: bold;}
    .confidence-low {color: #F44336; font-weight: bold;}
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: 500;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    
    /* About section */
    .about-section {
        margin-top: 1rem;
        padding: 1rem;
    }
    
    /* Hide unnecessary elements */
    .css-18e3th9 {
        padding-top: 0rem;
    }
    .css-hxt7ib {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .css-10trblm {
        margin-top: 0;
    }
    
    /* Footer styling */
    .footer {
        text-align: center; 
        padding-top: 1rem;
        margin-top: 2rem;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #444;
    }
</style>
""", unsafe_allow_html=True)

# Main header with microphone icon
st.markdown("<h1 class='main-header'>🎙️ English Accent Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analyze speech accent from any video</p>", unsafe_allow_html=True)

# URL input container without the white background
st.markdown("<div class='url-container'>", unsafe_allow_html=True)
video_url = st.text_input(
    "",
    placeholder="Paste your video URL here",
)
process_btn = st.button("Analyze Accent", type="primary")
st.markdown("</div>", unsafe_allow_html=True)

# About section moved below the search
st.subheader("About")
st.info("""
This AI tool:

✅ Detects spoken English\n
✅ Classifies accent type\n
✅ Provides confidence score\n

Perfect for accent training, language learning, and candidate screening.
""")

# Agent debugging tools
with st.expander("🔍 Agent System Debug", expanded=False):
    agent_manager = get_agent_manager()
    status = agent_manager.get_agent_status()
    
    st.markdown("### Agent Status")
    st.write(f"Transcription Agent: {'✅ Active' if status['transcription_agent'] else '❌ Inactive'}")
    st.write(f"Accent Classifier Agent: {'✅ Active' if status['accent_agent'] else '❌ Inactive'}")
    
    if st.button("Initialize Agents"):
        agent_manager.initialize()
        st.success("Agents initialized successfully")
    
    # Show recent history
    if agent_manager.initialized:
        st.markdown("### Recent Agent Activity")
        history = agent_manager.get_history()
        for i, entry in enumerate(history[-3:]):  # Show last 3 entries
            st.markdown(f"**Analysis #{len(history)-i}**")
            st.markdown(f"*Timestamp: {time.ctime(entry['timestamp'])}*")
            # Use collapsible markdown for details
            details = "\n".join([f"- Agent: {step['agent']}, Action: {step['action']}" for step in entry['steps']])
            st.markdown(details)
            st.divider()

# Main processing logic
if process_btn and video_url:
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create temp files
    with tempfile.NamedTemporaryFile(suffix=".mp4") as video_tmp, tempfile.NamedTemporaryFile(suffix=".wav") as audio_tmp:
        
        # Step 1: Get the video
        status_text.text("🎬 Processing video...")
        progress_bar.progress(10)
        
        status_text.text("📥 Downloading video from URL...")
        if not download_video(video_url, video_tmp):
            st.stop()
        video_path = video_tmp.name
        
        progress_bar.progress(30)
        
        # Step 2: Extract audio
        status_text.text("🔊 Extracting audio...")
        if not extract_audio_from_video(video_path, audio_tmp.name):
            st.stop()
        
        progress_bar.progress(50)
        
        # Step 3: Analyze with agents
        status_text.text("🧠 Analyzing accent patterns...")
        
        # Get agent manager
        agent_manager = get_agent_manager()
        
        # Make sure agents are initialized
        if not agent_manager.initialized:
            agent_manager.initialize()
        
        # Analyze the speech
        result = agent_manager.analyze_speech(audio_tmp.name)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        
        # Display results with improved styling
        st.markdown("## 📊 Results")
        
        # Display transcription
        st.subheader("Transcribed Speech")
        st.write(result.get('transcription', 'No transcription available'))
        
        # Display accent classification with improved visualization
        st.subheader("Accent Analysis")
        
        accent_info = result.get('accent', {})
        accent = accent_info.get('name', 'Unknown')
        confidence = accent_info.get('confidence', 0.0)
        rounded_confidence = round(confidence * 100, 1)
        all_results = accent_info.get('all_results', [])
        audio_quality = accent_info.get('audio_quality', 0.0)
        
        # Format confidence label
        conf_class = "confidence-high" if confidence > 0.75 else "confidence-medium" if confidence > 0.5 else "confidence-low"
        
        st.markdown(f"""
        **Detected Accent:** {accent}  
        **Confidence Score:** <span class='{conf_class}'>{rounded_confidence}%</span>  
        **Audio Quality:** <span class='{conf_class}'>{audio_quality*100:.1f}%</span>
        """, unsafe_allow_html=True)
        
        # Add audio waveform visualization
        st.subheader("Audio Waveform")
        try:
            # Load the audio file for visualization
            y, sr = librosa.load(audio_tmp.name)
            
            # Create waveform plot
            fig, ax = plt.subplots(figsize=(10, 3))
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set_title("Audio Waveform")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)
            
            # Optionally add spectrogram
            st.subheader("Spectrogram")
            fig, ax = plt.subplots(figsize=(10, 4))
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            img=librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
            ax.set_title("Audio Spectrogram")
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            st.pyplot(fig)
            
        except Exception as e:
            st.warning(f"Could not generate audio visualization: {str(e)}")
        
        # Add confidence score explanation
        with st.expander("What does the confidence score mean?"):
            st.markdown("""
            **Confidence Score Explanation:**
            
            - **High (>75%)**: Strong accent match with high reliability
            - **Medium (50-75%)**: Probable accent match with moderate reliability
            - **Low (<50%)**: Possible accent match but with low reliability
            
            Factors affecting confidence include audio quality, speech clarity, 
            background noise, and how well the accent matches known patterns.
            """)
        
        # Visual chart for accent confidence
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Get top 4 accents or all if less than 4
            top_accents = all_results[:min(4, len(all_results))]
            labels = [acc for acc, _ in top_accents]
            values = []
            for i, (acc, prob) in enumerate(top_accents):
            # If this is the top accent, use the calibrated confidence
                if i == 0:
                    values.append(rounded_confidence)
                else:
                    values.append(prob * 100)
            
            # Create a horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.barh(labels, values, color=sns.color_palette("Blues", len(labels)))
            
            # Add percentage labels
            for i, v in enumerate(values):
                formatted_value = f"{v:.1f}%"
                ax.text(v + 1, i, f"{v:.1f}%", va='center')
                
            ax.set_xlabel("Confidence (%)")
            ax.set_title("Accent Confidence Scores")
            
            st.pyplot(fig)
        except Exception:
            # Fallback to text display
            st.write("Other possibilities:")
            for i, (acc, prob) in enumerate(all_results[1:min(4, len(all_results))]):  # Show top 3 alternatives
                st.write(f"- {acc}: {prob*100:.1f}%")
        
        # Summary
        st.subheader("💡 Summary")
        
        # Classification certainty description
        if confidence > 0.85:
            certainty = "very high"
        elif confidence > 0.7:
            certainty = "high"
        elif confidence > 0.5:
            certainty = "moderate"
        elif confidence > 0.3:
            certainty = "low"
        else:
            certainty = "very low"
            
        # Audio quality description  
        if audio_quality > 0.8:
            quality_desc = "excellent"
        elif audio_quality > 0.6:
            quality_desc = "good"
        elif audio_quality > 0.4:
            quality_desc = "acceptable"
        else:
            quality_desc = "poor"
        
        st.markdown(f"""
        The speaker's accent is classified as **{accent}** with **{certainty} confidence** ({rounded_confidence}%).
        
        The audio quality is **{quality_desc}** ({audio_quality*100:.1f}%), which {
        "strongly supports" if audio_quality > 0.7 else 
        "adequately supports" if audio_quality > 0.5 else 
        "somewhat limits"} the accuracy of the accent classification.
        
        **Alternative possibilities:**
        """)
        
        # Show alternative accents
        for i, (acc, prob) in enumerate(all_results[1:4]):  # Top 3 alternatives
            st.markdown(f"- {acc}: {prob*100:.1f}%")
        

elif process_btn:  # Button was clicked but no URL was provided
    st.warning("⚠️ Please enter a video URL to analyze.")

# Simple footer
st.markdown("""
<div class="footer">
    Built with an agent-based architecture | Accent classification using Whisper + Wav2Vec2
</div>
""", unsafe_allow_html=True)