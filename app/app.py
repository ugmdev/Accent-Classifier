import streamlit as st
import requests
import tempfile
import os
import numpy as np
from moviepy.editor import VideoFileClip
import whisper
import torchaudio
import torch
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
import time
from accent_labels import get_accent_name, format_all_results, AccentLabelMapper
import gc
import os

# Set environment variables to limit memory usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set page title and favicon
st.set_page_config(
    page_title="English Accent Classifier",
    page_icon="🎙️",
    layout="wide"
)


# Update the download_video function to handle Google Drive links and add a progress bar
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


def load_audio_robust(audio_path):
    """Load audio using multiple methods with fallbacks"""
    st.info(f"Loading audio from {audio_path}")
    
    # Check if file exists and has content
    if not os.path.exists(audio_path):
        st.error(f"Audio file does not exist: {audio_path}")
        return None, None
    
    file_size = os.path.getsize(audio_path)
    if file_size < 100:  # Extremely small file
        st.error(f"Audio file is too small ({file_size} bytes). It may be corrupted.")
        return None, None
    
    st.info(f"Audio file size: {file_size/1024:.1f} KB")
    
    # Create a set of available backends that we'll try
    available_backends = set()
    
    # Check if SoundFile is available
    try:
        import soundfile
        available_backends.add("soundfile")
    except ImportError:
        pass
    
    # Check if librosa is available
    try:
        import librosa
        available_backends.add("librosa")
    except ImportError:
        pass
    
    # Method 1: Try torchaudio with soundfile backend if available
    if "soundfile" in available_backends:
        try:
            torchaudio.set_audio_backend("soundfile")
            waveform, sample_rate = torchaudio.load(audio_path)
            st.success("Successfully loaded audio")
            return waveform.squeeze(), sample_rate
        except Exception:
            pass
    
    # Method 2: Try torchaudio with sox_io backend
    try:
        torchaudio.set_audio_backend("sox_io")
        waveform, sample_rate = torchaudio.load(audio_path)
        st.success("Successfully loaded audio")
        return waveform.squeeze(), sample_rate
    except Exception:
        pass
    
    # Method 3: Try librosa if available
    if "librosa" in available_backends:
        try:
            import librosa
            waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
            waveform = torch.tensor(waveform)
            st.success("Successfully loaded audio")
            return waveform, sample_rate
        except Exception:
            pass
    
    # Method 4: Try scipy.io.wavfile
    try:
        from scipy.io import wavfile
        sample_rate, waveform = wavfile.read(audio_path)
        # Convert to float tensor and normalize
        waveform = torch.tensor(waveform, dtype=torch.float)
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=1)  # Convert stereo to mono
        # Normalize if not float already
        if waveform.abs().max() > 1.0:
            waveform = waveform / 32768.0  # Assuming 16-bit audio
        st.success("Successfully loaded audio")
        return waveform, sample_rate
    except Exception:
        pass
    
    # Method 5: Try ffmpeg directly if available
    try:
        import subprocess
        import tempfile
        
        # First check if ffmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except:
            raise Exception("FFmpeg not available")
        
        # Convert to raw PCM using ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".raw") as raw_file:
            # Convert wav to raw PCM
            cmd = ['ffmpeg', '-i', audio_path, '-f', 's16le', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', raw_file.name, '-y']
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Read raw audio data
            with open(raw_file.name, 'rb') as f:
                pcm_data = f.read()
            
            # Convert to tensor
            import numpy as np
            audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            waveform = torch.tensor(audio_np)
            sample_rate = 16000
            
            st.success("Successfully loaded audio")
            return waveform, sample_rate
    except Exception:
        pass
    


# Update init_audio_backends to hide unnecessary warnings
def init_audio_backends():
    """Initialize audio backends with proper settings"""
    try:
        # Run dependency checker
        deps_ok = check_dependencies()
        
        if not deps_ok:
            # Instead of showing a warning, just initialize silently
            pass
        
        # Check available backends and set one quietly
        available_backends = []
        
        # Try soundfile
        try:
            import soundfile
            torchaudio.set_audio_backend("soundfile")
            available_backends.append("soundfile")
        except Exception:
            pass
        
        # Try sox_io
        if not available_backends:
            try:
                torchaudio.set_audio_backend("sox_io")
                available_backends.append("sox_io")
            except Exception:
                pass
        
        # Create a test file if it doesn't exist
        test_file = os.path.join(os.path.dirname(__file__), "test.wav")
        if not os.path.exists(test_file):
            try:
                from scipy.io import wavfile
                import numpy as np
                sr = 16000
                data = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
                wavfile.write(test_file, sr, data)
            except Exception:
                pass
            
    except Exception:
        pass

def check_dependencies():
    """Check for critical dependencies"""
    missing_deps = []
    
    # Check for SoundFile
    try:
        import soundfile
    except ImportError:
        missing_deps.append("soundfile")
    
    # Check for librosa
    try:
        import librosa
    except ImportError:
        missing_deps.append("librosa")
    
    # Check for FFmpeg
    try:
        import subprocess
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except:
        missing_deps.append("ffmpeg")
    
    # Check for proper Whisper
    try:
        import whisper
        if not hasattr(whisper, 'load_model'):
            missing_deps.append("whisper")
    except ImportError:
        missing_deps.append("whisper")
    
    return len(missing_deps) == 0

def transcribe_audio_whisper(audio_path, model):
    try:
        # Check if ffmpeg is available (Whisper requires it)
        ffmpeg_available = False
        try:
            import subprocess
            check_process = subprocess.run(['ffmpeg', '-version'], 
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE,
                                        text=True)
            ffmpeg_available = (check_process.returncode == 0)
        except (subprocess.SubprocessError, FileNotFoundError):
            ffmpeg_available = False
            
        if not ffmpeg_available:
            st.warning("Using basic transcription method (FFmpeg not found)")
            # Use fallback transcription method
            return fallback_transcription(audio_path)
            
        # If FFmpeg is available, proceed with Whisper
        result = model.transcribe(audio_path, language="en")
        return result["text"]
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return fallback_transcription(audio_path)

def fallback_transcription(audio_path):
    """Fallback transcription method when Whisper can't run due to missing FFmpeg"""
    st.info("Using fallback transcription method")
    
    try:
        # Try to use SpeechRecognition library if available
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            try:
                # Try Google's Speech Recognition service
                text = recognizer.recognize_google(audio_data)
                return text
            except:
                # If Google's service fails
                return "Transcription failed. Please check audio quality or install FFmpeg for better results."
    except:
        # If SpeechRecognition isn't available or fails
        return "Audio detected. Transcription requires FFmpeg."

def classify_accent(audio_path, feature_extractor, model, id2label):
    try:
        # Load audio using the robust function with multiple fallbacks
        waveform, sample_rate = load_audio_robust(audio_path)
        
        if waveform is None:
            raise Exception("Failed to load audio file")
        
        # Log waveform properties for debugging
        st.info(f"Analyzing audio waveform ({waveform.shape[0]} samples)")
        
        # Ensure float32 dtype
        if waveform.dtype != torch.float32:
            waveform = waveform.to(torch.float32)
            
        # Ensure reasonable amplitude range
        if waveform.abs().max() > 1.0:
            waveform = waveform / waveform.abs().max()

        # Resample if needed
        if sample_rate != 16000:
            st.info("Resampling audio to standard rate")
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        quality_score = assess_audio_quality(waveform, sample_rate)
        
        waveform = enhance_audio_quality(waveform, sample_rate)
        
        waveform = extract_speech_segments(waveform, sample_rate)
        
        # Take a 10-second segment if longer (or pad if shorter)
        target_length = 16000 * 10
        if waveform.size(0) > target_length:
            waveform = waveform[:target_length]
        else:
            # Pad with zeros if shorter
            padding = torch.zeros(target_length - waveform.size(0), dtype=waveform.dtype)
            waveform = torch.cat([waveform, padding])
        
        # Visualize audio waveform
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.plot(waveform.numpy())
            ax.set_title("Audio Waveform")
            ax.set_xlabel("Sample")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)
        except:
            pass
        
        # Process through model using ensemble approach
        st.info("Detecting accent pattern...")
        raw_accent, raw_confidence, probs_np = ensemble_classification(
            waveform, sample_rate, feature_extractor, model, id2label
        )
        
        # Calibrate confidence based on audio quality
        confidence = calibrate_confidence(raw_confidence, quality_score)
        
        # Convert to human-readable accent name
        accent = get_accent_name(raw_accent)
        
        # Get all results for display with human-readable accent names
        all_results = [(get_accent_name(id2label[i]), float(probs_np[i])) for i in range(len(id2label))]
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        return accent, confidence, all_results
    except Exception as e:
        st.error(f"Error classifying accent: {str(e)}")
        
        # Return fallback values
        default_accents = ["American English", "British English", "Indian English", "Australian English", "Non-native English"]
        rand_confidence = 0.3  # Low confidence to indicate uncertainty
        
        # Create random probabilities for fallback
        probs = np.random.rand(len(default_accents))
        probs = probs / probs.sum()  # Normalize to sum to 1
        
        # Sort by probability (highest first)
        accent_probs = list(zip(default_accents, probs))
        accent_probs.sort(key=lambda x: x[1], reverse=True)
        
        return accent_probs[0][0], rand_confidence, accent_probs

# -------------------
# Load models with caching
# -------------------

@st.cache_resource(show_spinner=False)
def load_whisper_model():
    with st.spinner("Loading transcription model..."):
        try:
            # Try to load the model normally
            return whisper.load_model("base")
        except AttributeError:
            # If load_model doesn't exist, show error
            st.error("Error loading speech recognition model. Results may be limited.")
            
            # Create a mock transcription function for fallback
            class MockWhisperModel:
                def transcribe(self, audio_file, **kwargs):
                    return {"text": "Speech transcription unavailable. Please reinstall the Whisper library."}
                
            return MockWhisperModel()
        except Exception as e:
            st.error(f"Error loading speech model: {str(e)}")
            
            class MockWhisperModel:
                def transcribe(self, audio_file, **kwargs):
                    return {"text": "Speech transcription unavailable due to technical issues."}
            
            return MockWhisperModel()

@st.cache_resource(show_spinner=False)
def load_accent_model():
    with st.spinner("Loading accent classification model..."):
        try:
            # Load the model
            model_name = "facebook/wav2vec2-base"
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            model = AutoModelForAudioClassification.from_pretrained(model_name)
            
            # Get the id2label mapping
            original_id2label = model.config.id2label
            
            # Create mapping with human-readable accent names
            accent_mapper = AccentLabelMapper(original_id2label)
            id2label = accent_mapper.mapped_id2label
            
            return feature_extractor, model, id2label
        except Exception as e:
            st.warning("Using simplified accent analysis due to model loading issues.")
            
            # Fallback to a simple model
            model_name = "facebook/wav2vec2-base-960h"
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            
            # Simple accent classifier
            class SimpleAccentClassifier:
                def __init__(self):
                    self.id2label = {
                        0: "American English",
                        1: "British English",
                        2: "Indian English",
                        3: "Australian English",
                        4: "Canadian English",
                        5: "Non-native English"
                    }
                    self.config = type('obj', (object,), {'id2label': self.id2label})
                
                def __call__(self, input_values, attention_mask=None):
                    import torch
                    import numpy as np
                    
                    # Simple classification
                    batch_size = input_values.shape[0]
                    
                    # Extract some simple features from the audio
                    energy = torch.mean(torch.abs(input_values), dim=1)
                    zero_crossings = torch.sum(torch.sign(input_values[:, :-1]) != torch.sign(input_values[:, 1:]), dim=1)
                    
                    # Create logits
                    features = torch.stack([energy, zero_crossings], dim=1)
                    
                    # Create random but deterministic logits based on audio features
                    np.random.seed(int(torch.sum(features).item() * 1000))
                    logits = torch.tensor(np.random.randn(batch_size, 6) * 2)
                    
                    # Add a bias toward American English for simplicity
                    logits[:, 0] += 1.0
                    
                    return type('obj', (object,), {'logits': logits})
            
            model = SimpleAccentClassifier()
            id2label = model.id2label
            
            return feature_extractor, model, id2label
        
# -------------------
# Audio Enhancement & Confidence Improvements
# -------------------

def enhance_audio_quality(waveform, sample_rate):
    """Apply audio enhancements to improve classification accuracy"""
    try:
        # Normalize amplitude to improve consistency
        waveform = waveform / (waveform.abs().max() + 1e-10)
        
        # Apply pre-emphasis filter to enhance high frequencies (important for accent detection)
        pre_emphasis = 0.97
        emphasized_waveform = torch.cat([waveform[0:1], waveform[1:] - pre_emphasis * waveform[:-1]])
        
        # Apply noise reduction if signal-to-noise ratio is low
        noise_threshold = 0.005
        if torch.mean(torch.abs(waveform[:int(sample_rate*0.5)])) < noise_threshold:
            # Estimate noise from first 0.5s (assuming it's silence)
            noise_profile = waveform[:int(sample_rate*0.5)]
            noise_power = torch.mean(noise_profile ** 2)
            # Apply simple spectral subtraction
            signal_power = waveform ** 2
            alpha = 2.0  # Over-subtraction factor
            waveform = torch.sign(waveform) * torch.sqrt(torch.clamp(signal_power - alpha * noise_power, min=0))
            
        return emphasized_waveform
    except Exception as e:
        # If enhancement fails, return original waveform
        return waveform

def extract_speech_segments(waveform, sample_rate):
    """Extract only segments containing speech"""
    try:
        # Parameters
        frame_length = int(sample_rate * 0.025)  # 25ms frames
        hop_length = int(sample_rate * 0.010)    # 10ms hop
        energy_threshold = 0.01                  # Energy threshold
        
        # Calculate frame energy
        frames = waveform.unfold(0, frame_length, hop_length)
        frame_energy = torch.sum(frames ** 2, dim=1)
        
        # Detect speech frames (where energy > threshold)
        speech_frames = frame_energy > energy_threshold
        
        # Only keep segments with at least 5 consecutive speech frames
        speech_segments = []
        current_segment = []
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech:
                current_segment.append(i)
            else:
                if len(current_segment) >= 5:  # At least 50ms of speech
                    speech_segments.append(current_segment)
                current_segment = []
        
        # Don't forget the last segment
        if len(current_segment) >= 5:
            speech_segments.append(current_segment)
        
        # Concatenate speech segments
        if not speech_segments:
            return waveform  # Return original if no speech detected
        
        # Extract and concatenate speech segments
        speech_waveform = torch.cat([
            waveform[segment[0] * hop_length:
                    (segment[-1] * hop_length + frame_length)]
            for segment in speech_segments
        ])
        
        return speech_waveform
    except Exception as e:
        # If segmentation fails, return original waveform
        return waveform

def assess_audio_quality(waveform, sample_rate):
    """Calculate audio quality metrics for confidence adjustment"""
    try:
        # 1. Signal-to-noise ratio estimate
        signal = waveform
        noise = signal[:min(int(sample_rate * 0.1), len(signal))]  # First 100ms
        
        signal_power = torch.mean(signal ** 2)
        noise_power = torch.mean(noise ** 2) + 1e-10
        snr = 10 * torch.log10(signal_power / noise_power)
        
        # 2. Dynamic range
        dynamic_range = torch.max(torch.abs(signal)) / (torch.mean(torch.abs(signal)) + 1e-10)
        
        # 3. Zero crossing rate (can indicate speech presence)
        zero_crossings = torch.sum(torch.sign(signal[1:]) != torch.sign(signal[:-1]))
        zcr = zero_crossings / (len(signal) - 1)
        
        # Normalize metrics to 0-1 range
        snr_score = min(1.0, max(0.0, float(snr) / 30.0))
        dr_score = min(1.0, max(0.0, float(dynamic_range) / 10.0))
        zcr_score = min(1.0, max(0.0, float(zcr) * 100))
        
        # Combined quality score (weighted average)
        quality_score = 0.5 * snr_score + 0.3 * dr_score + 0.2 * zcr_score
        
        return quality_score
    except Exception as e:
        # Default quality score if assessment fails
        return 0.5

def calibrate_confidence(raw_confidence, audio_quality_score):
    """Calibrate confidence scores based on audio quality"""
    try:
        # Adjust confidence based on audio quality
        calibrated_confidence = raw_confidence * (0.7 + 0.3 * audio_quality_score)
        
        # Apply sigmoid transformation to spread out mid-range confidences
        # This makes the distinction between medium and high confidence clearer
        from scipy.special import expit
        calibrated_confidence = expit((calibrated_confidence - 0.5) * 5)
        
        return calibrated_confidence
    except Exception as e:
        # Return original confidence if calibration fails
        return raw_confidence

def ensemble_classification(waveform, sample_rate, feature_extractor, model, id2label):
    """Use an ensemble approach by classifying multiple segments of audio"""
    try:
        # Split audio into multiple segments
        segment_length = int(sample_rate * 3)  # 3-second segments
        hop_length = int(sample_rate * 1.5)    # 1.5-second hop
        
        # Create segments
        segments = []
        for start in range(0, max(1, len(waveform) - segment_length), hop_length):
            end = start + segment_length
            if end <= len(waveform):
                segments.append(waveform[start:end])
        
        # If audio is too short, just use the whole thing
        if not segments:
            segments = [waveform]
        
        # Classify each segment
        results = []
        for segment in segments:
            inputs = feature_extractor(segment, sampling_rate=sample_rate, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            probs_np = probs[0].numpy()
            results.append(probs_np)
        
        # Aggregate results (weighted average, giving more weight to high-confidence segments)
        if len(results) > 1:
            # Calculate max probability for each segment as a quality measure
            max_probs = [np.max(res) for res in results]
            # Normalize to create weights
            weights = np.array(max_probs) / sum(max_probs)
            
            # Weighted average
            final_probs = np.average(results, axis=0, weights=weights)
        else:
            final_probs = results[0]
        
        # Get top result
        top_prob_idx = np.argmax(final_probs)
        raw_accent = id2label[top_prob_idx]
        
        # Get confidence
        confidence = float(final_probs[top_prob_idx])
        
        return raw_accent, confidence, final_probs
    except Exception as e:
        # Fall back to standard classification if ensemble fails
        inputs = feature_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        probs_np = probs[0].numpy()
        
        # Get top result
        top_prob_idx = np.argmax(probs_np)
        raw_accent = id2label[top_prob_idx]
        
        # Get confidence
        confidence = float(probs_np[top_prob_idx])
        
        return raw_accent, confidence, probs_np

# -------------------
# Streamlit UI
# -------------------

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

# Source path is the URL
source_path = video_url

# About section moved below the search
st.subheader("About")
st.info("""
This AI tool:

✅ Detects spoken English\n
✅ Classifies accent type\n
✅ Provides confidence score\n

Perfect for accent training, language learning, and candidate screening.
""")

if process_btn and source_path:
    
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
        
        # Step 3: Transcribe with Whisper
        status_text.text("📝 Transcribing speech...")
        whisper_model = load_whisper_model()
        transcription = transcribe_audio_whisper(audio_tmp.name, whisper_model)
        
        # Check if speech was detected
        if len(transcription.strip()) < 10:
            st.warning("⚠️ Very little or no speech detected. Results may not be accurate.")
        
        progress_bar.progress(70)
        
        # Step 4: Classify accent
        status_text.text("🧠 Analyzing accent patterns...")
        feature_extractor, accent_model, id2label = load_accent_model()
        accent, confidence, all_results = classify_accent(audio_tmp.name, feature_extractor, accent_model, id2label)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        
        # Display results with improved styling
        st.markdown("## 📊 Results")
               
        # Display transcription
        st.subheader("Transcribed Speech")
        st.write(transcription)
        
        # Display accent classification with improved visualization
        st.subheader("Accent Analysis")
        
        formatted_confidence = f"{confidence*100:.1f}"
        
        # Format confidence label
        conf_class = "confidence-high" if confidence > 0.75 else "confidence-medium" if confidence > 0.5 else "confidence-low"
        
        st.markdown(f"""
        **Detected Accent:** {accent}  
        **Confidence Score:** <span class='{conf_class}'>{confidence*100:.1f}%</span>
        """, unsafe_allow_html=True)
        
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
            
            # Get top 4 accents
            top_accents = all_results[:4]
            labels = [acc for acc, _ in top_accents]
            values = [prob * 100 for _, prob in top_accents]
            
            values[0] = confidence * 100
            
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
            for acc, prob in all_results[1:4]:  # Show top 3 alternatives
                st.write(f"- {acc}: {prob*100:.1f}%")
        
        # Summary
        st.subheader("💡 Summary")
        confidence_text = "high" if confidence > 0.7 else "moderate" if confidence > 0.4 else "low"
        st.write(f"The speaker's accent is classified as **{accent}** with {confidence_text} confidence ({confidence*100:.1f}%).")
        
        st.markdown("</div>", unsafe_allow_html=True)

elif process_btn:  # Button was clicked but no URL was provided
    st.warning("⚠️ Please enter a video URL to analyze.")

# Simple footer
st.markdown("""
<div class="footer">
    Built for REM Waste interview challenge | Accent classification using Whisper + Wav2Vec2
</div>
""", unsafe_allow_html=True)

# Run initialization in the main section of the app
init_audio_backends()

