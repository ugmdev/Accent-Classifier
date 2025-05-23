import os
import whisper
import streamlit as st
import numpy as np
from typing import Dict, Any, Optional, Tuple
from .base_agent import Agent

class TranscriptionAgent(Agent):
    """Agent responsible for audio preprocessing and speech transcription"""
    
    def __init__(self, model_name: str = "base"):
        super().__init__(name="TranscriptionAgent")
        self.model = self._load_model(model_name)
    
    def _load_model(self, model_name: str):
        """Load the whisper model for transcription"""
        try:
            st.info(f"TranscriptionAgent: Loading Whisper {model_name} model...")
            return whisper.load_model(model_name)
        except Exception as e:
            st.error(f"TranscriptionAgent: Error loading model: {str(e)}")
            
            # Return a mock model if the real one can't be loaded
            class MockWhisperModel:
                def transcribe(self, audio_file, **kwargs):
                    return {"text": "Speech transcription unavailable. Please check your Whisper installation."}
            
            return MockWhisperModel()
    
    def _transcribe_audio(self, audio_path: str) -> Tuple[str, float]:
        """Transcribe audio file to text"""
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                st.error(f"Audio file not found: {audio_path}")
                return "Audio file not found", 0.0
            
            # Check file is not empty
            if os.path.getsize(audio_path) < 100:
                st.error(f"Audio file is too small or empty: {audio_path}")
                return "Audio file is empty or corrupted", 0.0
            
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
                st.warning("TranscriptionAgent: Using basic transcription method (FFmpeg not found)")
                # Use fallback transcription method
                return self._fallback_transcription(audio_path), 0.5
                
            # If FFmpeg is available, proceed with Whisper
            result = self.model.transcribe(audio_path, language="en")
            
            # Calculate quality metrics (approximation)
            quality_score = self._calculate_transcription_quality(result)
            
            return result["text"], quality_score
            
        except Exception as e:
            st.error(f"TranscriptionAgent: Error transcribing audio: {str(e)}")
            
            # Try fallback transcription
            return self._fallback_transcription(audio_path), 0.3
    
    def _fallback_transcription(self, audio_path: str) -> str:
        """Fallback transcription method when Whisper can't run due to missing FFmpeg"""
        st.info("TranscriptionAgent: Using fallback transcription method")
        
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
    
    def _calculate_transcription_quality(self, result: Dict[str, Any]) -> float:
        """Calculate a quality score for the transcription"""
        try:
            # Get the transcribed text
            text = result.get("text", "")
            
            # Check if text is empty
            if not text.strip():
                return 0.0
            
            # Check length (very short transcriptions might be less reliable)
            length_score = min(1.0, len(text) / 500)
            
            # Check segments confidence if available
            segments = result.get("segments", [])
            if segments:
                avg_confidence = np.mean([segment.get("confidence", 0.5) for segment in segments])
            else:
                avg_confidence = 0.5
            
            # Calculate overall quality score
            quality_score = 0.7 * avg_confidence + 0.3 * length_score
            
            return quality_score
        except Exception as e:
            # Return a default quality score
            return 0.5
    
    def process_message(self, message: Dict[str, Any], sender: Optional[str] = None) -> Dict[str, Any]:
        """Process incoming messages and take appropriate action"""
        if message.get('action') == 'transcribe_audio':
            audio_path = message.get('audio_path')
            
            if not audio_path:
                return {
                    'status': 'error',
                    'error': 'No audio path provided'
                }
            
            # Transcribe audio
            transcription, quality = self._transcribe_audio(audio_path)
            
            return {
                'status': 'success',
                'transcription': transcription,
                'quality': quality
            }
        else:
            return {
                'status': 'error',
                'error': f"Unknown action: {message.get('action')}"
            }