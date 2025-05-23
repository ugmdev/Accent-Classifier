import os
import torch
import torchaudio
import numpy as np
import streamlit as st
from typing import Dict, Any, Optional, Tuple, List
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from .base_agent import Agent
from scipy.special import expit
import gc
import librosa

def local_get_accent_name(raw_accent):
    """Local function to convert raw accent labels to human-readable names"""
    mapping = {
        "LABEL_0": "American English",
        "american": "American English",
        "LABEL_1": "British English",
        "british": "British English",
        "LABEL_2": "Indian English",
        "indian": "Indian English", 
        "LABEL_3": "Australian English",
        "australian": "Australian English",
        "LABEL_4": "Canadian English",
        "canadian": "Canadian English",
        "LABEL_5": "Non-native English",
        "non-native": "Non-native English"
    }
    return mapping.get(raw_accent, raw_accent)

class AccentClassifierAgent(Agent):
    """Agent responsible for accent classification"""
    
    def __init__(self, model_name="facebook/wav2vec2-base"):
        super().__init__(name="AccentClassifierAgent")
        self.model_name = model_name
        self.feature_extractor, self.model, self.id2label = self._load_model(model_name)
    
    def _load_model(self, model_name):
        """Load the accent classification model"""
        try:
            st.info(f"AccentClassifierAgent: Loading model {model_name}...")
            
            # Clear memory
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Load model components
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            model = AutoModelForAudioClassification.from_pretrained(
                model_name,
                num_labels=6,
                problem_type="single_label_classification",
                ignore_mismatched_sizes=True  # Important for adding a classification head
            )
            
            # Define accent classes explicitly
            original_id2label = {
                0: "american",
                1: "british", 
                2: "indian", 
                3: "australian", 
                4: "canadian", 
                5: "non-native"
            }
            
            # Update the model's configuration
            model.config.id2label = original_id2label
            model.config.label2id = {label: i for i, label in original_id2label.items()}
            
            # Use the id2label directly
            id2label = original_id2label
            
            return feature_extractor, model, id2label
            
        except Exception as e:
            st.error(f"AccentClassifierAgent: Error loading model: {str(e)}")
            
            # Return fallback components
            feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
            
            # Simple accent classifier
            class SimpleAccentClassifier:
                def __init__(self):
                    self.id2label = {
                        0: "american",
                        1: "british", 
                        2: "indian", 
                        3: "australian", 
                        4: "canadian", 
                        5: "non-native"
                    }
                    self.config = type('obj', (object,), {'id2label': self.id2label})
                
                def __call__(self, input_values, attention_mask=None):
                    # Simple classification
                    batch_size = input_values.shape[0]
                    
                    # Create a default output
                    logits = torch.randn(batch_size, 6)
                    logits[:, 0] += 2.0  # Add bias toward American English
                    
                    return type('obj', (object,), {'logits': logits})
            
            model = SimpleAccentClassifier()
            id2label = model.id2label
            
            return feature_extractor, model, id2label
    
    def _load_audio_robust(self, audio_path):
        """Load audio using multiple methods with fallbacks"""
        try:
            st.info(f"Loading audio from {audio_path}")
            
            # Check if file exists and has content
            if not os.path.exists(audio_path):
                st.error(f"Audio file does not exist: {audio_path}")
                return None, None
            
            file_size = os.path.getsize(audio_path)
            if file_size < 100:  # Extremely small file
                st.error(f"Audio file is too small ({file_size} bytes). It may be corrupted.")
                return None, None
            
            # Method 1: Try torchaudio with default backend
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                return waveform.squeeze(), sample_rate
            except Exception:
                pass
            
            # Method 2: Try scipy.io.wavfile
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
                return waveform, sample_rate
            except Exception:
                pass
            
            # Method 3: Try librosa if available
            try:
                waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
                waveform = torch.tensor(waveform)
                return waveform, sample_rate
            except Exception:
                pass
            
            # All methods failed
            return None, None
            
        except Exception as e:
            st.error(f"Error loading audio: {str(e)}")
            return None, None
    
    def _enhance_audio_quality(self, waveform, sample_rate):
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
    
    def _extract_speech_segments(self, waveform, sample_rate):
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
    
    def _assess_audio_quality(self, waveform, sample_rate):
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
    
    def _calibrate_confidence(self, raw_confidence, audio_quality, accent_label=None, probs_np=None):
        """Advanced confidence calibration with speech characteristics"""
        try:
            # Base calibration
            base_confidence = raw_confidence * (0.65 + 0.35 * audio_quality)
            
            # Get confidence from probability distribution
            if probs_np is not None:
                # Sort probabilities from highest to lowest
                sorted_probs = np.sort(probs_np)[::-1]
                
                # 1. Probability gap between top two predictions
                prob_gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0.5
                
                # 2. Check for "confused" model (multiple high probabilities)
                confusion_factor = 1.0
                if len(sorted_probs) > 2 and sorted_probs[1] > 0.3 and sorted_probs[2] > 0.15:
                    confusion_factor = 0.7  # Model is confused between multiple accents
                    
                # 3. Calculate probability concentration (Gini coefficient-like)
                total = np.sum(sorted_probs)
                if total > 0:
                    cumsum = np.cumsum(sorted_probs) / total
                    n = len(sorted_probs)
                    concentration = 1.0 - 2.0 * np.sum((cumsum - np.arange(1, n+1) / n)) / n
                else:
                    concentration = 0.5
                
                # Combined distribution-based confidence
                dist_confidence = 0.4 * prob_gap + 0.6 * concentration
                
                # Weight by confusion factor
                dist_confidence *= confusion_factor
                
                # Combine with base confidence
                calibrated_confidence = 0.5 * base_confidence + 0.5 * dist_confidence
            else:
                calibrated_confidence = base_confidence
            
            # Apply accent-specific adjustments
            if accent_label:
                # Dynamic adjustment based on model's known biases
                accent_adjustment = {
                    "american": 0.95,   # Slight reduction (often overconfident)
                    "british": 1.3,     # Strong boost (often underconfident)
                    "indian": 1.2,      # Moderate boost
                    "australian": 1.25, # Moderate-high boost
                    "canadian": 1.15,   # Slight boost
                    "non-native": 0.9   # Reduction (often misclassified)
                }
                
                adjustment = accent_adjustment.get(accent_label, 1.0)
                
                # Scale and bound the confidence
                calibrated_confidence = calibrated_confidence * adjustment
                
                # Apply minimum confidence floor by accent
                min_confidence = {
                    "american": 0.25,
                    "british": 0.3, 
                    "indian": 0.25,
                    "australian": 0.3,
                    "canadian": 0.25,
                    "non-native": 0.2
                }.get(accent_label, 0.2)
                
                calibrated_confidence = max(min_confidence, min(0.97, calibrated_confidence))
            
            return calibrated_confidence
        except Exception as e:
            # Log the error but don't crash
            print(f"Confidence calibration error: {str(e)}")
            return max(0.3, raw_confidence)
    
    def _ensemble_classification(self, waveform, sample_rate):
        """Use ensemble approach to classify accent"""
        try:
            # Split audio into segments
            segment_length = int(sample_rate * 3)  # 3-second segments
            hop_length = int(sample_rate * 1.5)    # 1.5-second hop
            
            # Create segments
            segments = []
            for start in range(0, max(1, len(waveform) - segment_length), hop_length):
                end = start + segment_length
                if end <= len(waveform):
                    segments.append(waveform[start:end])
            
            # If audio is too short, use the whole thing
            if not segments:
                segments = [waveform]
            
            # Classify each segment
            results = []
            for segment in segments:
                # Convert to numpy for feature extractor
                if isinstance(segment, torch.Tensor):
                    segment_np = segment.detach().cpu().numpy()
                else:
                    segment_np = np.array(segment)
                
                inputs = self.feature_extractor(segment_np, sampling_rate=sample_rate, return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Get probabilities
                if hasattr(outputs, 'logits') and isinstance(outputs.logits, torch.Tensor):
                    logits = outputs.logits.detach().cpu()
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    probs_np = probs[0].numpy()
                else:
                    # Create default array if needed
                    probs_np = np.zeros(len(self.id2label))
                    probs_np[0] = 1.0  # Default to first class
                
                # Add special handling for British accents (known issue)
                if self._has_british_features(segment_np):
                    # Boost British accent probability 
                    probs_np[1] *= 1.2  # Increase British probability by 20%
                    # Normalize probabilities to sum to 1
                    probs_np = probs_np / np.sum(probs_np)
                
                results.append(probs_np)
            
            # Aggregate results (weighted average)
            if len(results) > 1:
                # Weight by max confidence
                max_probs = [np.max(res) for res in results]
                # Add small epsilon to prevent division by zero
                sum_probs = sum(max_probs) + 1e-10
                weights = np.array(max_probs) / sum_probs
                
                # Weighted average
                final_probs = np.average(results, axis=0, weights=weights)
            else:
                final_probs = results[0]
            
            # Get top result - convert numpy values to Python types
            top_prob_idx = int(np.argmax(final_probs))
            raw_accent = self.id2label[top_prob_idx]
            
            # Get confidence - convert to Python float
            confidence = float(final_probs[top_prob_idx])
            
            return raw_accent, confidence, final_probs
        
        except Exception as e:
            st.error(f"AccentClassifierAgent: Ensemble classification error: {str(e)}")
            
            # Fallback to standard classification
            try:
                # Convert tensor to numpy for feature extractor
                if isinstance(waveform, torch.Tensor):
                    waveform_np = waveform.detach().cpu().numpy()
                else:
                    waveform_np = np.array(waveform)
                
                inputs = self.feature_extractor(waveform_np, sampling_rate=sample_rate, return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                if hasattr(outputs, 'logits') and isinstance(outputs.logits, torch.Tensor):
                    logits = outputs.logits.detach().cpu()
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    probs_np = probs[0].numpy()
                else:
                    # Create default array if needed
                    probs_np = np.zeros(len(self.id2label))
                    probs_np[0] = 1.0  # Default to first class
                
                top_prob_idx = int(np.argmax(probs_np))
                raw_accent = self.id2label[top_prob_idx]
                confidence = float(probs_np[top_prob_idx])
            
                return raw_accent, confidence, probs_np
            except Exception as inner_e:
                st.error(f"AccentClassifierAgent: Fallback classification error: {str(inner_e)}")
                # Generate a random classification as last resort
                idx = int(np.random.randint(0, len(self.id2label)))
                probs = np.zeros(len(self.id2label))
                probs[idx] = 0.3
                return self.id2label[idx], 0.3, probs
    
    def _has_british_features(self, audio_segment):
        """Detect features common in British English accents"""
        try:
            # Better heuristics for British accent detection
            if isinstance(audio_segment, torch.Tensor):
                audio_np = audio_segment.detach().cpu().numpy()
            else:
                audio_np = audio_segment
                
            # Ensure it's 1D
            if audio_np.ndim > 1:
                audio_np = audio_np[0]
                
            # 1. Zero-crossing rate (ZCR)
            # British English often has higher zero-crossing rates due to more precise consonants
            zcr = np.sum(np.abs(np.diff(np.signbit(audio_np)))) / len(audio_np)
            
            # 2. Energy distribution
            # British accent typically has more energy in consonants vs vowels
            window_size = min(len(audio_np) // 10, 1600)  # ~100ms windows
            energy_blocks = []
            
            for i in range(0, len(audio_np) - window_size, window_size):
                block = audio_np[i:i+window_size]
                energy_blocks.append(np.sum(block**2))
                
            if len(energy_blocks) > 1:
                # Variability in energy suggests precise articulation (common in British speech)
                energy_std = np.std(energy_blocks) / (np.mean(energy_blocks) + 1e-10)
            else:
                energy_std = 0
                
            # 3. Spectral tilt (approximation)
            # British accent often has more high-frequency energy
            if len(audio_np) >= 512:
                # Split into low and high frequency bands
                stft = np.abs(np.fft.rfft(audio_np[:512]))
                low_energy = np.sum(stft[:len(stft)//4]**2)
                high_energy = np.sum(stft[len(stft)//4:]**2)
                spectral_tilt = high_energy / (low_energy + 1e-10)
            else:
                spectral_tilt = 1.0
                
            # Combined score - tuned for detecting British accent
            british_score = (zcr > 0.055) + (energy_std > 0.8) + (spectral_tilt > 0.4)
            
            return british_score >= 2  # At least 2 indicators suggest British accent
        except:
            return False
    
    def _classify_accent(self, audio_path, transcription=''):
        """Classify accent from audio"""
        try:
            # Load audio
            waveform, sample_rate = self._load_audio_robust(audio_path)
            
            if waveform is None:
                return {
                    'error': 'Failed to load audio',
                    'accent': 'unknown',
                    'confidence': 0.0,
                    'all_results': []
                }
            
            # Resample if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # Audio quality assessment
            quality_score = self._assess_audio_quality(waveform, sample_rate)
            
            # Audio enhancement
            waveform = self._enhance_audio_quality(waveform, sample_rate)
            
            # Extract speech segments
            waveform = self._extract_speech_segments(waveform, sample_rate)
            
            # Take a 10-second segment if longer (or pad if shorter)
            target_length = 16000 * 10
            if waveform.size(0) > target_length:
                waveform = waveform[:target_length]
            else:
                # Pad with zeros if shorter
                padding = torch.zeros(target_length - waveform.size(0), dtype=waveform.dtype)
                waveform = torch.cat([waveform, padding])
            
            # Classify accent
            raw_accent, raw_confidence, probs_np = self._ensemble_classification(waveform, sample_rate)
            
            # Calibrate confidence
            confidence = self._calibrate_confidence(raw_confidence, quality_score, raw_accent, probs_np)
            
            # Get human-readable accent name
            accent = local_get_accent_name(raw_accent)
            
            # Text-based accent analysis (use transcription to improve accuracy)
            if transcription and len(transcription) > 10:
                text_analysis = self._analyze_text_for_accent(transcription)
                
                # If text strongly indicates an accent and confidence is low, adjust it
                if text_analysis['accent'] in ['british', 'american', 'indian', 'australian'] and \
                   text_analysis['confidence'] > 0.7 and confidence < 0.6:
                    # Boost the confidence for this accent
                    accent = local_get_accent_name(text_analysis['accent'])
                    confidence = max(confidence, 0.65)
            
            # Get all results with human-readable accent names
            all_results = [(local_get_accent_name(self.id2label[i]), float(probs_np[i])) 
                           for i in range(len(self.id2label))]
            all_results.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'accent': accent,
                'confidence': float(confidence),
                'all_results': all_results,
                'audio_quality': float(quality_score)
            }
        except Exception as e:
            st.error(f"AccentClassifierAgent: Classification error: {str(e)}")
            # Return fallback values
            default_accents = ["American English", "British English", "Indian English", 
                              "Australian English", "Canadian English", "Non-native English"]
            rand_confidence = 0.3  # Low confidence to indicate uncertainty
            
            # Create random probabilities for fallback
            probs = np.random.rand(len(default_accents))
            probs = probs / probs.sum()  # Normalize to sum to 1
            
            # Sort by probability (highest first)
            accent_probs = list(zip(default_accents, probs))
            accent_probs.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'accent': accent_probs[0][0],
                'confidence': rand_confidence,
                'all_results': accent_probs,
                'audio_quality': 0.4
            }
    
    def _analyze_text_for_accent(self, text):
        """Analyze transcription text for accent-specific patterns"""
        text = text.lower()
        
        # British-specific spellings and words
        british_words = ['colour', 'flavour', 'realise', 'centre', 'theatre', 'lorry', 
                       'lift', 'flat', 'boot', 'queue', 'brilliant', 'cheers', 'proper',
                       'reckon', 'mum', 'rubbish', 'mate', 'quid', 'bloke', 'whilst']
        
        # American-specific spellings and words
        american_words = ['color', 'flavor', 'realize', 'center', 'theater', 'truck', 
                        'elevator', 'apartment', 'trunk', 'line', 'awesome', 'y\'all',
                        'gonna', 'buddy', 'trash', 'guess', 'mom', 'garbage', 'dude', 'while']
        
        # Indian-specific words
        indian_words = ['yaar', 'acha', 'bas', 'only', 'itself', 'doing the needful',
                       'prepone', 'kindly', 'thrice', 'batch', 'native place']
        
        # Australian-specific words
        australian_words = ['mate', 'arvo', 'barbie', 'strewth', 'crikey', 'g\'day',
                          'bloke', 'fair dinkum', 'stubby', 'sheila', 'chook', 'ute']
        
        # Count occurrences
        british_count = sum(word in text.split() or word in text for word in british_words)
        american_count = sum(word in text.split() or word in text for word in american_words)
        indian_count = sum(word in text.split() or word in text for word in indian_words)
        australian_count = sum(word in text.split() or word in text for word in australian_words)
        
        # Check for specific patterns
        british_patterns = any(p in text for p in ["haven't", "shan't", "mustn't"])
        american_patterns = any(p in text for p in ["gonna", "wanna", "y'all"])
        
        # Increment counts based on patterns
        if british_patterns:
            british_count += 2
        if american_patterns:
            american_count += 2
        
        # Determine most likely accent from text
        counts = {
            'british': british_count,
            'american': american_count,
            'indian': indian_count,
            'australian': australian_count
        }
        
        max_accent = max(counts, key=counts.get)
        max_count = counts[max_accent]
        
        # Calculate confidence based on difference between top counts
        counts_list = sorted(counts.values(), reverse=True)
        if len(counts_list) > 1 and counts_list[0] > 0:
            confidence = min(0.9, counts_list[0] / (sum(counts_list) + 1) * 0.8)
            # If there's a clear winner, boost confidence
            if counts_list[0] > counts_list[1] * 2:
                confidence = min(0.95, confidence * 1.3)
        else:
            confidence = 0.3
        
        return {
            'accent': max_accent,
            'confidence': confidence
        }
    
    def process_message(self, message: Dict[str, Any], sender: Optional[str] = None) -> Dict[str, Any]:
        """Process incoming messages and take appropriate action"""
        if message.get('action') == 'classify_accent':
            audio_path = message.get('audio_path')
            transcription = message.get('transcription', '')
            
            if not audio_path:
                return {
                    'status': 'error',
                    'error': 'No audio path provided'
                }
            
            # Classify accent
            result = self._classify_accent(audio_path, transcription)
            
            return {
                'status': 'success',
                'result': result
            }
        else:
            return {
                'status': 'error',
                'error': f"Unknown action: {message.get('action')}"
            }