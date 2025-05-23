import streamlit as st
from typing import Dict, Any, Optional, List, Tuple
import time
from .transcription_agent import TranscriptionAgent
from .accent_classifier_agent import AccentClassifierAgent

class AgentManager:
    """Manages communication between agents and coordinates tasks"""
    
    def __init__(self):
        self.transcription_agent = None
        self.accent_agent = None
        self.initialized = False
        self.history = []
    
    def initialize(self):
        """Initialize all agents"""
        if self.initialized:
            return
        
        st.info("Initializing agents...")
        
        try:
            # Initialize transcription agent first (for audio handling)
            self.transcription_agent = TranscriptionAgent(model_name="base")
            
            # Initialize accent classifier agent
            self.accent_agent = AccentClassifierAgent(model_name="CAiRE/wav2vec2-large-xlsr-53-english-accents")
            
            self.initialized = True
            st.success("Agents initialized successfully")
        except Exception as e:
            st.error(f"Failed to initialize agents: {str(e)}")
    
    def get_agent_status(self) -> Dict[str, bool]:
        """Get status of all agents"""
        return {
            'transcription_agent': self.transcription_agent is not None,
            'accent_agent': self.accent_agent is not None
        }
    
    def analyze_speech(self, audio_path: str) -> Dict[str, Any]:
        """Analyze speech from audio file using the agent system"""
        # Make sure agents are initialized
        if not self.initialized:
            self.initialize()
        
        # Create a history entry for this analysis
        history_entry = {
            'timestamp': time.time(),
            'steps': []
        }
        
        try:
            # Step 1: Transcribe audio
            transcription_result = self.transcription_agent.process_message({
                'action': 'transcribe_audio',
                'audio_path': audio_path
            })
            
            history_entry['steps'].append({
                'agent': 'TranscriptionAgent',
                'action': 'transcribe_audio',
                'success': transcription_result.get('status') == 'success'
            })
            
            if transcription_result.get('status') != 'success':
                return {
                    'error': transcription_result.get('error', 'Transcription failed'),
                    'transcription': 'Unable to transcribe audio',
                    'accent': {
                        'name': 'Unknown',
                        'confidence': 0.0,
                        'all_results': []
                    }
                }
            
            transcription = transcription_result.get('transcription', '')
            
            # Step 2: Classify accent
            classification_result = self.accent_agent.process_message({
                'action': 'classify_accent',
                'audio_path': audio_path,
                'transcription': transcription
            })
            
            history_entry['steps'].append({
                'agent': 'AccentClassifierAgent',
                'action': 'classify_accent',
                'success': classification_result.get('status') == 'success'
            })
            
            if classification_result.get('status') != 'success':
                return {
                    'transcription': transcription,
                    'error': classification_result.get('error', 'Accent classification failed'),
                    'accent': {
                        'name': 'Unknown',
                        'confidence': 0.0,
                        'all_results': []
                    }
                }
            
            # Record history
            self.history.append(history_entry)
            
            # Return the combined results
            accent_result = classification_result.get('result', {})
            
            return {
                'transcription': transcription,
                'accent': {
                    'name': accent_result.get('accent', 'Unknown'),
                    'confidence': accent_result.get('confidence', 0.0),
                    'all_results': accent_result.get('all_results', []),
                    'audio_quality': accent_result.get('audio_quality', 0.0)
                }
            }
        
        except Exception as e:
            st.error(f"Error analyzing speech: {str(e)}")
            # Still record this failed attempt in history
            history_entry['steps'].append({
                'agent': 'AgentManager',
                'action': 'analyze_speech',
                'success': False,
                'error': str(e)
            })
            self.history.append(history_entry)
            
            return {
                'error': f"Analysis error: {str(e)}",
                'transcription': 'Error processing audio',
                'accent': {
                    'name': 'Unknown',
                    'confidence': 0.0,
                    'all_results': []
                }
            }
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get history of agent interactions"""
        return self.history