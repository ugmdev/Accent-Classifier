
import os
import sys
import torch

def print_colored(text, color):
    """Print text in color."""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'end': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['end']}")

def main():
    print_colored("Whisper Verification Script", "blue")
    print_colored("=========================", "blue")
    
    print_colored("\nChecking CUDA availability:", "blue")
    print_colored(f"CUDA available: {torch.cuda.is_available()}", "blue")
    if torch.cuda.is_available():
        print_colored(f"CUDA device count: {torch.cuda.device_count()}", "blue")
        print_colored(f"CUDA device name: {torch.cuda.get_device_name(0)}", "blue")
    
    print_colored("\nImporting whisper...", "blue")
    try:
        import whisper
        print_colored("✅ Whisper imported successfully!", "green")
        
        # Get version
        try:
            version = whisper.__version__
            print_colored(f"✅ Whisper version: {version}", "green")
        except AttributeError:
            print_colored("⚠️ Could not determine Whisper version", "yellow")
        
        # Check for load_model function
        if hasattr(whisper, 'load_model'):
            print_colored("✅ whisper.load_model() function available", "green")
        else:
            print_colored("❌ whisper.load_model() function NOT available", "red")
            print_colored("This suggests you're using a different 'whisper' package, not OpenAI's", "red")
            return 1
        
        # Try loading the tiny model (fastest)
        print_colored("\nTrying to load 'tiny' model (this may take a moment)...", "blue")
        try:
            model = whisper.load_model("tiny")
            print_colored("✅ Successfully loaded 'tiny' model!", "green")
            
            # Try basic transcription
            print_colored("\nTrying basic transcription API...", "blue")
            if hasattr(model, 'transcribe'):
                print_colored("✅ model.transcribe() function available", "green")
            else:
                print_colored("❌ model.transcribe() function NOT available", "red")
                
            # Everything looks good!
            print_colored("\n✅ Whisper is properly installed and working!", "green")
            return 0
        except Exception as e:
            print_colored(f"❌ Error loading model: {e}", "red")
            return 1
    except ImportError as e:
        print_colored(f"❌ Failed to import whisper: {e}", "red")
        return 1

if __name__ == "__main__":
    sys.exit(main())
