#!/usr/bin/env python3
"""
Fix for soundfile/libsndfile issues in Anaconda environments
This script will help resolve the 'cannot load library libsndfile.dylib' error
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
import site
import importlib.util

def print_colored(text, color):
    """Print text in color."""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'end': '\033[0m',
        'bold': '\033[1m'
    }
    print(f"{colors.get(color, '')}{text}{colors['end']}")

def check_soundfile():
    """Check if soundfile is installed and can be loaded."""
    try:
        import soundfile as sf
        print_colored(f"✓ soundfile is installed (version: {sf.__version__})", "green")
        
        # Try loading a simple file to check if libsndfile works
        try:
            # Create a simple WAV file
            import numpy as np
            from scipy.io import wavfile
            
            # Generate a simple sine wave
            sample_rate = 16000
            duration = 0.1  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            
            # Save as WAV using scipy (doesn't require libsndfile)
            test_file = "test_soundfile.wav"
            wavfile.write(test_file, sample_rate, audio_data)
            
            # Try to load with soundfile
            data, samplerate = sf.read(test_file)
            print_colored("✓ libsndfile is working correctly", "green")
            return True
        except Exception as e:
            print_colored(f"✗ libsndfile is not working: {str(e)}", "red")
            return False
    except ImportError:
        print_colored("✗ soundfile is not installed", "red")
        return False
    except Exception as e:
        print_colored(f"✗ Error importing soundfile: {str(e)}", "red")
        return False

def find_libsndfile():
    """Find libsndfile on the system."""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        try:
            # Check if Homebrew is installed
            result = subprocess.run(["brew", "--version"], 
                                   capture_output=True, text=True)
            
            if result.returncode == 0:
                # Try to find libsndfile using brew
                result = subprocess.run(["brew", "list", "libsndfile"], 
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Get the actual library path
                    lib_path = Path(result.stdout.strip().split("\n")[-1])
                    if lib_path.exists():
                        print_colored(f"Found libsndfile via Homebrew: {lib_path}", "green")
                        return lib_path
                    
                    # Try using find command
                    result = subprocess.run(["find", "/usr/local/Cellar/libsndfile", "-name", "*.dylib"],
                                          capture_output=True, text=True)
                    if result.returncode == 0 and result.stdout.strip():
                        lib_path = Path(result.stdout.strip().split("\n")[0])
                        if lib_path.exists():
                            print_colored(f"Found libsndfile in Homebrew Cellar: {lib_path}", "green")
                            return lib_path
        except:
            pass
        
        # Check common locations
        common_paths = [
            "/usr/local/lib/libsndfile.dylib",
            "/usr/local/lib/libsndfile.1.dylib",
            "/opt/homebrew/lib/libsndfile.dylib",
            "/opt/homebrew/lib/libsndfile.1.dylib"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                print_colored(f"Found libsndfile at: {path}", "green")
                return Path(path)
    
    elif system == "Linux":
        # Check common Linux paths
        common_paths = [
            "/usr/lib/x86_64-linux-gnu/libsndfile.so",
            "/usr/lib/x86_64-linux-gnu/libsndfile.so.1",
            "/usr/lib/libsndfile.so",
            "/usr/lib/libsndfile.so.1"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                print_colored(f"Found libsndfile at: {path}", "green")
                return Path(path)
        
        # Try using ldconfig
        try:
            result = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "libsndfile.so" in line:
                        path = line.split("=>")[-1].strip()
                        if os.path.exists(path):
                            print_colored(f"Found libsndfile via ldconfig: {path}", "green")
                            return Path(path)
        except:
            pass
    
    # Not found
    print_colored("Could not find libsndfile on your system", "yellow")
    return None

def install_libsndfile():
    """Install libsndfile using package manager."""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        print_colored("Attempting to install libsndfile via Homebrew...", "blue")
        try:
            subprocess.run(["brew", "install", "libsndfile"], check=True)
            print_colored("Successfully installed libsndfile", "green")
            return True
        except Exception as e:
            print_colored(f"Failed to install libsndfile: {str(e)}", "red")
            print_colored("Please install it manually: brew install libsndfile", "yellow")
            return False
    
    elif system == "Linux":
        # Try to detect the Linux distribution and package manager
        try:
            if os.path.exists("/etc/debian_version"):
                # Debian/Ubuntu
                print_colored("Detected Debian/Ubuntu, using apt...", "blue")
                subprocess.run(["sudo", "apt-get", "update"], check=True)
                subprocess.run(["sudo", "apt-get", "install", "-y", "libsndfile1"], check=True)
                print_colored("Successfully installed libsndfile", "green")
                return True
            elif os.path.exists("/etc/fedora-release"):
                # Fedora
                print_colored("Detected Fedora, using dnf...", "blue")
                subprocess.run(["sudo", "dnf", "install", "-y", "libsndfile"], check=True)
                print_colored("Successfully installed libsndfile", "green")
                return True
            elif os.path.exists("/etc/redhat-release"):
                # RHEL/CentOS
                print_colored("Detected RHEL/CentOS, using yum...", "blue")
                subprocess.run(["sudo", "yum", "install", "-y", "libsndfile"], check=True)
                print_colored("Successfully installed libsndfile", "green")
                return True
            else:
                print_colored("Could not detect Linux distribution, please install libsndfile manually", "yellow")
                return False
        except Exception as e:
            print_colored(f"Failed to install libsndfile: {str(e)}", "red")
            print_colored("Please install it manually using your distribution's package manager", "yellow")
            return False
    
    else:
        print_colored(f"Unsupported system: {system}", "red")
        print_colored("Please install libsndfile manually", "yellow")
        return False

def find_soundfile_package_dir():
    """Find where the soundfile package is installed."""
    try:
        import soundfile
        package_dir = Path(soundfile.__file__).parent
        print_colored(f"Found soundfile package at: {package_dir}", "green")
        return package_dir
    except:
        print_colored("Could not locate soundfile package", "red")
        return None

def fix_anaconda_soundfile():
    """Fix the soundfile library in Anaconda environment."""
    # First check if we're in an Anaconda environment
    in_anaconda = "anaconda" in sys.prefix.lower() or "conda" in sys.prefix.lower()
    
    if in_anaconda:
        print_colored(f"Detected Anaconda environment: {sys.prefix}", "blue")
    else:
        print_colored("Not running in an Anaconda environment", "yellow")
    
    # Find soundfile package directory
    package_dir = find_soundfile_package_dir()
    if not package_dir:
        return False
    
    # Check for _soundfile_data directory
    data_dir = package_dir / "_soundfile_data"
    if not data_dir.exists():
        data_dir.mkdir(exist_ok=True)
        print_colored(f"Created directory: {data_dir}", "green")
    
    # Find libsndfile on the system
    libsndfile_path = find_libsndfile()
    if not libsndfile_path:
        # Try to install it
        print_colored("libsndfile not found, attempting to install...", "yellow")
        if not install_libsndfile():
            print_colored("Could not install libsndfile. Please install it manually.", "red")
            return False
        
        # Try to find it again
        libsndfile_path = find_libsndfile()
        if not libsndfile_path:
            print_colored("Still could not find libsndfile after installation.", "red")
            return False
    
    # Create symlinks or copy the library to _soundfile_data
    try:
        # Determine target filename based on platform
        if platform.system() == "Darwin":
            target_filename = "libsndfile.dylib"
        elif platform.system() == "Linux":
            target_filename = "libsndfile.so"
        else:
            target_filename = os.path.basename(libsndfile_path)
        
        target_path = data_dir / target_filename
        
        # Try creating a symlink first
        try:
            if target_path.exists():
                target_path.unlink()
            
            os.symlink(libsndfile_path, target_path)
            print_colored(f"Created symlink: {libsndfile_path} -> {target_path}", "green")
        except:
            # If symlink fails, try copying the file
            print_colored("Symlink failed, trying to copy the library...", "yellow")
            shutil.copy2(libsndfile_path, target_path)
            print_colored(f"Copied {libsndfile_path} to {target_path}", "green")
        
        return True
    except Exception as e:
        print_colored(f"Error fixing soundfile: {str(e)}", "red")
        return False

def fix_soundfile_issues():
    """Main function to fix soundfile/libsndfile issues."""
    print_colored("SoundFile Fixer for Accent Classifier", "blue")
    print_colored("====================================", "blue")
    
    # Check if soundfile and libsndfile are working
    if check_soundfile():
        print_colored("soundfile and libsndfile are working correctly!", "green")
        return True
    
    # Try to fix the issue in Anaconda environment
    print_colored("\nAttempting to fix soundfile in your environment...", "blue")
    if fix_anaconda_soundfile():
        print_colored("\nFix applied successfully!", "green")
        
        # Verify the fix worked
        print_colored("\nVerifying fix...", "blue")
        if check_soundfile():
            print_colored("soundfile and libsndfile are now working correctly!", "green")
            
            # Create a simple example script that demonstrates working with soundfile
            create_test_script()
            
            print_colored("\nYou can now continue using the Accent Classifier app.", "green")
            return True
        else:
            print_colored("Fix was applied but soundfile still doesn't work correctly.", "red")
    
    # If we couldn't fix it automatically, provide manual instructions
    print_colored("\nAutomatic fix failed. Here are some things you can try manually:", "yellow")
    print_colored("1. Install libsndfile using your package manager:", "yellow")
    print_colored("   - macOS: brew install libsndfile", "yellow")
    print_colored("   - Ubuntu/Debian: sudo apt-get install libsndfile1", "yellow")
    print_colored("   - RHEL/CentOS: sudo yum install libsndfile", "yellow")
    print_colored("   - Fedora: sudo dnf install libsndfile", "yellow")
    
    print_colored("\n2. Reinstall soundfile with pip:", "yellow")
    print_colored("   pip uninstall -y soundfile pysoundfile", "yellow")
    print_colored("   pip install soundfile", "yellow")
    
    print_colored("\n3. If you're using Anaconda, you can try:", "yellow")
    print_colored("   conda install -c conda-forge libsndfile", "yellow")
    print_colored("   conda install -c conda-forge python-soundfile", "yellow")
    
    print_colored("\n4. Consider using an alternative audio loading method:", "yellow")
    print_colored("   - The Accent Classifier app has fallback mechanisms for audio loading", "yellow")
    print_colored("   - You can modify app/app.py to prioritize other audio loading methods", "yellow")
    
    return False

def create_test_script():
    """Create a test script that demonstrates working with soundfile."""
    script_path = Path("/Users/ganesh/Desktop/Accent Classifier/tests/test_soundfile.py")
    script_path.parent.mkdir(exist_ok=True)
    
    with open(script_path, "w") as f:
        f.write("""#!/usr/bin/env python3
\"\"\"
Test script for the soundfile library.
This script demonstrates that soundfile is working correctly.
\"\"\"

import numpy as np
import soundfile as sf
import os

def main():
    # Define parameters for our test tone
    sample_rate = 16000
    duration = 1  # seconds
    frequency = 440  # A4 note
    
    # Generate a sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    # Path for our test file
    test_file = "test_tone.wav"
    
    # Write audio using soundfile
    print(f"Writing audio to {test_file}...")
    sf.write(test_file, tone, sample_rate)
    print(f"File created: {os.path.abspath(test_file)}")
    
    # Read it back
    print("Reading audio file...")
    data, sr = sf.read(test_file)
    
    # Print stats
    print(f"Audio file stats:")
    print(f"- Sample rate: {sr} Hz")
    print(f"- Duration: {len(data)/sr:.2f} seconds")
    print(f"- Shape: {data.shape}")
    print(f"- Min value: {data.min():.6f}")
    print(f"- Max value: {data.max():.6f}")
    
    print("\\nSoundfile is working correctly!")
    return 0

if __name__ == "__main__":
    main()
""")
    
    print_colored(f"Created test script at: {script_path}", "green")
    print_colored(f"You can run it with: python {script_path}", "green")

if __name__ == "__main__":
    fix_soundfile_issues()
