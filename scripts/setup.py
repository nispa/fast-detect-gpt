import os
import subprocess
import sys

def create_virtual_environment():
    """Creates a virtual environment named 'env'."""
    print("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", "env"], check=True)

def install_requirements():
    """Installs requirements from requirements.txt."""
    print("Installing requirements...")
    # Determine the correct pip path based on the OS
    if sys.platform == "win32":
        pip_executable = os.path.join("env", "Scripts", "pip.exe")
    else:
        pip_executable = os.path.join("env", "bin", "pip")

    subprocess.run([pip_executable, "install", "-r", "requirements.txt"], check=True)

if __name__ == "__main__":
    # Create virtual environment if it doesn't exist
    if not os.path.exists("env"):
        create_virtual_environment()
    
    # Install requirements
    install_requirements()

    print("\nSetup complete.")
    print("To activate the virtual environment, run:")
    if sys.platform == "win32":
        print(f"  .\\env\\Scripts\\activate")
    else:
        print(f"  source env/bin/activate")
