#!/usr/bin/env python3
"""
Setup script for Media Recommender API
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.11"""
    if sys.version_info < (3, 11):
        print("❌ Error: Python 3.11 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_env_file():
    """Check if .env file exists"""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        print("Please copy .env.example to .env and fill in your API keys")
        return False
    print("✅ .env file found")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "pinecone-client",
        "requests",
        "python-dotenv",
        "pydantic",
        "openai",
        "numpy"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_api_keys():
    """Check if API keys are set in .env file"""
    from dotenv import load_dotenv
    load_dotenv()
    
    required_keys = [
        "TMDB_API_KEY",
        "PINECONE_API_KEY", 
        "PINECONE_ENVIRONMENT",
        "OPENAI_API_KEY"
    ]
    
    missing_keys = []
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        print(f"❌ Missing API keys in .env: {', '.join(missing_keys)}")
        return False
    
    print("✅ All API keys are set")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def create_env_file():
    """Create .env file from .env.example"""
    if Path(".env").exists():
        print("✅ .env file already exists")
        return True
    
    if not Path(".env.example").exists():
        print("❌ .env.example not found")
        return False
    
    try:
        import shutil
        shutil.copy(".env.example", ".env")
        print("✅ Created .env file from .env.example")
        print("⚠️  Please edit .env file with your API keys")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def main():
    """Main setup function"""
    print("🎬 Media Recommender API Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create .env file if it doesn't exist
    if not check_env_file():
        if not create_env_file():
            return False
    
    # Check dependencies
    if not check_dependencies():
        print("\nInstalling dependencies...")
        if not install_dependencies():
            return False
    
    # Check API keys
    if not check_api_keys():
        print("\n⚠️  Please set your API keys in the .env file:")
        print("   - TMDB_API_KEY")
        print("   - PINECONE_API_KEY")
        print("   - PINECONE_ENVIRONMENT")
        print("   - OPENAI_API_KEY")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Start the server: python run.py")
    print("2. Visit http://localhost:8000/docs for API documentation")
    print("3. Populate the vector store: POST /populate")
    print("4. Get recommendations: POST /recommendations")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 