#!/usr/bin/env python3
"""
Simple run script for Media Recommender API
"""

import sys
import os
from pathlib import Path

def check_env():
    """Check if .env file exists"""
    if not Path(".env").exists():
        print("❌ .env file not found!")
        print("Please run: python setup.py")
        return False
    return True

def main():
    """Main function to run the API"""
    print("🎬 Starting Media Recommender API...")
    
    # Check environment
    if not check_env():
        sys.exit(1)
    
    # Import and run the app
    try:
        from main import app
        import uvicorn
        from config import Config
        
        print(f"🚀 Starting server on {Config.APP_HOST}:{Config.APP_PORT}")
        print(f"📖 API Documentation: http://{Config.APP_HOST}:{Config.APP_PORT}/docs")
        print("⏹️  Press Ctrl+C to stop")
        
        uvicorn.run(
            "main:app",
            host=Config.APP_HOST,
            port=Config.APP_PORT,
            reload=True,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 