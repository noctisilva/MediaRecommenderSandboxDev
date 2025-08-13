#!/usr/bin/env python3
"""
Open Swagger UI for Media Recommender API
"""

import webbrowser
import time
import requests
from config import Config

def check_server():
    """Check if the server is running"""
    try:
        response = requests.get(f"http://{Config.APP_HOST}:{Config.APP_PORT}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Open Swagger UI"""
    print("🎬 Media Recommender API - Swagger UI")
    print("=" * 40)
    
    # Check if server is running
    print("🔍 Checking if server is running...")
    if not check_server():
        print("❌ Server is not running!")
        print("Please start the server first:")
        print("   python run.py")
        return
    
    print("✅ Server is running!")
    
    # URLs
    swagger_url = f"http://{Config.APP_HOST}:{Config.APP_PORT}/docs"
    redoc_url = f"http://{Config.APP_HOST}:{Config.APP_PORT}/redoc"
    standalone_url = f"file://{__file__.replace('open_swagger.py', 'swagger_ui.html')}"
    
    print(f"\n📖 Available documentation:")
    print(f"   Swagger UI: {swagger_url}")
    print(f"   ReDoc: {redoc_url}")
    print(f"   Standalone: {standalone_url}")
    
    # Open Swagger UI
    print(f"\n🚀 Opening Swagger UI...")
    webbrowser.open(swagger_url)
    
    print("✅ Swagger UI opened in your browser!")
    print("\n💡 Tips:")
    print("   - Try the /health endpoint first")
    print("   - Check /genres to see available genres")
    print("   - Use /populate to load data (run this first)")
    print("   - Then try /recommendations with your preferences")

if __name__ == "__main__":
    main() 