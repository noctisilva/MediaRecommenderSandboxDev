#!/usr/bin/env python3
"""
Initialize Pinecone index for Media Recommender
"""

import os
import sys
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_pinecone():
    """Initialize Pinecone index"""
    print("🎬 Initializing Pinecone for Media Recommender")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "media-recommender")
    
    # Validate configuration
    if not api_key:
        print("❌ PINECONE_API_KEY not found in .env file")
        return False
    
    print(f"🔑 API Key: {api_key[:10]}...")
    print(f"📊 Index Name: {index_name}")
    
    try:
        # Initialize Pinecone client
        print("\n🔌 Initializing Pinecone...")
        pc = Pinecone(api_key=api_key)
        
        # List existing indexes
        print("📋 Checking existing indexes...")
        existing_indexes = pc.list_indexes()
        existing_names = [idx.name for idx in existing_indexes]
        print(f"Found {len(existing_names)} indexes: {existing_names}")
        
        # Check if our index exists
        if index_name in existing_names:
            print(f"\n✅ Index '{index_name}' already exists")
            
            # Get index stats
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            print(f"📊 Index stats: {stats}")
            
        else:
            print(f"\n⚠️  Index '{index_name}' does not exist")
            print("Creating new index...")
            
            # Create the index with serverless spec
            pc.create_index(
                name=index_name,
                dimension=1536,  # OpenAI text-embedding-ada-002 dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            
            print(f"✅ Successfully created index '{index_name}'")
            print("⏳ Waiting for index to be ready...")
            
            # Wait for index to be ready
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
            
            print("🎯 Index is ready!")
            
            # Verify the index
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            print(f"📊 Index stats: {stats}")
        
        print("\n🎉 Pinecone initialization completed successfully!")
        print("\nNext steps:")
        print("1. Start the server: python run.py")
        print("2. Populate the index: POST /populate")
        print("3. Get recommendations: POST /recommendations")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pinecone initialization failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Verify your Pinecone API key")
        print("3. Make sure your Pinecone account is active")
        print("4. Check if you have enough credits in your Pinecone account")
        print("5. Ensure you're using the latest Pinecone client: pip install -U pinecone-client")
        print("6. If using pods, update the spec to PodSpec instead of ServerlessSpec")
        return False

if __name__ == "__main__":
    success = init_pinecone()
    sys.exit(0 if success else 1)