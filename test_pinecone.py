#!/usr/bin/env python3
"""
Test Pinecone connection and debug issues
"""

import os
import sys
from dotenv import load_dotenv

def test_pinecone_version():
    """Check Pinecone version and inference support"""
    try:
        import pinecone
        version = getattr(pinecone, '__version__', 'unknown')
        print(f"✅ Pinecone version: {version}")
        
        # Check if inference is available
        from pinecone import Pinecone
        pc = Pinecone(api_key="dummy")
        
        if hasattr(pc, 'inference'):
            print("✅ Pinecone inference API is available")
            return True
        else:
            print("❌ Pinecone inference API not available")
            print("💡 Upgrade with: pip install --upgrade pinecone-client")
            return False
            
    except ImportError:
        print("❌ Pinecone not installed")
        print("💡 Install with: pip install pinecone-client")
        return False

def test_pinecone_connection():
    """Test Pinecone connection step by step"""
    print("🔍 Testing Pinecone Connection")
    print("=" * 40)
    
    # Check version first
    if not test_pinecone_version():
        return False
    
    # Load environment
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "netflix-recommender")
    
    if not api_key:
        print("❌ PINECONE_API_KEY not found in .env file")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...")
    print(f"✅ Index name: {index_name}")
    
    try:
        # Test basic Pinecone import and initialization
        print("\n📦 Testing Pinecone import...")
        from pinecone import Pinecone, ServerlessSpec
        print("✅ Pinecone imported successfully")
        
        # Initialize client
        print("\n🔌 Initializing Pinecone client...")
        pc = Pinecone(api_key=api_key)
        print("✅ Pinecone client initialized")
        
        # Test inference API if available
        print("\n🧠 Testing Pinecone inference...")
        if hasattr(pc, 'inference'):
            try:
                response = pc.inference.embed(
                    model="multilingual-e5-large",
                    inputs=["test connection"],
                    parameters={"input_type": "passage", "truncate": "END"}
                )
                
                if response and len(response) > 0:
                    embedding = response[0]['values']
                    tokens_used = response.usage.get('total_tokens', 0)
                    print(f"✅ Pinecone inference working: {len(embedding)}D embedding, {tokens_used} tokens")
                else:
                    print("❌ Pinecone inference response is empty")
                    return False
                    
            except Exception as e:
                print(f"❌ Pinecone inference failed: {e}")
                print("⚠️ Will fall back to hash-based embeddings")
        else:
            print("⚠️ Pinecone inference not available in this version")
            print("💡 Upgrade to get Pinecone embeddings: pip install --upgrade pinecone-client")
            print("✅ Will use hash-based embeddings as fallback")
        
        # Test index operations
        print("\n📊 Testing index operations...")
        existing_indexes = list(pc.list_indexes())
        print(f"✅ Found {len(existing_indexes)} existing indexes")
        
        if index_name in [idx.name for idx in existing_indexes]:
            print(f"✅ Index '{index_name}' exists")
            
            # Test index stats
            index = pc.Index(index_name)
            try:
                stats = index.describe_index_stats()
                
                # Handle different response types
                if hasattr(stats, 'total_vector_count'):
                    vector_count = stats.total_vector_count
                elif hasattr(stats, '__dict__'):
                    vector_count = getattr(stats, 'total_vector_count', 0)
                elif isinstance(stats, dict):
                    vector_count = stats.get('total_vector_count', 0)
                else:
                    vector_count = "unknown"
                
                print(f"✅ Index stats retrieved: {vector_count} vectors")
                
            except Exception as e:
                print(f"⚠️ Could not get index stats: {e}")
        else:
            print(f"ℹ️ Index '{index_name}' does not exist (will be created automatically)")
        
        print("\n🎉 Pinecone connection tests completed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try: pip install --upgrade pinecone-client")
        return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check your Pinecone API key is correct")
        print("2. Ensure you have internet connection")
        print("3. Verify your Pinecone account is active")
        print("4. Check if you have credits/quota remaining")
        return False

def test_vector_store():
    """Test the VectorStore class"""
    print("\n" + "=" * 40)
    print("🧪 Testing VectorStore Class")
    print("=" * 40)
    
    try:
        from vector_store import VectorStore
        print("✅ VectorStore imported successfully")
        
        # Initialize vector store
        print("\n🔄 Initializing VectorStore...")
        vs = VectorStore()
        
        print(f"   Pinecone available: {vs.pinecone_available}")
        print(f"   Pinecone inference available: {vs.pinecone_inference_available}")
        print(f"   Gemini available: {vs.gemini_available}")
        
        if vs.pinecone_available:
            print("\n📊 Getting index stats...")
            stats = vs.get_index_stats()
            
            if "error" not in stats:
                vector_count = stats.get('total_vector_count', 0)
                tokens_used = stats.get('tokens_used_this_session', 0)
                embedding_method = "Pinecone" if vs.pinecone_inference_available else "hash-based"
                print(f"✅ Stats retrieved: {vector_count} vectors, {tokens_used} tokens used")
                print(f"✅ Embedding method: {embedding_method}")
            else:
                print(f"⚠️ Stats error: {stats['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ VectorStore test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🎬 Netflix Recommender - Pinecone Debug")
    print("=" * 50)
    
    success = True
    
    # Test basic connection
    if not test_pinecone_connection():
        success = False
    
    # Test vector store class
    if not test_vector_store():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Tests completed! Your Pinecone setup is working.")
        print("\n💡 Next steps:")
        print("1. Start your API: python run.py")
        print("2. Populate Netflix content: POST /populate")
        print("3. Get recommendations: POST /recommendations")
        print("\n📝 Note:")
        print("   - If Pinecone inference is available: High-quality embeddings")
        print("   - If only hash-based: Still functional but lower accuracy")
        print("   - Upgrade Pinecone client for best results!")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\n🔧 Quick fixes:")
        print("1. pip install --upgrade pinecone-client")
        print("2. Check your .env file has PINECONE_API_KEY")
        print("3. Verify your Pinecone account is active")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)