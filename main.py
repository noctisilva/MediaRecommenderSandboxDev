from __future__ import annotations
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any
import logging
import uvicorn

from models import (
    RecommendationRequest, 
    RecommendationsResponse, 
    HealthResponse
)
from recommendation_service import RecommendationService
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variable for recommendation service
recommendation_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global recommendation_service
    
    # Startup
    try:
        logger.info("🎬 Starting Netflix Media Recommender API with Pinecone Embeddings...")
        recommendation_service = RecommendationService()
        
        # Check service status
        service_info = recommendation_service.get_service_info()
        if "error" not in service_info:
            total_vectors = service_info['vector_database']['total_vectors']
            logger.info(f"✅ Service initialized - {total_vectors} Netflix titles ready")
            
            if total_vectors == 0:
                logger.warning("⚠️ Vector store is empty. Use POST /populate to load Netflix content")
        else:
            logger.error(f"❌ Service initialization issues: {service_info['error']}")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize service: {e}")
        recommendation_service = None
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Netflix recommendation service...")

# Initialize FastAPI app
app = FastAPI(
    title="🎬 Netflix Recommender API - Pinecone Edition",
    description="""
# Netflix Media Recommender with Pinecone Embeddings

High-performance Netflix movie and TV show recommendations using:
- **🔍 Pinecone Embeddings**: multilingual-e5-large model ($0.08/1M tokens, 5M free monthly)
- **📊 Pinecone Vector Database**: Serverless vector storage and search
- **🎬 Netflix-Only Content**: TMDB API with Netflix provider filtering
- **🤖 AI Explanations**: Gemini-powered reasoning and insights

## 🚀 Key Features

- **Cost-Effective**: 5M free tokens monthly = ~50,000 Netflix titles
- **High Performance**: Pinecone's optimized inference and vector search
- **Netflix-Focused**: Only recommends content available on Netflix
- **Smart Explanations**: AI-generated reasons why you'll love each recommendation

## 📋 Quick Start

1. **Populate Netflix Content**: `POST /populate?max_pages=5`
2. **Get Recommendations**: `POST /recommendations`
3. **Monitor Usage**: `GET /cost-summary`

## 💰 Cost Information

- **Free Tier**: 5M tokens/month (≈50,000 movies)
- **Paid Usage**: $0.08 per 1M tokens after free tier
- **Typical Cost**: ~$0.008 per 1,000 movies embedded

## 🎯 Supported Genres

Action, Adventure, Comedy, Drama, Thriller, Horror, Fantasy, Science Fiction, 
Animation, Reality, Talk Show, Documentary, Romance, Crime, Family
    """,
    version="4.0.0-Pinecone",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:3000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

@app.get("/", response_model=HealthResponse)
async def root():
    """🏠 Root endpoint"""
    return HealthResponse(
        status="healthy",
        message="🎬 Netflix Recommender API with Pinecone Embeddings"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """💚 Health check endpoint"""
    try:
        if not recommendation_service:
            return HealthResponse(
                status="unhealthy",
                message="Service not initialized"
            )
        
        service_info = recommendation_service.get_service_info()
        
        if "error" in service_info:
            return HealthResponse(
                status="unhealthy",
                message=f"Service error: {service_info['error']}"
            )
        
        status = service_info['status']
        vector_count = service_info['vector_database']['total_vectors']
        
        if not status['pinecone_available']:
            return HealthResponse(
                status="unhealthy",
                message="Pinecone not available - check API key"
            )
        elif not status['pinecone_inference_available']:
            return HealthResponse(
                status="degraded",
                message="Pinecone vector store available but inference unavailable"
            )
        else:
            return HealthResponse(
                status="healthy",
                message=f"All systems operational - {vector_count} Netflix titles indexed"
            )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}"
        )

@app.get("/service-info")
async def get_service_info():
    """ℹ️ Get comprehensive service information"""
    try:
        if not recommendation_service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        return recommendation_service.get_service_info()
        
    except Exception as e:
        logger.error(f"Error getting service info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cost-summary")
async def get_cost_summary():
    """💰 Get detailed cost breakdown and usage"""
    try:
        if not recommendation_service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        return recommendation_service.get_cost_summary()
        
    except Exception as e:
        logger.error(f"Error getting cost summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/genres")
async def get_genres():
    """🎭 Get available Netflix genres"""
    try:
        if not recommendation_service:
            return {genre: info['description'] for genre, info in Config.GENRES.items()}
        
        return recommendation_service.get_available_genres()
        
    except Exception as e:
        logger.error(f"Error getting genres: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """📊 Get vector store statistics"""
    try:
        if not recommendation_service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        stats = recommendation_service.vector_store.get_index_stats()
        
        # Format for API response
        return {
            "total_netflix_titles": stats.get('total_vector_count', 0),
            "embedding_model": stats.get('embedding_model', 'multilingual-e5-large'),
            "embedding_dimension": stats.get('dimension', 1024),
            "tokens_used_session": stats.get('tokens_used_this_session', 0),
            "estimated_monthly_tokens": stats.get('estimated_monthly_tokens', 0),
            "free_tokens_remaining": max(0, Config.PINECONE_FREE_TOKENS_MONTHLY - stats.get('estimated_monthly_tokens', 0)),
            "index_name": Config.PINECONE_INDEX_NAME
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/populate")
async def populate_netflix_content(background_tasks: BackgroundTasks, max_pages: int = 5):
    """📥 Populate vector store with Netflix content using Pinecone embeddings"""
    try:
        if not recommendation_service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        if not recommendation_service.vector_store.pinecone_available:
            raise HTTPException(status_code=503, detail="Pinecone not available")
        
        if not recommendation_service.vector_store.pinecone_inference_available:
            raise HTTPException(status_code=503, detail="Pinecone inference not available")
        
        # Validate max_pages
        if max_pages < 1 or max_pages > 20:
            raise HTTPException(status_code=400, detail="max_pages must be between 1 and 20")
        
        # Estimate cost before starting
        estimated_items = max_pages * 40  # ~20 movies + 20 TV shows per page
        estimated_tokens = estimated_items * 100  # ~100 tokens per item
        cost_estimate = estimated_tokens * Config.PINECONE_COST_PER_MILLION_TOKENS / 1_000_000
        
        # Run population in background
        background_tasks.add_task(
            recommendation_service.populate_vector_store,
            max_pages
        )
        
        logger.info(f"Started Netflix population with {max_pages} pages using Pinecone embeddings")
        
        return {
            "message": f"Netflix population started with {max_pages} pages",
            "status": "processing",
            "embedding_provider": "pinecone",
            "embedding_model": Config.PINECONE_EMBEDDING_MODEL,
            "estimated_items": estimated_items,
            "estimated_tokens": estimated_tokens,
            "estimated_cost": cost_estimate,
            "free_tokens_available": Config.PINECONE_FREE_TOKENS_MONTHLY,
            "note": "Using Pinecone multilingual-e5-large embeddings for high-quality recommendations"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting population: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations", response_model=RecommendationsResponse)
async def get_netflix_recommendations(request: RecommendationRequest):
    """🎬 Get personalized Netflix recommendations"""
    try:
        if not recommendation_service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        logger.info(f"Netflix recommendation request: '{request.user_input}' | {request.genre} | limit={request.limit}")
        
        result = recommendation_service.get_recommendations(
            user_input=request.user_input,
            genre=request.genre,
            limit=request.limit
        )
        
        if result.get('status') != 'success':
            raise HTTPException(
                status_code=400,
                detail=result.get('message', 'Unknown error')
            )
        
        return RecommendationsResponse(
            recommendations=result['recommendations'],
            total_found=result['total_found']
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear-index")
async def clear_netflix_index():
    """🗑️ Clear all Netflix data from vector store"""
    try:
        if not recommendation_service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        if not recommendation_service.vector_store.pinecone_available:
            raise HTTPException(status_code=503, detail="Pinecone not available")
        
        # Clear the index
        recommendation_service.vector_store.index.delete(delete_all=True)
        
        logger.info("Cleared all Netflix data from Pinecone index")
        
        return {
            "message": "Netflix index cleared successfully",
            "status": "success",
            "note": "Repopulate with POST /populate to restore Netflix content"
        }
        
    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/{genre}")
async def debug_netflix_search(genre: str, query: str = "Netflix", limit: int = 10):
    """🔍 Debug Netflix content search"""
    try:
        if not recommendation_service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        # Get unfiltered results
        all_results = recommendation_service.vector_store.search_similar(
            query=query,
            genre_filter=None,
            top_k=limit
        )
        
        # Get filtered results
        filtered_results = recommendation_service.vector_store.search_similar(
            query=query,
            genre_filter=genre,
            top_k=limit
        )
        
        # Analyze genre distribution
        all_genres = set()
        for movie in all_results:
            for gid in movie.get('genre_ids', []):
                try:
                    all_genres.add(int(float(gid)))
                except:
                    continue
        
        return {
            "query": query,
            "requested_genre": genre,
            "target_genre_ids": Config.GENRES.get(genre, {}).get('tmdb_ids', []),
            "total_results": len(all_results),
            "filtered_results": len(filtered_results),
            "all_genre_ids_in_database": sorted(list(all_genres)),
            "embedding_model": Config.PINECONE_EMBEDDING_MODEL,
            "sample_results": [
                {
                    "title": movie.get('title'),
                    "genre_ids": movie.get('genre_ids'),
                    "media_type": movie.get('media_type'),
                    "similarity_score": movie.get('score')
                }
                for movie in all_results[:5]
            ],
            "filtered_sample": [
                {
                    "title": movie.get('title'),
                    "genre_ids": movie.get('genre_ids'),
                    "media_type": movie.get('media_type'),
                    "similarity_score": movie.get('score')
                }
                for movie in filtered_results[:5]
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in debug search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    host = "127.0.0.1" if Config.APP_HOST == "0.0.0.0" else Config.APP_HOST
    
    print(f"🚀 Starting Netflix Recommender API with Pinecone Embeddings")
    print(f"   🌐 Server: http://{host}:{Config.APP_PORT}")
    print(f"   📚 Docs: http://{host}:{Config.APP_PORT}/docs")
    print(f"   💰 Cost info: http://{host}:{Config.APP_PORT}/cost-summary")
    print(f"   📊 Stats: http://{host}:{Config.APP_PORT}/stats")
    print(f"   🔍 Embedding: {Config.PINECONE_EMBEDDING_MODEL}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=Config.APP_PORT,
        reload=True,
        log_level="info"
    )