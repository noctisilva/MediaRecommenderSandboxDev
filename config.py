from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # TMDB Configuration (for Netflix content discovery)
    TMDB_API_KEY = os.getenv("TMDB_API_KEY")
    TMDB_BASE_URL = os.getenv("TMDB_BASE_URL", "https://api.themoviedb.org/3")
    
    # Pinecone Configuration (primary vector store + embeddings)
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "netflix-recommender")
    
    # Pinecone Embedding Model (cheapest option)
    PINECONE_EMBEDDING_MODEL = "multilingual-e5-large"  # $0.08 per 1M tokens, 5M free monthly
    PINECONE_EMBEDDING_DIMENSION = 1024  # multilingual-e5-large dimension
    
    # Gemini Configuration (only for AI explanations, not embeddings)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    
    # Application Configuration
    APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
    APP_PORT = int(os.getenv("APP_PORT", "8000"))
    
    # Netflix Configuration
    NETFLIX_REGION = os.getenv("NETFLIX_REGION", "US")
    NETFLIX_PROVIDER_ID = 8  # TMDB provider ID for Netflix
    
    # Pinecone Cost Information
    PINECONE_COST_PER_MILLION_TOKENS = 0.08  # $0.08 per 1M tokens
    PINECONE_FREE_TOKENS_MONTHLY = 5_000_000  # 5M free tokens per month
    
    # Processing Configuration
    BATCH_SIZE = 25  # Optimal batch size for Pinecone inference
    MAX_OVERVIEW_LENGTH = 400  # Token efficiency for embeddings
    
    # Netflix Genre Mappings (TMDB genre IDs)
    GENRES = {
        "Action": {
            "description": "Fast-paced Netflix action movies and series with intense sequences and high-energy scenes.",
            "tmdb_ids": [28, 10759]  # Movie Action + TV Action & Adventure
        },
        "Adventure": {
            "description": "Netflix adventure content focused on journeys, exploration, and exciting experiences.",
            "tmdb_ids": [12, 10759]  # Movie Adventure + TV Action & Adventure
        },
        "Comedy": {
            "description": "Netflix comedy movies, series, and specials designed to entertain with humor and wit.",
            "tmdb_ids": [35, 10762]  # Movie Comedy + TV Comedy
        },
        "Drama": {
            "description": "Character-driven Netflix dramas exploring serious themes and emotional situations.",
            "tmdb_ids": [18]  # Drama (both movie and TV)
        },
        "Thriller": {
            "description": "Netflix suspense and thriller content focused on generating excitement and anticipation.",
            "tmdb_ids": [53]  # Thriller
        },
        "Horror": {
            "description": "Netflix horror movies and series intended to frighten and create suspense.",
            "tmdb_ids": [27]  # Horror
        },
        "Fantasy": {
            "description": "Netflix fantasy content featuring magical elements and imaginary worlds.",
            "tmdb_ids": [14, 10765]  # Movie Fantasy + TV Sci-Fi & Fantasy
        },
        "Science Fiction": {
            "description": "Netflix sci-fi content exploring futuristic concepts, technology, and alternate realities.",
            "tmdb_ids": [878, 10765]  # Movie Sci-Fi + TV Sci-Fi & Fantasy
        },
        "Animation": {
            "description": "Netflix animated movies, series, and content for all ages.",
            "tmdb_ids": [16, 10762]  # Movie Animation + TV Kids
        },
        "Reality": {
            "description": "Netflix reality TV shows, competitions, and documentary-style programming.",
            "tmdb_ids": [10764]  # TV Reality
        },
        "Talk Show": {
            "description": "Netflix talk shows, interviews, and discussion-based programming.",
            "tmdb_ids": [10767]  # TV Talk
        },
        "Documentary": {
            "description": "Netflix documentaries and factual programming.",
            "tmdb_ids": [99, 10763]  # Movie Documentary + TV Documentary
        },
        "Romance": {
            "description": "Netflix romantic movies and series focusing on love stories.",
            "tmdb_ids": [10749]  # Romance
        },
        "Crime": {
            "description": "Netflix crime dramas, thrillers, and true crime content.",
            "tmdb_ids": [80]  # Crime
        },
        "Family": {
            "description": "Netflix family-friendly content suitable for all ages.",
            "tmdb_ids": [10751, 10762]  # Movie Family + TV Kids
        }
    }
    
    # Netflix Quality Filters
    NETFLIX_MIN_VOTES = 50  # Minimum votes for quality content
    NETFLIX_INCLUDE_ADULT = False
    NETFLIX_INCLUDE_VIDEO = False
    
    # Rate Limiting (for TMDB only now)
    TMDB_REQUESTS_PER_10_SEC = 35
    TMDB_DELAY_BETWEEN_REQUESTS = 0.3