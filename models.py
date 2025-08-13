from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any

class RecommendationRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_input": "I love action movies with great fight scenes and car chases",
                "genre": "Action",
                "limit": 3
            }
        }
    )
    
    user_input: str = Field(
        ..., 
        description="Your preferences, interests, or description of what you're looking for",
        examples=["I love action movies with great fight scenes and car chases"]
    )
    genre: str = Field(
        ..., 
        description="Specific genre to filter recommendations",
        examples=["Action"]
    )
    limit: Optional[int] = Field(
        default=5, 
        description="Number of recommendations to return (1-10)",
        ge=1,
        le=10,
        examples=[3]
    )

class MovieInfo(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 12345,
                "title": "Fast & Furious 9",
                "overview": "Dom and the crew must take on an international terrorist...",
                "release_date": "2021-06-25",
                "vote_average": 7.2,
                "vote_count": 1234,
                "genre_ids": [28, 12, 53],
                "poster_path": "/poster.jpg",
                "backdrop_path": "/backdrop.jpg",
                "media_type": "movie"
            }
        }
    )
    
    id: int = Field(..., description="TMDB movie/show ID")
    title: str = Field(..., description="Movie/show title")
    overview: str = Field(..., description="Brief description of the movie/show")
    release_date: Optional[str] = Field(None, description="Release date (YYYY-MM-DD)")
    vote_average: float = Field(..., description="Average rating (0-10)")
    vote_count: int = Field(..., description="Number of votes")
    genre_ids: List[int] = Field(..., description="List of genre IDs")
    poster_path: Optional[str] = Field(None, description="Poster image path")
    backdrop_path: Optional[str] = Field(None, description="Backdrop image path")
    media_type: str = Field(..., description="Type of media: 'movie' or 'tv'")

class RecommendationResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "movie": {
                    "id": 12345,
                    "title": "Fast & Furious 9",
                    "overview": "Dom and the crew must take on an international terrorist...",
                    "release_date": "2021-06-25",
                    "vote_average": 7.2,
                    "vote_count": 1234,
                    "genre_ids": [28, 12, 53],
                    "poster_path": "/poster.jpg",
                    "backdrop_path": "/backdrop.jpg",
                    "media_type": "movie"
                },
                "why_you_would_like_it": "Based on your love for action movies with great fight scenes and car chases, you'll absolutely love Fast & Furious 9!",
                "memorable_quotes": [
                    "I don't have friends. I got family.",
                    "Ride or die, remember?"
                ],
                "memorable_moments": [
                    "The epic car chase through the streets of London",
                    "The gravity-defying car jump across a massive canyon"
                ]
            }
        }
    )
    
    movie: MovieInfo = Field(..., description="Movie/show information")
    why_you_would_like_it: str = Field(
        ..., 
        description="AI-generated explanation of why you would enjoy this movie/show"
    )
    memorable_quotes: List[str] = Field(
        ..., 
        description="AI-generated memorable quotes from the movie/show"
    )
    memorable_moments: List[str] = Field(
        ..., 
        description="AI-generated memorable moments or scenes"
    )

class RecommendationsResponse(BaseModel):
    recommendations: List[RecommendationResponse] = Field(
        ..., 
        description="List of personalized recommendations"
    )
    total_found: int = Field(
        ..., 
        description="Total number of recommendations found"
    )

class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "message": "🎬 Media Recommender API is running"
            }
        }
    )
    
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")

class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Validation Error",
                "message": "Invalid genre provided"
            }
        }
    )
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")

class CustomRecommendationRequest(BaseModel):
    user_input: str = Field(
        ..., 
        description="Your preferences, interests, or description of what you're looking for",
        example="I love action movies with great fight scenes and car chases"
    )
    genre: str = Field(
        ..., 
        description="Specific genre to filter recommendations",
        example="Action"
    )
    limit: Optional[int] = Field(
        default=5, 
        description="Number of recommendations to return (1-10)",
        ge=1,
        le=10,
        example=3
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description="Custom embedding model to use (e.g., 'text-embedding-3-small', 'text-embedding-ada-002')",
        example="text-embedding-3-small"
    )
    chat_model: Optional[str] = Field(
        default=None,
        description="Custom chat model to use for explanations (e.g., 'gpt-3.5-turbo', 'gpt-4')",
        example="gpt-3.5-turbo"
    )

    class Config:
        schema_extra = {
            "example": {
                "user_input": "I love action movies with great fight scenes and car chases",
                "genre": "Action",
                "limit": 3,
                "embedding_model": "text-embedding-3-small",
                "chat_model": "gpt-3.5-turbo"
            }
        }