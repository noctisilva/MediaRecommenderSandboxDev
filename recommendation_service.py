from __future__ import annotations
import logging
from typing import List, Dict, Any
from tmdb_client import TMDBClient
from vector_store import VectorStore
from models import MovieInfo, RecommendationResponse
from config import Config
from smart_query_processor import SmartQueryProcessor

logger = logging.getLogger(__name__)

class RecommendationService:
    def __init__(self):
        self.tmdb_client = TMDBClient()
        self.vector_store = VectorStore()
        
        # Initialize smart query processor with Gemini if available
        gemini_model = self.vector_store.gemini_model if self.vector_store.gemini_available else None
        self.query_processor = SmartQueryProcessor(gemini_model)
        
        logger.info("✅ Netflix Recommendation Service initialized")
        logger.info(f"   📊 Embedding: Pinecone {Config.PINECONE_EMBEDDING_MODEL}")
        logger.info(f"   🎬 Content: Netflix-only via TMDB")
        logger.info(f"   🤖 AI Explanations: {'Gemini' if self.vector_store.gemini_available else 'Static'}")
        logger.info(f"   🧠 Smart Query Processing: {'Enabled with AI' if gemini_model else 'Enabled with patterns'}")
    
    def get_recommendations(self, user_input: str, genre: str, limit: int = 5) -> Dict[str, Any]:
        """Get personalized Netflix recommendations with smart query processing"""
        try:
            # Validate genre
            if genre not in Config.GENRES:
                return {
                    'status': 'error',
                    'message': f"Invalid genre. Available genres: {list(Config.GENRES.keys())}",
                    'available_genres': list(Config.GENRES.keys())
                }
            
            logger.info(f"🎯 Processing query: '{user_input}' | {genre} | limit={limit}")
            
            # Step 1: Process and enhance the query
            query_result = self.query_processor.process_query(user_input, genre)
            
            # Step 2: Handle off-topic queries
            if query_result['status'] == 'off_topic':
                logger.info(f"Off-topic query detected: {user_input}")
                return {
                    'status': 'off_topic',
                    'message': query_result['helpful_suggestion'],
                    'guidance': query_result['guidance'],
                    'example_queries': query_result['example_queries'],
                    'recommendations': [],
                    'total_found': 0
                }
            
            # Step 3: Use enhanced query for search
            enhanced_query = query_result['enhanced_query']
            logger.info(f"🔍 Enhanced query: '{enhanced_query}'")
            
            # Step 4: Search for similar Netflix content
            similar_movies = self.vector_store.search_similar(
                query=enhanced_query,
                genre_filter=genre,
                top_k=limit * 2  # Get more results for better filtering
            )
            
            # Step 5: If no results with enhanced query, try fallback approaches
            if not similar_movies:
                logger.warning("No results with enhanced query, trying fallbacks...")
                
                # Fallback 1: Try original query
                similar_movies = self.vector_store.search_similar(
                    query=user_input,
                    genre_filter=genre,
                    top_k=limit * 2
                )
                
                # Fallback 2: Try generic genre query
                if not similar_movies:
                    generic_query = f"popular {genre.lower()} Netflix content"
                    similar_movies = self.vector_store.search_similar(
                        query=generic_query,
                        genre_filter=genre,
                        top_k=limit * 2
                    )
                
                # Fallback 3: Get any content from this genre
                if not similar_movies:
                    similar_movies = self.vector_store.search_similar(
                        query=genre.lower(),
                        genre_filter=genre,
                        top_k=limit * 2
                    )
            
            if not similar_movies:
                return {
                    'status': 'no_content',
                    'message': f"No {genre} content found in our Netflix database. The database might be empty or need population.",
                    'suggestion': "Try populating the database first with POST /populate, or try a different genre.",
                    'available_genres': list(Config.GENRES.keys()),
                    'recommendations': [],
                    'total_found': 0
                }
            
            # Step 6: Generate recommendations with AI explanations
            recommendations = []
            
            for movie in similar_movies[:limit]:  # Take only requested number
                try:
                    # Generate AI explanation
                    explanation = self.vector_store.generate_explanation(movie, enhanced_query)
                    
                    # Create MovieInfo object
                    movie_info = MovieInfo(
                        id=movie.get('id'),
                        title=movie.get('title', ''),
                        overview=movie.get('overview', ''),
                        release_date=movie.get('release_date'),
                        vote_average=movie.get('vote_average', 0),
                        vote_count=movie.get('vote_count', 0),
                        genre_ids=[int(gid) for gid in movie.get('genre_ids', []) if str(gid).isdigit()],
                        poster_path=movie.get('poster_path'),
                        backdrop_path=movie.get('backdrop_path'),
                        media_type=movie.get('media_type', 'movie')
                    )
                    
                    # Create recommendation response
                    recommendation = RecommendationResponse(
                        movie=movie_info,
                        why_you_would_like_it=explanation['why'],
                        memorable_quotes=explanation['quotes'],
                        memorable_moments=explanation['moments']
                    )
                    
                    recommendations.append(recommendation)
                    
                except Exception as e:
                    logger.error(f"Error processing movie {movie.get('title', 'unknown')}: {e}")
                    continue
            
            logger.info(f"✅ Generated {len(recommendations)} Netflix recommendations")
            
            return {
                'status': 'success',
                'query_processing': {
                    'original_query': user_input,
                    'enhanced_query': enhanced_query,
                    'confidence': query_result['confidence'],
                    'suggestions': query_result.get('suggestions', [])
                },
                'recommendations': recommendations,
                'total_found': len(recommendations)
            }
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return {
                'status': 'error',
                'message': f"An error occurred while processing your request: {str(e)}",
                'recommendations': [],
                'total_found': 0
            }
    
    def get_recommendations_simple(self, user_input: str, genre: str, limit: int = 5) -> List[RecommendationResponse]:
        """Simple version that returns just the recommendations (for backward compatibility)"""
        result = self.get_recommendations(user_input, genre, limit)
        return result.get('recommendations', [])
    
    def populate_vector_store(self, max_pages: int = 5):
        """Populate vector store with Netflix content using Pinecone embeddings"""
        try:
            logger.info(f"🚀 Starting Netflix population with Pinecone embeddings ({max_pages} pages)")
            
            all_netflix_content = []
            
            # Fetch Netflix movies
            logger.info("📽️ Fetching Netflix movies...")
            for page in range(1, max_pages + 1):
                try:
                    movies = self.tmdb_client.get_netflix_movies(page=page)
                    if movies:
                        for movie in movies:
                            movie['media_type'] = 'movie'
                        all_netflix_content.extend(movies)
                        logger.info(f"   ✅ Page {page}: {len(movies)} Netflix movies")
                    else:
                        logger.warning(f"   ⚠️ Page {page}: No movies found")
                except Exception as e:
                    logger.error(f"   ❌ Page {page}: {e}")
                    continue
            
            # Fetch Netflix TV shows
            logger.info("📺 Fetching Netflix TV shows...")
            for page in range(1, max_pages + 1):
                try:
                    tv_shows = self.tmdb_client.get_netflix_tv_shows(page=page)
                    if tv_shows:
                        for show in tv_shows:
                            show['media_type'] = 'tv'
                        all_netflix_content.extend(tv_shows)
                        logger.info(f"   ✅ Page {page}: {len(tv_shows)} Netflix TV shows")
                    else:
                        logger.warning(f"   ⚠️ Page {page}: No TV shows found")
                except Exception as e:
                    logger.error(f"   ❌ Page {page}: {e}")
                    continue
            
            # Remove duplicates
            seen = set()
            unique_content = []
            for item in all_netflix_content:
                key = (item.get('id'), item.get('media_type'))
                if key not in seen:
                    seen.add(key)
                    unique_content.append(item)
            
            logger.info(f"📊 Collected {len(unique_content)} unique Netflix titles (from {len(all_netflix_content)} total)")
            
            # Upsert to vector store using Pinecone embeddings
            if unique_content:
                logger.info("🔄 Creating Pinecone embeddings and upserting to vector store...")
                self.vector_store.upsert_movies(unique_content)
                
                # Get final stats
                try:
                    stats = self.vector_store.get_index_stats()
                    total_vectors = stats.get('total_vector_count', 0)
                    tokens_used = stats.get('tokens_used_this_session', 0)
                    logger.info(f"🎉 Population completed successfully!")
                    logger.info(f"   📈 Total Netflix titles in database: {total_vectors}")
                    logger.info(f"   💰 Tokens used: {tokens_used:,}")
                    
                    # Cost information
                    free_limit = Config.PINECONE_FREE_TOKENS_MONTHLY
                    if tokens_used < free_limit:
                        remaining = free_limit - tokens_used
                        logger.info(f"   🆓 Remaining free tokens: {remaining:,}")
                    else:
                        cost = (tokens_used - free_limit) * Config.PINECONE_COST_PER_MILLION_TOKENS / 1_000_000
                        logger.info(f"   💵 Estimated cost: ${cost:.4f}")
                        
                except Exception as e:
                    logger.warning(f"Could not get final stats: {e}")
            else:
                logger.warning("⚠️ No Netflix content collected")
                logger.warning("Check your TMDB API key and Netflix region settings")
            
        except Exception as e:
            logger.error(f"❌ Error populating vector store: {e}")
            raise
    
    def get_available_genres(self) -> Dict[str, str]:
        """Get available Netflix genres with descriptions"""
        return {genre: info['description'] for genre, info in Config.GENRES.items()}
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get comprehensive service information"""
        try:
            stats = self.vector_store.get_index_stats()
            
            return {
                "service_type": "Netflix Media Recommender with Pinecone Embeddings",
                "embedding_model": Config.PINECONE_EMBEDDING_MODEL,
                "embedding_provider": "pinecone",
                "ai_explanations": "gemini" if self.vector_store.gemini_available else "static",
                "content_source": "Netflix via TMDB",
                "vector_database": {
                    "provider": "pinecone",
                    "total_vectors": stats.get('total_vector_count', 0),
                    "dimension": Config.PINECONE_EMBEDDING_DIMENSION,
                    "index_name": Config.PINECONE_INDEX_NAME
                },
                "cost_info": {
                    "embedding_cost_per_million_tokens": Config.PINECONE_COST_PER_MILLION_TOKENS,
                    "free_tokens_monthly": Config.PINECONE_FREE_TOKENS_MONTHLY,
                    "tokens_used_this_session": stats.get('tokens_used_this_session', 0),
                    "estimated_monthly_tokens": stats.get('estimated_monthly_tokens', 0)
                },
                "status": {
                    "pinecone_available": self.vector_store.pinecone_available,
                    "pinecone_inference_available": self.vector_store.pinecone_inference_available,
                    "gemini_available": self.vector_store.gemini_available,
                    "tmdb_available": True  # Assumed if service starts
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting service info: {e}")
            return {"error": str(e)}
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get detailed cost breakdown"""
        try:
            stats = self.vector_store.get_index_stats()
            tokens_used = stats.get('tokens_used_this_session', 0)
            monthly_tokens = stats.get('estimated_monthly_tokens', 0)
            
            free_limit = Config.PINECONE_FREE_TOKENS_MONTHLY
            cost_per_million = Config.PINECONE_COST_PER_MILLION_TOKENS
            
            # Calculate costs
            if monthly_tokens <= free_limit:
                monthly_cost = 0.0
                remaining_free = free_limit - monthly_tokens
            else:
                overage = monthly_tokens - free_limit
                monthly_cost = overage * cost_per_million / 1_000_000
                remaining_free = 0
            
            # Estimate capacity with remaining free tokens
            avg_tokens_per_movie = 100  # Rough estimate
            movies_possible = remaining_free // avg_tokens_per_movie if remaining_free > 0 else 0
            
            return {
                "current_session": {
                    "tokens_used": tokens_used,
                    "estimated_cost": tokens_used * cost_per_million / 1_000_000 if tokens_used > free_limit else 0.0
                },
                "monthly_estimate": {
                    "tokens_used": monthly_tokens,
                    "free_tokens_remaining": remaining_free,
                    "estimated_cost": monthly_cost,
                    "movies_possible_with_free_tokens": movies_possible
                },
                "pricing": {
                    "free_tokens_monthly": free_limit,
                    "cost_per_million_tokens": cost_per_million,
                    "cost_per_1000_movies": cost_per_million * avg_tokens_per_movie / 1_000_000
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting cost summary: {e}")
            return {"error": str(e)}
    
    def get_netflix_stats(self) -> Dict[str, Any]:
        """Get Netflix-specific statistics"""
        try:
            stats = self.vector_store.get_index_stats()
            cost_info = self.get_cost_summary()
            
            return {
                "netflix_titles_indexed": stats.get('total_vector_count', 0),
                "embedding_model": Config.PINECONE_EMBEDDING_MODEL,
                "embedding_dimension": Config.PINECONE_EMBEDDING_DIMENSION,
                "index_name": Config.PINECONE_INDEX_NAME,
                "content_source": "Netflix via TMDB API",
                "regions_supported": [Config.NETFLIX_REGION],
                "genres_supported": len(Config.GENRES),
                "cost_summary": cost_info,
                "service_status": {
                    "pinecone_available": self.vector_store.pinecone_available,
                    "pinecone_inference_available": self.vector_store.pinecone_inference_available,
                    "gemini_explanations_available": self.vector_store.gemini_available,
                    "netflix_content_available": stats.get('total_vector_count', 0) > 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting Netflix stats: {e}")
            return {"error": str(e)}
    
    def test_recommendation_quality(self, test_queries: List[str] = None) -> Dict[str, Any]:
        """Test recommendation quality with sample queries"""
        if test_queries is None:
            test_queries = [
                "action movies with great fight scenes",
                "romantic comedies for date night",
                "sci-fi series with complex plots",
                "family-friendly animated movies",
                "crime documentaries"
            ]
        
        results = []
        
        for query in test_queries:
            try:
                # Try to get recommendations for each test query
                recommendations = self.get_recommendations(
                    user_input=query,
                    genre="Action",  # Default genre for testing
                    limit=3
                )
                
                results.append({
                    "query": query,
                    "found_recommendations": len(recommendations),
                    "success": len(recommendations) > 0,
                    "sample_titles": [rec.movie.title for rec in recommendations[:2]]
                })
                
            except Exception as e:
                results.append({
                    "query": query,
                    "found_recommendations": 0,
                    "success": False,
                    "error": str(e)
                })
        
        successful_tests = sum(1 for r in results if r['success'])
        
        return {
            "test_summary": {
                "total_tests": len(test_queries),
                "successful_tests": successful_tests,
                "success_rate": successful_tests / len(test_queries) * 100
            },
            "test_results": results,
            "recommendation": "populate with POST /populate" if successful_tests == 0 else "system working well"
        }
    
    def estimate_population_cost(self, max_pages: int) -> Dict[str, Any]:
        """Estimate cost for populating Netflix content"""
        # Rough estimates based on TMDB API responses
        movies_per_page = 20
        tv_shows_per_page = 20
        total_items = (movies_per_page + tv_shows_per_page) * max_pages
        
        # Average tokens per item (title + overview + metadata)
        avg_tokens_per_item = 100
        total_tokens = total_items * avg_tokens_per_item
        
        # Cost calculation
        free_limit = Config.PINECONE_FREE_TOKENS_MONTHLY
        cost_per_million = Config.PINECONE_COST_PER_MILLION_TOKENS
        
        if total_tokens <= free_limit:
            cost = 0.0
            using_free_tier = True
        else:
            overage = total_tokens - free_limit
            cost = overage * cost_per_million / 1_000_000
            using_free_tier = False
        
        return {
            "estimation": {
                "max_pages": max_pages,
                "estimated_items": total_items,
                "estimated_tokens": total_tokens,
                "estimated_cost": cost,
                "using_free_tier": using_free_tier
            },
            "breakdown": {
                "movies_per_page": movies_per_page,
                "tv_shows_per_page": tv_shows_per_page,
                "avg_tokens_per_item": avg_tokens_per_item,
                "free_tokens_available": free_limit,
                "cost_per_million_tokens": cost_per_million
            },
            "recommendation": f"Use max_pages={max_pages}" if using_free_tier else f"Consider max_pages={free_limit // avg_tokens_per_item // 40} to stay in free tier"
        }