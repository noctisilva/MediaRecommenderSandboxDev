from __future__ import annotations
import logging
import time
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        """Initialize vector store with Pinecone embeddings"""
        self.pinecone_available = False
        self.pinecone_inference_available = False
        self.gemini_available = False
        self.index = None
        self.pc = None
        self.index_name = Config.PINECONE_INDEX_NAME
        self.gemini_model = None
        
        # Token usage tracking for cost monitoring
        self.tokens_used_this_session = 0
        self.estimated_monthly_tokens = 0
        
        # Batch processing settings
        self.batch_size = 25  # Optimal batch size for Pinecone inference
        
        self._init_pinecone()
        self._init_gemini_for_explanations()
    
    def _init_pinecone(self):
        """Initialize Pinecone client and check inference availability"""
        try:
            if not Config.PINECONE_API_KEY:
                logger.warning("Pinecone API key not found")
                return
            
            self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
            
            # Check if inference API is available (newer Pinecone versions)
            if hasattr(self.pc, 'inference'):
                try:
                    test_response = self.pc.inference.embed(
                        model="multilingual-e5-large",
                        inputs=["test"],
                        parameters={"input_type": "passage", "truncate": "END"}
                    )
                    self.pinecone_inference_available = True
                    logger.info("✅ Pinecone inference API available with multilingual-e5-large")
                except Exception as e:
                    logger.warning(f"Pinecone inference API test failed: {e}")
                    self.pinecone_inference_available = False
            else:
                logger.warning("❌ Pinecone inference API not available in this client version")
                logger.info("💡 To use Pinecone embeddings, upgrade: pip install --upgrade pinecone-client")
                logger.info("💡 For now, falling back to hash-based embeddings")
                self.pinecone_inference_available = False
            
            # Initialize or get existing index (works regardless of inference availability)
            try:
                self.index = self._get_or_create_index()
                self.pinecone_available = True
                logger.info("✅ Pinecone vector store initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone index: {e}")
                self.pinecone_available = False
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            self.pinecone_available = False
            self.pinecone_inference_available = False
    
    def _init_gemini_for_explanations(self):
        """Initialize Gemini only for AI explanations (not embeddings)"""
        try:
            if Config.GEMINI_API_KEY:
                genai.configure(api_key=Config.GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel(Config.GEMINI_MODEL)
                
                # Quick test
                response = self.gemini_model.generate_content("test")
                self.gemini_available = True
                logger.info("✅ Gemini initialized for AI explanations only")
            else:
                logger.info("Gemini not configured - will use static explanations")
                self.gemini_available = False
                
        except Exception as e:
            logger.warning(f"Gemini initialization failed: {e} - will use static explanations")
            self.gemini_available = False
    
    def _get_or_create_index(self):
        """Get existing index or create a new one with correct dimensions"""
        try:
            # List existing indexes
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1024,  # multilingual-e5-large uses 1024 dimensions
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                logger.info(f"✅ Created Pinecone index with 1024 dimensions for multilingual-e5-large")
                
                # Wait for index to be ready
                max_wait = 60  # Maximum wait time in seconds
                waited = 0
                while waited < max_wait:
                    try:
                        index_description = self.pc.describe_index(self.index_name)
                        if index_description.status['ready']:
                            break
                    except:
                        pass
                    time.sleep(2)
                    waited += 2
                
                if waited >= max_wait:
                    raise Exception(f"Index creation timed out after {max_wait} seconds")
                    
            else:
                logger.info(f"Using existing Pinecone index: {self.index_name}")
            
            # Get the index
            index = self.pc.Index(self.index_name)
            
            # Test connection with a simple operation
            try:
                # Try to get stats - this will fail gracefully if index isn't ready
                stats = index.describe_index_stats()
                vector_count = getattr(stats, 'total_vector_count', 0)
                logger.info(f"📊 Index ready with {vector_count} Netflix titles")
            except Exception as e:
                logger.warning(f"Could not get initial index stats: {e}")
                # Return the index anyway - it might work for operations
            
            return index
            
        except Exception as e:
            logger.error(f"Error with Pinecone index: {e}")
            raise
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding using best available method"""
        # Try Pinecone inference first (if available)
        if self.pinecone_inference_available:
            try:
                return self._get_pinecone_embedding(text)
            except Exception as e:
                logger.warning(f"Pinecone embedding failed, falling back to hash: {e}")
                return self._get_hash_based_embedding(text)
        else:
            # Fall back to hash-based embeddings
            logger.debug("Using hash-based embedding (Pinecone inference not available)")
            return self._get_hash_based_embedding(text)
    
    def _get_pinecone_embedding(self, text: str) -> List[float]:
        """Get embedding using Pinecone's multilingual-e5-large model"""
        if not self.pinecone_inference_available:
            raise Exception("Pinecone inference not available")
        
        try:
            # Use Pinecone's cheapest embedding model
            response = self.pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[text],
                parameters={
                    "input_type": "passage",  # Use "passage" for content, "query" for search
                    "truncate": "END"
                }
            )
            
            embedding = response[0]['values']
            
            # Track token usage for cost monitoring
            tokens_used = response.usage.get('total_tokens', len(text) // 4)
            self.tokens_used_this_session += tokens_used
            
            logger.debug(f"Generated Pinecone embedding: {len(embedding)}D, {tokens_used} tokens")
            return embedding
            
        except Exception as e:
            logger.error(f"Pinecone embedding failed: {e}")
            raise
    
    def _get_hash_based_embedding(self, text: str) -> List[float]:
        """Generate hash-based embedding as fallback (1024 dimensions to match Pinecone)"""
        import hashlib
        
        embeddings = []
        
        # Create 1024-dimensional embedding to match multilingual-e5-large
        for i in range(1024):
            # Create unique seed for each dimension
            seed_text = f"{text}_{i}".encode('utf-8')
            hash_value = hashlib.sha256(seed_text).hexdigest()
            
            # Convert first 8 hex chars to float
            hex_chunk = hash_value[:8]
            float_val = (int(hex_chunk, 16) % 65536 - 32768) / 32768.0  # Normalize to -1 to 1
            embeddings.append(float_val)
        
        logger.debug(f"Generated hash-based embedding: {len(embeddings)}D")
        return embeddings
    
    def _get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts efficiently"""
        if self.pinecone_inference_available:
            try:
                # Batch request to Pinecone
                response = self.pc.inference.embed(
                    model="multilingual-e5-large",
                    inputs=texts,
                    parameters={
                        "input_type": "passage",
                        "truncate": "END"
                    }
                )
                
                # Extract embeddings
                embeddings = [item['values'] for item in response]
                
                # Track token usage
                tokens_used = response.usage.get('total_tokens', sum(len(text) // 4 for text in texts))
                self.tokens_used_this_session += tokens_used
                self.estimated_monthly_tokens += tokens_used
                
                logger.info(f"Generated {len(embeddings)} Pinecone embeddings using {tokens_used} tokens")
                return embeddings
                
            except Exception as e:
                logger.warning(f"Batch Pinecone embedding failed, using hash fallback: {e}")
                # Fall back to individual hash embeddings
                return [self._get_hash_based_embedding(text) for text in texts]
        else:
            # Use hash-based embeddings for all texts
            logger.info(f"Generating {len(texts)} hash-based embeddings (Pinecone inference not available)")
            return [self._get_hash_based_embedding(text) for text in texts]
    
    def _create_movie_text(self, movie: Dict[str, Any]) -> str:
        """Create text representation for embedding"""
        title = movie.get('title') or movie.get('name', '')
        overview = movie.get('overview', '')[:400]  # Limit for token efficiency
        release_date = movie.get('release_date') or movie.get('first_air_date', '')
        media_type = movie.get('media_type', 'movie')
        rating = movie.get('vote_average', 0)
        
        # Create concise, embedding-friendly text
        text_parts = [
            f"Netflix {media_type}: {title}",
            f"Released: {release_date[:4] if release_date else 'Unknown'}",
            f"Rating: {rating}/10",
            f"Description: {overview}"
        ]
        
        return " | ".join(filter(None, text_parts))
    
    def upsert_movies(self, movies: List[Dict[str, Any]]) -> None:
        """Upsert Netflix movies using best available embedding method"""
        if not self.pinecone_available or not self.index:
            logger.error("Pinecone vector store not available")
            return
        
        if not movies:
            logger.warning("No movies to upsert")
            return
        
        embedding_method = "Pinecone multilingual-e5-large" if self.pinecone_inference_available else "hash-based (fallback)"
        logger.info(f"🚀 Starting upsert for {len(movies)} Netflix titles using {embedding_method} embeddings...")
        
        successful_vectors = 0
        failed_vectors = 0
        
        # Process in batches for efficiency
        for i in range(0, len(movies), self.batch_size):
            batch = movies[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(movies) - 1) // self.batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} titles)")
            
            try:
                # Prepare texts for batch embedding
                texts = []
                valid_movies = []
                
                for movie in batch:
                    try:
                        text = self._create_movie_text(movie)
                        texts.append(text)
                        valid_movies.append(movie)
                    except Exception as e:
                        logger.warning(f"Skipping movie {movie.get('title', 'unknown')}: {e}")
                        failed_vectors += 1
                        continue
                
                if not texts:
                    logger.warning(f"No valid movies in batch {batch_num}")
                    continue
                
                # Get embeddings for the entire batch
                embeddings = self._get_batch_embeddings(texts)
                
                # Prepare vectors for Pinecone
                vectors = []
                for movie, embedding in zip(valid_movies, embeddings):
                    try:
                        movie_id = f"{movie.get('media_type', 'movie')}_{movie.get('id', 0)}"
                        
                        # Verify embedding dimension (should be 1024 for both Pinecone and hash)
                        if len(embedding) != 1024:
                            logger.error(f"Wrong embedding dimension: {len(embedding)} != 1024")
                            failed_vectors += 1
                            continue
                        
                        # Create metadata
                        metadata = {
                            'id': int(movie.get('id', 0)),
                            'title': str(movie.get('title') or movie.get('name', '')),
                            'overview': str(movie.get('overview', ''))[:500],
                            'genre_ids': [str(int(float(gid))) for gid in movie.get('genre_ids', [])],
                            'release_date': str(movie.get('release_date') or movie.get('first_air_date', '')),
                            'vote_average': float(movie.get('vote_average', 0)),
                            'vote_count': int(movie.get('vote_count', 0)),
                            'poster_path': str(movie.get('poster_path', '')),
                            'backdrop_path': str(movie.get('backdrop_path', '')),
                            'media_type': str(movie.get('media_type', 'movie'))
                        }
                        
                        vectors.append({
                            'id': movie_id,
                            'values': embedding,
                            'metadata': metadata
                        })
                        
                    except Exception as e:
                        logger.error(f"Error preparing vector for {movie.get('title', 'unknown')}: {e}")
                        failed_vectors += 1
                        continue
                
                # Upsert batch to Pinecone
                if vectors:
                    self.index.upsert(vectors=vectors)
                    successful_vectors += len(vectors)
                    logger.info(f"✅ Batch {batch_num}/{total_batches} completed - {len(vectors)} vectors upserted")
                else:
                    logger.warning(f"No valid vectors in batch {batch_num}")
                
                # Small delay between batches
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ Batch {batch_num} failed: {e}")
                failed_vectors += len(batch)
                continue
        
        # Final summary
        logger.info(f"🎉 Upsert completed using {embedding_method}:")
        logger.info(f"   ✅ Successful: {successful_vectors} Netflix titles")
        logger.info(f"   ❌ Failed: {failed_vectors} titles")
        logger.info(f"   📊 Success rate: {successful_vectors/(successful_vectors + failed_vectors)*100:.1f}%")
        
        if self.pinecone_inference_available:
            logger.info(f"   💰 Tokens used this session: {self.tokens_used_this_session:,}")
            # Cost information
            monthly_limit = 5_000_000  # 5M free tokens
            remaining = monthly_limit - self.estimated_monthly_tokens
            if remaining > 0:
                logger.info(f"   🆓 Estimated remaining free tokens: {remaining:,}")
            else:
                cost = (self.estimated_monthly_tokens - monthly_limit) * 0.08 / 1_000_000
                logger.info(f"   💵 Estimated cost this month: ${cost:.4f}")
        else:
            logger.info(f"   🆓 No tokens used (hash-based embeddings)")
            logger.info(f"   💡 Upgrade Pinecone client for better embeddings: pip install --upgrade pinecone-client")
    
    def search_similar(self, query: str, genre_filter: Optional[str] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar Netflix content using best available embedding method"""
        if not self.pinecone_available or not self.index:
            logger.warning("Pinecone vector store not available")
            return []
        
        try:
            # Get query embedding using best available method
            query_text = f"Netflix content: {query}"
            
            if self.pinecone_inference_available:
                # Use Pinecone inference for query
                try:
                    query_response = self.pc.inference.embed(
                        model="multilingual-e5-large",
                        inputs=[query_text],
                        parameters={
                            "input_type": "query",  # Use "query" for search queries
                            "truncate": "END"
                        }
                    )
                    
                    query_embedding = query_response[0]['values']
                    
                    # Track query tokens
                    query_tokens = query_response.usage.get('total_tokens', len(query_text) // 4)
                    self.tokens_used_this_session += query_tokens
                    
                    logger.debug(f"Generated Pinecone query embedding using {query_tokens} tokens")
                    
                except Exception as e:
                    logger.warning(f"Pinecone query embedding failed, using hash fallback: {e}")
                    query_embedding = self._get_hash_based_embedding(query_text)
            else:
                # Use hash-based embedding for query
                query_embedding = self._get_hash_based_embedding(query_text)
                logger.debug("Generated hash-based query embedding")
            
            # Search in Pinecone
            search_k = min(top_k * 3, 100)  # Get more results for filtering
            
            results = self.index.query(
                vector=query_embedding,
                top_k=search_k,
                include_metadata=True
            )
            
            # Process and filter results
            movies = []
            for match in results.matches:
                metadata = match.metadata
                
                # Apply genre filter if specified
                if genre_filter and genre_filter in Config.GENRES:
                    genre_ids = metadata.get('genre_ids', [])
                    target_genre_ids = [str(gid) for gid in Config.GENRES[genre_filter]['tmdb_ids']]
                    if not any(genre_id in target_genre_ids for genre_id in genre_ids):
                        continue
                
                movies.append({
                    'id': metadata.get('id'),
                    'title': metadata.get('title'),
                    'overview': metadata.get('overview'),
                    'genre_ids': metadata.get('genre_ids', []),
                    'release_date': metadata.get('release_date'),
                    'vote_average': metadata.get('vote_average'),
                    'vote_count': metadata.get('vote_count'),
                    'poster_path': metadata.get('poster_path'),
                    'backdrop_path': metadata.get('backdrop_path'),
                    'media_type': metadata.get('media_type'),
                    'score': match.score
                })
                
                if len(movies) >= top_k:
                    break
            
            embedding_method = "Pinecone" if self.pinecone_inference_available else "hash-based"
            logger.info(f"🔍 Found {len(movies)} Netflix titles for '{query}' using {embedding_method} embeddings (genre: {genre_filter})")
            return movies
            
        except Exception as e:
            logger.error(f"Error searching Netflix content: {e}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get comprehensive index statistics"""
        if not self.pinecone_available or not self.index:
            return {
                "error": "Pinecone not available", 
                "total_vector_count": 0,
                "embedding_model": "multilingual-e5-large",
                "embedding_provider": "pinecone",
                "dimension": 1024,
                "tokens_used_this_session": self.tokens_used_this_session,
                "estimated_monthly_tokens": self.estimated_monthly_tokens,
                "cost_per_million_tokens": 0.08,
                "free_tokens_monthly": 5_000_000
            }
        
        try:
            # Call the describe_index_stats method properly
            stats = self.index.describe_index_stats()
            
            # Convert stats to dict safely
            if hasattr(stats, '__dict__'):
                result = vars(stats)
            elif hasattr(stats, 'to_dict'):
                result = stats.to_dict()
            else:
                # Fallback: create basic structure
                result = {
                    "total_vector_count": getattr(stats, 'total_vector_count', 0),
                    "dimension": getattr(stats, 'dimension', 1024),
                    "index_fullness": getattr(stats, 'index_fullness', 0.0),
                    "namespaces": getattr(stats, 'namespaces', {})
                }
            
            # Add our usage tracking
            result.update({
                "embedding_model": "multilingual-e5-large",
                "embedding_provider": "pinecone",
                "dimension": 1024,
                "tokens_used_this_session": self.tokens_used_this_session,
                "estimated_monthly_tokens": self.estimated_monthly_tokens,
                "cost_per_million_tokens": 0.08,
                "free_tokens_monthly": 5_000_000
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {
                "error": str(e), 
                "total_vector_count": 0,
                "embedding_model": "multilingual-e5-large",
                "embedding_provider": "pinecone",
                "dimension": 1024,
                "tokens_used_this_session": self.tokens_used_this_session,
                "estimated_monthly_tokens": self.estimated_monthly_tokens,
                "cost_per_million_tokens": 0.08,
                "free_tokens_monthly": 5_000_000
            }
    
    def generate_explanation(self, movie: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """Generate AI explanation using Gemini or static fallback"""
        if self.gemini_available:
            try:
                prompt = f"""You are a Netflix content expert. Explain why someone would enjoy this content.

Content: {movie.get('title', '')}
Description: {movie.get('overview', '')}
User preferences: {user_input}

Provide a 2-sentence explanation of why they'd enjoy this, then list 3 memorable quotes and 3 memorable moments.

Format:
EXPLANATION: [explanation]
QUOTES: [quote1] | [quote2] | [quote3]  
MOMENTS: [moment1] | [moment2] | [moment3]"""

                response = self.gemini_model.generate_content(prompt)
                return self._parse_gemini_response(response.text, movie, user_input)
                
            except Exception as e:
                logger.warning(f"Gemini explanation failed: {e}")
                return self._static_explanation(movie, user_input)
        else:
            return self._static_explanation(movie, user_input)
    
    def _parse_gemini_response(self, content: str, movie: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """Parse Gemini response into structured data"""
        explanation = ""
        quotes = []
        moments = []
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('EXPLANATION:'):
                explanation = line.replace('EXPLANATION:', '').strip()
            elif line.startswith('QUOTES:'):
                quotes_text = line.replace('QUOTES:', '').strip()
                quotes = [q.strip() for q in quotes_text.split('|') if q.strip()]
            elif line.startswith('MOMENTS:'):
                moments_text = line.replace('MOMENTS:', '').strip()
                moments = [m.strip() for m in moments_text.split('|') if m.strip()]
        
        # Ensure we have content
        if not explanation:
            explanation = f"Based on your interest in {user_input}, you'll enjoy {movie.get('title', 'this Netflix content')} for its engaging storytelling and quality production."
        
        if len(quotes) < 3:
            quotes = [
                f"A memorable line from {movie.get('title', 'this show')}",
                f"An impactful quote that defines the story",
                f"A notable dialogue from the production"
            ]
        
        if len(moments) < 3:
            moments = [
                "An captivating opening sequence",
                "A pivotal character development moment",
                "A memorable climactic scene"
            ]
        
        return {
            'why': explanation,
            'quotes': quotes[:3],
            'moments': moments[:3]
        }
    
    def _static_explanation(self, movie: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """Generate static explanation when Gemini is not available"""
        title = movie.get('title', 'this Netflix content')
        media_type = movie.get('media_type', 'content')
        rating = movie.get('vote_average', 0)
        
        return {
            'why': f"This Netflix {media_type} aligns perfectly with your interest in {user_input}. With a {rating}/10 rating, {title} offers high-quality entertainment that matches your preferences.",
            'quotes': [
                f"A standout quote from {title}",
                f"An memorable line that captures the essence",
                f"A powerful moment from the story"
            ],
            'moments': [
                "An engaging opening that draws you in",
                "A compelling character development arc",
                "A satisfying and memorable conclusion"
            ]
        }