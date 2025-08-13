from __future__ import annotations
import requests
import logging
import time
from typing import List, Dict, Any, Optional
from config import Config

logger = logging.getLogger(__name__)

class TMDBClient:
    def __init__(self):
        self.api_key = Config.TMDB_API_KEY
        self.base_url = Config.TMDB_BASE_URL
        self.session = requests.Session()
        
        # Netflix provider ID (this is consistent across TMDB)
        self.NETFLIX_PROVIDER_ID = 8
        
        # Rate limiting - TMDB allows 40 requests per 10 seconds
        self.request_count = 0
        self.reset_time = time.time() + 10
        self.max_requests_per_10_sec = 35  # Leave some buffer
        
        if not self.api_key:
            raise ValueError("TMDB_API_KEY is required")
        
        logger.info("✅ TMDB client initialized for Netflix content")
    
    def _rate_limit_check(self):
        """Check and enforce rate limits"""
        current_time = time.time()
        
        # Reset counter if 10 seconds have passed
        if current_time > self.reset_time:
            self.request_count = 0
            self.reset_time = current_time + 10
        
        # If we're at the limit, wait
        if self.request_count >= self.max_requests_per_10_sec:
            wait_time = self.reset_time - current_time
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                self.request_count = 0
                self.reset_time = time.time() + 10
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make rate-limited request to TMDB API"""
        if params is None:
            params = {}
        
        # Add API key
        params['api_key'] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        
        # Check rate limits
        self._rate_limit_check()
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            # Increment counter
            self.request_count += 1
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"TMDB API request failed for {endpoint}: {e}")
            raise
    
    def get_netflix_movies(self, page: int = 1, region: str = "US") -> List[Dict[str, Any]]:
        """Get movies available on Netflix using discover endpoint"""
        try:
            # Use discover endpoint with Netflix provider filter
            params = {
                "page": page,
                "with_watch_providers": self.NETFLIX_PROVIDER_ID,
                "watch_region": region,
                "sort_by": "popularity.desc",
                "include_adult": False,
                "include_video": False,
                "vote_count.gte": 50  # Ensure quality content
            }
            
            data = self._make_request("discover/movie", params)
            movies = data.get("results", [])
            
            # Add media_type for consistency
            for movie in movies:
                movie['media_type'] = 'movie'
            
            logger.info(f"Fetched {len(movies)} Netflix movies from page {page}")
            return movies
            
        except Exception as e:
            logger.error(f"Error fetching Netflix movies page {page}: {e}")
            return []
    
    def get_netflix_tv_shows(self, page: int = 1, region: str = "US") -> List[Dict[str, Any]]:
        """Get TV shows available on Netflix using discover endpoint"""
        try:
            # Use discover endpoint with Netflix provider filter
            params = {
                "page": page,
                "with_watch_providers": self.NETFLIX_PROVIDER_ID,
                "watch_region": region,
                "sort_by": "popularity.desc",
                "include_adult": False,
                "vote_count.gte": 50  # Ensure quality content
            }
            
            data = self._make_request("discover/tv", params)
            tv_shows = data.get("results", [])
            
            # Add media_type and normalize title field
            for show in tv_shows:
                show['media_type'] = 'tv'
                # TV shows use 'name' instead of 'title'
                if 'name' in show and 'title' not in show:
                    show['title'] = show['name']
            
            logger.info(f"Fetched {len(tv_shows)} Netflix TV shows from page {page}")
            return tv_shows
            
        except Exception as e:
            logger.error(f"Error fetching Netflix TV shows page {page}: {e}")
            return []
    
    def get_popular_movies(self, page: int = 1) -> List[Dict[str, Any]]:
        """Get popular movies - redirected to Netflix movies"""
        logger.info(f"Redirecting popular movies request to Netflix movies (page {page})")
        return self.get_netflix_movies(page)
    
    def get_popular_tv_shows(self, page: int = 1) -> List[Dict[str, Any]]:
        """Get popular TV shows - redirected to Netflix TV shows"""
        logger.info(f"Redirecting popular TV shows request to Netflix TV shows (page {page})")
        return self.get_netflix_tv_shows(page)
    
    def get_netflix_content_by_genre(self, genre_ids: List[int], page: int = 1, region: str = "US") -> List[Dict[str, Any]]:
        """Get Netflix content filtered by specific genres"""
        try:
            all_content = []
            
            # Get movies with genre filter
            movie_params = {
                "page": page,
                "with_watch_providers": self.NETFLIX_PROVIDER_ID,
                "watch_region": region,
                "with_genres": ",".join(map(str, genre_ids)),
                "sort_by": "vote_average.desc",
                "vote_count.gte": 100,  # Ensure quality content
                "include_adult": False
            }
            
            try:
                movie_data = self._make_request("discover/movie", movie_params)
                movies = movie_data.get("results", [])
                for movie in movies:
                    movie['media_type'] = 'movie'
                all_content.extend(movies)
                logger.info(f"Found {len(movies)} Netflix movies for genres {genre_ids}")
            except Exception as e:
                logger.error(f"Error fetching Netflix movies by genre: {e}")
            
            # Get TV shows with genre filter
            tv_params = {
                "page": page,
                "with_watch_providers": self.NETFLIX_PROVIDER_ID,
                "watch_region": region,
                "with_genres": ",".join(map(str, genre_ids)),
                "sort_by": "vote_average.desc",
                "vote_count.gte": 100,
                "include_adult": False
            }
            
            try:
                tv_data = self._make_request("discover/tv", tv_params)
                tv_shows = tv_data.get("results", [])
                for show in tv_shows:
                    show['media_type'] = 'tv'
                    if 'name' in show and 'title' not in show:
                        show['title'] = show['name']
                all_content.extend(tv_shows)
                logger.info(f"Found {len(tv_shows)} Netflix TV shows for genres {genre_ids}")
            except Exception as e:
                logger.error(f"Error fetching Netflix TV shows by genre: {e}")
            
            return all_content
            
        except Exception as e:
            logger.error(f"Error fetching Netflix content by genre: {e}")
            return []
    
    def get_trending_netflix_content(self, time_window: str = "week", region: str = "US") -> List[Dict[str, Any]]:
        """Get trending content and filter for Netflix availability"""
        try:
            all_content = []
            
            # Get trending movies
            try:
                trending_movies = self._make_request(f"trending/movie/{time_window}")
                for movie in trending_movies.get("results", []):
                    if self._is_on_netflix(movie['id'], 'movie', region):
                        movie['media_type'] = 'movie'
                        all_content.append(movie)
            except Exception as e:
                logger.error(f"Error fetching trending movies: {e}")
            
            # Get trending TV shows
            try:
                trending_tv = self._make_request(f"trending/tv/{time_window}")
                for show in trending_tv.get("results", []):
                    if self._is_on_netflix(show['id'], 'tv', region):
                        show['media_type'] = 'tv'
                        if 'name' in show and 'title' not in show:
                            show['title'] = show['name']
                        all_content.append(show)
            except Exception as e:
                logger.error(f"Error fetching trending TV shows: {e}")
            
            logger.info(f"Found {len(all_content)} trending Netflix items")
            return all_content
            
        except Exception as e:
            logger.error(f"Error fetching trending Netflix content: {e}")
            return []
    
    def _is_on_netflix(self, content_id: int, media_type: str, region: str = "US") -> bool:
        """Check if specific content is available on Netflix"""
        try:
            # Get watch providers for the content
            data = self._make_request(f"{media_type}/{content_id}/watch/providers")
            
            # Check if Netflix is in the providers for the specified region
            results = data.get("results", {})
            region_data = results.get(region, {})
            
            # Check both flatrate (subscription) and buy/rent options
            flatrate = region_data.get("flatrate", [])
            buy = region_data.get("buy", [])
            rent = region_data.get("rent", [])
            
            all_providers = flatrate + buy + rent
            
            # Check if Netflix (provider_id = 8) is in any of these
            for provider in all_providers:
                if provider.get("provider_id") == self.NETFLIX_PROVIDER_ID:
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Could not check Netflix availability for {media_type} {content_id}: {e}")
            return False
    
    def search_movies(self, query: str, page: int = 1) -> List[Dict[str, Any]]:
        """Search movies and filter for Netflix availability"""
        try:
            data = self._make_request("search/movie", {"query": query, "page": page})
            movies = data.get("results", [])
            
            # Filter for Netflix availability
            netflix_movies = []
            for movie in movies:
                if self._is_on_netflix(movie['id'], 'movie'):
                    movie['media_type'] = 'movie'
                    netflix_movies.append(movie)
            
            logger.info(f"Found {len(netflix_movies)} Netflix movies for search: {query}")
            return netflix_movies
            
        except Exception as e:
            logger.error(f"Error searching Netflix movies: {e}")
            return []
    
    def search_tv_shows(self, query: str, page: int = 1) -> List[Dict[str, Any]]:
        """Search TV shows and filter for Netflix availability"""
        try:
            data = self._make_request("search/tv", {"query": query, "page": page})
            tv_shows = data.get("results", [])
            
            # Filter for Netflix availability
            netflix_shows = []
            for show in tv_shows:
                if self._is_on_netflix(show['id'], 'tv'):
                    show['media_type'] = 'tv'
                    if 'name' in show and 'title' not in show:
                        show['title'] = show['name']
                    netflix_shows.append(show)
            
            logger.info(f"Found {len(netflix_shows)} Netflix TV shows for search: {query}")
            return netflix_shows
            
        except Exception as e:
            logger.error(f"Error searching Netflix TV shows: {e}")
            return []
    
    def get_genre_mapping(self) -> Dict[int, str]:
        """Get genre ID to name mapping"""
        try:
            # Get movie genres
            movie_genres = self._make_request("genre/movie/list")
            tv_genres = self._make_request("genre/tv/list")
            
            genre_mapping = {}
            
            # Combine movie and TV genres
            for genre in movie_genres.get("genres", []):
                genre_mapping[genre["id"]] = genre["name"]
            
            for genre in tv_genres.get("genres", []):
                genre_mapping[genre["id"]] = genre["name"]
            
            logger.info(f"Retrieved {len(genre_mapping)} genres from TMDB")
            return genre_mapping
            
        except Exception as e:
            logger.error(f"Error getting genre mapping: {e}")
            # Return default mapping if API fails
            return {
                28: "Action",
                12: "Adventure", 
                35: "Comedy",
                18: "Drama",
                53: "Thriller",
                27: "Horror",
                14: "Fantasy",
                878: "Science Fiction",
                16: "Animation",
                10402: "Music",
                10759: "Action & Adventure",
                10762: "Kids",
                10763: "News",
                10764: "Reality",
                10765: "Sci-Fi & Fantasy",
                10766: "Soap",
                10767: "Talk",
                10768: "War & Politics"
            }