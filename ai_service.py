from __future__ import annotations
import logging
from typing import Dict, Any, Optional
from config import Config

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        self.provider = Config.AI_PROVIDER.lower()
        self.openai_client = None
        self.gemini_model = None
        
        # Initialize Gemini first (primary)
        if self.provider == "gemini" or self.provider == "auto":
            if self._init_gemini():
                self.provider = "gemini"
            elif self.provider == "auto" and self._init_openai():
                self.provider = "openai"
                logger.info("Gemini failed, falling back to OpenAI")
            else:
                self.provider = "none"
                logger.warning("Both Gemini and OpenAI failed, using static responses")
        
        # Initialize OpenAI if specifically requested
        elif self.provider == "openai":
            if not self._init_openai():
                self.provider = "none"
        
        # None provider
        elif self.provider == "none":
            logger.info("AI provider set to 'none' - using static responses only")
        
        else:
            logger.warning(f"Unknown AI provider '{self.provider}', falling back to static responses")
            self.provider = "none"
    
    def _init_gemini(self) -> bool:
        """Initialize Gemini client"""
        try:
            if not Config.GEMINI_API_KEY:
                logger.warning("Gemini API key not found")
                return False
            
            import google.generativeai as genai
            genai.configure(api_key=Config.GEMINI_API_KEY)
            
            # Initialize model
            self.gemini_model = genai.GenerativeModel(Config.GEMINI_MODEL)
            
            # Test the connection
            try:
                response = self.gemini_model.generate_content("test")
                logger.info(f"✅ Gemini initialized successfully with model: {Config.GEMINI_MODEL}")
                return True
            except Exception as e:
                logger.warning(f"Gemini test failed: {e}")
                return False
                
        except ImportError:
            logger.error("google-generativeai package not installed. Run: pip install google-generativeai")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            return False
    
    def _init_openai(self) -> bool:
        """Initialize OpenAI client"""
        try:
            if not Config.OPENAI_API_KEY:
                logger.warning("OpenAI API key not found")
                return False
            
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
            
            # Test the connection
            if Config.OPENAI_CHAT_MODEL and Config.OPENAI_CHAT_MODEL.lower() != "none":
                try:
                    self.openai_client.chat.completions.create(
                        model=Config.OPENAI_CHAT_MODEL,
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=1
                    )
                    logger.info(f"✅ OpenAI initialized successfully with model: {Config.OPENAI_CHAT_MODEL}")
                    return True
                except Exception as e:
                    logger.warning(f"OpenAI test failed: {e}")
                    return False
            else:
                logger.warning("OpenAI chat model not configured")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            return False
    
    def generate_recommendation_explanation(self, movie: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """Generate AI explanation using the configured provider"""
        
        if self.provider == "gemini":
            return self._generate_gemini_explanation(movie, user_input)
        elif self.provider == "openai":
            return self._generate_openai_explanation(movie, user_input)
        else:
            return self._generate_static_explanation(movie, user_input)
    
    def _generate_gemini_explanation(self, movie: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """Generate explanation using Gemini (primary method)"""
        try:
            prompt = f"""
            You are a movie recommendation expert. Based on the user's preferences and the movie information below, provide a personalized recommendation.
            
            Movie/Show: {movie.get('title', '')}
            Overview: {movie.get('overview', '')}
            Release Date: {movie.get('release_date', '')}
            Rating: {movie.get('vote_average', 0)}/10 ({movie.get('vote_count', 0)} votes)
            Type: {movie.get('media_type', '')}
            
            User's preferences: {user_input}
            
            Please provide:
            1. A compelling 2-3 sentence explanation of why this user would enjoy this content
            2. 3 memorable quotes (realistic but can be created if not known exactly)
            3. 3 memorable moments or scenes (realistic descriptions)
            
            Format your response exactly as:
            EXPLANATION: [your explanation here]
            QUOTES: [quote1] | [quote2] | [quote3]
            MOMENTS: [moment1] | [moment2] | [moment3]
            """
            
            response = self.gemini_model.generate_content(prompt)
            result = self._parse_ai_response(response.text, movie, user_input)
            logger.debug(f"Generated Gemini explanation for {movie.get('title')}")
            return result
            
        except Exception as e:
            logger.error(f"Gemini explanation failed: {e}")
            return self._generate_static_explanation(movie, user_input)
    
    def _generate_openai_explanation(self, movie: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """Generate explanation using OpenAI (fallback method)"""
        try:
            context = f"""
            Movie/Show Information:
            Title: {movie.get('title', '')}
            Overview: {movie.get('overview', '')}
            Genre IDs: {movie.get('genre_ids', [])}
            Release Date: {movie.get('release_date', '')}
            Vote Average: {movie.get('vote_average', 0)}
            Media Type: {movie.get('media_type', '')}
            
            User Input: {user_input}
            
            Please provide:
            1. A compelling explanation of why this user would enjoy this movie/show (2-3 sentences)
            2. 3 memorable quotes from the movie/show (if known, otherwise create plausible ones)
            3. 3 memorable moments or scenes from the movie/show (if known, otherwise create plausible ones)
            
            Format your response as:
            EXPLANATION: [explanation]
            QUOTES: [quote1] | [quote2] | [quote3]
            MOMENTS: [moment1] | [moment2] | [moment3]
            """
            
            response = self.openai_client.chat.completions.create(
                model=Config.OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a movie recommendation expert. Provide engaging, personalized explanations."},
                    {"role": "user", "content": context}
                ],
                max_tokens=400,
                temperature=0.7
            )
            
            result = self._parse_ai_response(response.choices[0].message.content, movie, user_input)
            logger.debug(f"Generated OpenAI explanation for {movie.get('title')}")
            return result
            
        except Exception as e:
            logger.error(f"OpenAI explanation failed: {e}")
            return self._generate_static_explanation(movie, user_input)
    
    def _parse_ai_response(self, content: str, movie: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """Parse AI response into structured data"""
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
        
        # Ensure we have fallbacks
        if not explanation:
            explanation = f"Based on your preferences for {user_input}, you'll enjoy {movie.get('title', 'this content')} because it aligns perfectly with your interests and offers engaging storytelling."
        
        if not quotes or len(quotes) < 3:
            quotes = [
                f"A memorable quote from {movie.get('title', 'this production')}",
                f"Another impactful line from this {movie.get('media_type', 'content')}",
                "A third notable quote from the production"
            ]
        
        if not moments or len(moments) < 3:
            moments = [
                "An exciting opening sequence",
                "A memorable character moment",
                "A climactic scene"
            ]
        
        return {
            'why': explanation,
            'quotes': quotes[:3],
            'moments': moments[:3]
        }
    
    def _generate_static_explanation(self, movie: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """Generate static explanation when AI is not available"""
        return {
            'why': f"This {movie.get('media_type', 'content')} matches your interest in {user_input}. With a {movie.get('vote_average', 0)}/10 rating from {movie.get('vote_count', 0)} viewers, it's a well-regarded choice. The plot summary suggests it has the elements you're looking for.",
            'quotes': [
                f"A memorable quote from {movie.get('title', 'this production')}",
                f"Another impactful line from this {movie.get('media_type', 'content')}",
                "A third notable quote from the production"
            ],
            'moments': [
                "An engaging opening sequence",
                "A memorable character development moment",
                "A satisfying climactic scene"
            ]
        }
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current AI provider"""
        return {
            "provider": self.provider,
            "available": self.provider != "none",
            "model": {
                "gemini": Config.GEMINI_MODEL if self.provider == "gemini" else None,
                "openai": Config.OPENAI_CHAT_MODEL if self.provider == "openai" else None
            }
        }