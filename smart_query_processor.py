from __future__ import annotations
import logging
import re
from typing import Dict, Any, List, Tuple, Optional
from config import Config

logger = logging.getLogger(__name__)

class SmartQueryProcessor:
    def __init__(self, gemini_model=None):
        self.gemini_model = gemini_model
        
        # Keywords that indicate Netflix content requests
        self.netflix_keywords = {
            'content_types': ['movie', 'film', 'show', 'series', 'tv', 'netflix', 'watch', 'stream'],
            'genres': list(Config.GENRES.keys()),
            'moods': ['funny', 'scary', 'romantic', 'action', 'drama', 'comedy', 'thriller'],
            'activities': ['watch', 'binge', 'stream', 'viewing', 'entertainment']
        }
        
        # Off-topic keywords that clearly aren't about Netflix content
        self.off_topic_keywords = {
            'food': ['cook', 'bake', 'recipe', 'eat', 'food', 'restaurant', 'meal'],
            'games': ['game', 'gaming', 'play', 'xbox', 'playstation', 'nintendo'],
            'shopping': ['buy', 'purchase', 'shop', 'store', 'amazon'],
            'work': ['job', 'work', 'career', 'interview', 'salary'],
            'travel': ['travel', 'vacation', 'trip', 'flight', 'hotel'],
            'health': ['doctor', 'medicine', 'workout', 'fitness', 'diet'],
            'tech': ['phone', 'computer', 'software', 'app', 'device']
        }
        
        # Query enhancement patterns
        self.enhancement_patterns = {
            'vague_indicators': [
                'i don\'t know', 'not sure', 'anything', 'whatever', 'surprise me',
                'something', 'don\'t care', 'random', 'anything good'
            ],
            'mood_enhancers': {
                'happy': 'uplifting and feel-good',
                'sad': 'emotional and moving', 
                'tired': 'light and easy to watch',
                'excited': 'thrilling and engaging',
                'bored': 'entertaining and captivating',
                'stressed': 'relaxing and comforting'
            }
        }
    
    def process_query(self, user_input: str, genre: str) -> Dict[str, Any]:
        """
        Process any user query and return enhanced query or guidance
        """
        user_input = user_input.lower().strip()
        
        # Step 1: Check if query is about Netflix content
        content_relevance = self._assess_content_relevance(user_input)
        
        if content_relevance['is_off_topic']:
            return self._generate_off_topic_response(user_input, content_relevance['category'])
        
        # Step 2: Enhance vague or unclear queries
        enhanced_query = self._enhance_query(user_input, genre)
        
        # Step 3: Generate contextual improvements
        suggestions = self._generate_query_suggestions(user_input, genre)
        
        return {
            'status': 'enhanced',
            'original_query': user_input,
            'enhanced_query': enhanced_query,
            'suggestions': suggestions,
            'confidence': content_relevance['confidence']
        }
    
    def _assess_content_relevance(self, user_input: str) -> Dict[str, Any]:
        """Assess if the query is about Netflix content"""
        
        # Check for off-topic keywords
        for category, keywords in self.off_topic_keywords.items():
            if any(keyword in user_input for keyword in keywords):
                # But also check if it might still be about movies/shows
                netflix_score = sum(1 for keyword in self.netflix_keywords['content_types'] if keyword in user_input)
                if netflix_score == 0:
                    return {
                        'is_off_topic': True,
                        'category': category,
                        'confidence': 0.9
                    }
        
        # Check for Netflix content indicators
        netflix_score = 0
        for category, keywords in self.netflix_keywords.items():
            netflix_score += sum(1 for keyword in keywords if keyword in user_input)
        
        # Calculate confidence
        confidence = min(netflix_score / 3, 1.0)  # Normalize to 0-1
        
        # If very low confidence and contains non-Netflix terms, might be off-topic
        if confidence < 0.3 and any(
            any(keyword in user_input for keyword in off_topic_keywords) 
            for off_topic_keywords in self.off_topic_keywords.values()
        ):
            return {
                'is_off_topic': True,
                'category': 'general',
                'confidence': 0.3
            }
        
        return {
            'is_off_topic': False,
            'category': 'netflix_content',
            'confidence': max(confidence, 0.4)  # Minimum confidence for Netflix content
        }
    
    def _enhance_query(self, user_input: str, genre: str) -> str:
        """Enhance vague or unclear queries"""
        
        # If query is already specific and good, don't change much
        if len(user_input) > 20 and any(word in user_input for word in ['with', 'about', 'like', 'featuring']):
            return f"Netflix {genre.lower()} content: {user_input}"
        
        enhanced = user_input
        
        # Handle completely vague queries
        if any(indicator in user_input for indicator in self.enhancement_patterns['vague_indicators']):
            enhanced = f"popular and highly-rated {genre.lower()} content"
        
        # Add genre context if missing
        if genre.lower() not in enhanced:
            enhanced = f"{genre.lower()} {enhanced}"
        
        # Add Netflix context
        if 'netflix' not in enhanced:
            enhanced = f"Netflix {enhanced}"
        
        # Enhance with mood if detected
        for mood, description in self.enhancement_patterns['mood_enhancers'].items():
            if mood in user_input:
                enhanced = f"{description} Netflix {genre.lower()} content"
                break
        
        # Make it more descriptive
        enhanced = enhanced.replace('movies', 'movies with engaging storylines')
        enhanced = enhanced.replace('shows', 'series with compelling characters')
        
        return enhanced
    
    def _generate_off_topic_response(self, user_input: str, category: str) -> Dict[str, Any]:
        """Generate helpful response for off-topic queries"""
        
        category_responses = {
            'food': "I can't help with cooking or baking, but I can recommend Netflix cooking shows!",
            'games': "I don't recommend video games, but how about Netflix series about gaming or esports?",
            'shopping': "I'm not a shopping assistant, but I can suggest Netflix documentaries about consumer culture!",
            'work': "I can't help with work advice, but I can recommend Netflix series about workplace drama or documentaries about careers!",
            'travel': "I don't plan trips, but I can recommend Netflix travel documentaries or shows set in exotic locations!",
            'health': "I can't give health advice, but I can suggest Netflix documentaries about wellness or medical dramas!",
            'tech': "I don't recommend tech products, but how about Netflix sci-fi shows or tech documentaries?",
            'general': "I can only recommend Netflix movies and TV shows."
        }
        
        response = category_responses.get(category, category_responses['general'])
        
        # Generate example queries
        example_queries = self._generate_example_queries()
        
        return {
            'status': 'off_topic',
            'message': response,
            'helpful_suggestion': f"I'm your Netflix content specialist! {response}",
            'example_queries': example_queries,
            'guidance': "Here are some example queries you can try instead:"
        }
    
    def _generate_query_suggestions(self, user_input: str, genre: str) -> List[str]:
        """Generate helpful query suggestions"""
        
        base_suggestions = {
            'Comedy': [
                "witty romantic comedies with great chemistry",
                "laugh-out-loud comedy series perfect for binge watching",
                "feel-good comedy movies for a relaxing evening"
            ],
            'Action': [
                "high-octane action movies with incredible stunts",
                "superhero action series with compelling storylines", 
                "martial arts action films with amazing choreography"
            ],
            'Drama': [
                "emotional character-driven dramas that make you think",
                "gripping psychological dramas with plot twists",
                "heartwarming family dramas with strong relationships"
            ],
            'Thriller': [
                "edge-of-your-seat psychological thrillers",
                "crime thrillers with unexpected plot twists",
                "suspenseful mystery series that keep you guessing"
            ],
            'Horror': [
                "genuinely scary horror movies that will give you chills",
                "supernatural horror series with creepy atmospheres",
                "psychological horror that messes with your mind"
            ]
        }
        
        return base_suggestions.get(genre, [
            f"popular {genre.lower()} content with great reviews",
            f"highly-rated {genre.lower()} series perfect for binge watching",
            f"acclaimed {genre.lower()} movies with compelling storylines"
        ])
    
    def _generate_example_queries(self) -> List[Dict[str, str]]:
        """Generate example queries for different genres"""
        return [
            {
                "genre": "Comedy",
                "query": "witty romantic comedies with great chemistry",
                "description": "For funny, heartwarming romance"
            },
            {
                "genre": "Action", 
                "query": "superhero movies with amazing special effects",
                "description": "For thrilling superhero adventures"
            },
            {
                "genre": "Drama",
                "query": "emotional family dramas that make you cry",
                "description": "For deep, moving storytelling"
            },
            {
                "genre": "Thriller",
                "query": "psychological thrillers with plot twists",
                "description": "For suspenseful, mind-bending content"
            },
            {
                "genre": "Documentary",
                "query": "fascinating true crime documentaries",
                "description": "For real-life mysteries and investigations"
            }
        ]
    
    def enhance_with_ai(self, user_input: str, genre: str) -> str:
        """Use Gemini AI to enhance queries when available"""
        if not self.gemini_model:
            return self._enhance_query(user_input, genre)
        
        try:
            prompt = f"""
            You are a Netflix content specialist. A user wants {genre} recommendations and said: "{user_input}"
            
            Transform this into a specific, descriptive query that will help find great Netflix {genre} content.
            
            Rules:
            1. Keep it under 50 words
            2. Be specific about mood, style, or preferences
            3. Focus on what makes {genre} content appealing
            4. Include concrete attributes that can be matched
            
            If the query is very vague like "I don't know what I want", create a query for popular, highly-rated {genre} content.
            
            Return ONLY the enhanced query, nothing else.
            """
            
            response = self.gemini_model.generate_content(prompt)
            enhanced = response.text.strip().strip('"')
            
            # Ensure it mentions Netflix
            if 'netflix' not in enhanced.lower():
                enhanced = f"Netflix {enhanced}"
                
            return enhanced
            
        except Exception as e:
            logger.warning(f"AI enhancement failed: {e}")
            return self._enhance_query(user_input, genre)