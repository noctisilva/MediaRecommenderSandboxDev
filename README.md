# 🎬 Netflix Media Recommender API

An AI-powered **Netflix-only** movie and TV show recommendation system using TMDB API, Pinecone vector database, and Gemini AI for personalized recommendations.

## ✨ Features

- **🎯 Netflix-Only Content**: All recommendations are filtered to show only Netflix-available titles
- **🤖 Gemini AI-Powered**: Uses Google's Gemini AI for embeddings and explanations (free tier available)
- **⚡ Optimized Performance**: Efficient batch processing, rate limiting, and async operations
- **🔍 Vector Search**: Pinecone vector database for semantic similarity search
- **🌍 Regional Support**: Supports different Netflix regions (default: US)
- **📊 Smart Filtering**: Quality filtering and genre-based discovery
- **🚀 Fast Population**: Parallel processing for quick initial setup

## 🎭 Supported Netflix Genres

- **Action** - Fast-paced Netflix action movies and series
- **Adventure** - Netflix adventure content and exploration shows
- **Comedy** - Netflix comedy movies, series, and specials
- **Drama** - Character-driven Netflix dramas
- **Thriller** - Netflix suspense and thriller content
- **Horror** - Netflix horror movies and series
- **Fantasy** - Netflix fantasy content with magical elements
- **Science Fiction** - Netflix sci-fi content
- **Animation** - Netflix animated movies and series
- **Reality** - Netflix reality TV shows and competitions
- **Talk Show** - Netflix talk shows and interviews
- **Documentary** - Netflix documentaries and factual content
- **Romance** - Netflix romantic movies and series
- **Crime** - Netflix crime dramas and true crime
- **Family** - Netflix family-friendly content

## 🚀 Quick Setup

### 1. Prerequisites

- Python 3.11+
- TMDB API key (free)
- Pinecone API key (free tier available)
- Gemini API key (recommended - generous free tier)

### 2. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd netflix-media-recommender

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (see below)
```

### 3. Get API Keys

#### TMDB API Key (Required - Free)
1. Go to [TMDB](https://www.themoviedb.org/)
2. Create an account
3. Go to Settings > API
4. Request an API key (free)

#### Pinecone API Key (Required - Free Tier Available)
1. Go to [Pinecone](https://www.pinecone.io/)
2. Create an account
3. Create a new project
4. Copy the API key

#### Gemini API Key (Recommended - Free)
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create or sign in to Google account
3. Generate API key (generous free tier)

#### OpenAI API Key (Optional - Paid)
1. Go to [OpenAI](https://platform.openai.com/)
2. Create account and add payment method
3. Generate API key

### 4. Configure Environment

Edit your `.env` file:

```env
# Required
TMDB_API_KEY=your_tmdb_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Optional (for fallback)
OPENAI_API_KEY=your_openai_api_key_here

# Configuration (defaults work well)
PINECONE_INDEX_NAME=netflix-recommender
AI_PROVIDER=gemini
EMBEDDING_PROVIDER=gemini
NETFLIX_REGION=US
```

## 🎬 Usage

### Starting the Server

```bash
# Start the Netflix API server
python run.py

# Or directly
python main.py
```

The server will start on `http://localhost:8000`

### API Documentation

- **Interactive Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`
- **Netflix Stats**: `http://localhost:8000/netflix-stats`

### Quick Start Guide

1. **Populate Netflix Content** (run this first):
```bash
curl -X POST "http://localhost:8000/populate-netflix?max_pages=3"
```

2. **Get Netflix Recommendations**:
```bash
curl -X POST "http://localhost:8000/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "I love action movies with great fight scenes",
    "genre": "Action",
    "limit": 3
  }'
```

### Example Response

```json
{
  "recommendations": [
    {
      "movie": {
        "id": 123456,
        "title": "Extraction",
        "overview": "A black-market mercenary who has nothing to lose...",
        "release_date": "2020-04-24",
        "vote_average": 6.8,
        "vote_count": 2547,
        "genre_ids": [28, 53, 18],
        "media_type": "movie"
      },
      "why_you_would_like_it": "Based on your love for action movies with great fight scenes, you'll absolutely love Extraction! This Netflix original delivers exactly what you're looking for with spectacular hand-to-hand combat sequences, intense gun battles, and non-stop action. Chris Hemsworth's performance brings raw intensity to every fight scene.",
      "memorable_quotes": [
        "You drown not by falling into a river, but by staying submerged in it.",
        "We all have it coming, kid.",
        "This is not about the money anymore."
      ],
      "memorable_moments": [
        "The incredible 12-minute single-take action sequence through the streets of Dhaka",
        "The intense bridge fight scene with hand-to-hand combat",
        "The rooftop chase and gunfight finale"
      ]
    }
  ],
  "total_found": 1
}
```

## 🎯 Netflix-Specific Endpoints

### Core Operations
- `POST /populate-netflix` - Populate with Netflix content (optimized)
- `POST /populate-netflix-fast` - Fast parallel Netflix population
- `POST /recommendations` - Get Netflix recommendations
- `POST /recommendations-netflix-only` - Guaranteed Netflix-only results

### Information & Stats
- `GET /netflix-stats` - Netflix content statistics
- `GET /trending-netflix` - Current trending Netflix content
- `GET /debug-netflix/{genre}` - Debug Netflix content by genre
- `GET /stats` - Vector store statistics

### Management
- `DELETE /clear-netflix-index` - Clear all Netflix data
- `GET /health` - Service health check
- `GET /ai-info` - AI provider information

## ⚡ Performance Optimizations

### 1. **Efficient Population**
- Batch processing for faster embedding generation
- Smart rate limiting to avoid API quotas
- Parallel fetching for improved speed
- Duplicate removal and quality filtering

### 2. **Netflix-Only Focus**
- Uses TMDB watch providers to ensure Netflix availability
- Regional filtering (default: US)
- Quality filtering (minimum vote count)
- Adult content filtering

### 3. **Robust Error Handling**
- Automatic fallbacks when APIs are unavailable
- Retry logic for temporary failures
- Rate limit protection with intelligent delays
- Graceful degradation to hash-based embeddings

### 4. **Memory Management**
- Smaller batch sizes for reliability
- Efficient caching of embeddings
- Streaming processing for large datasets

## 🔧 Configuration Options

### AI Provider Settings
```env
# Primary: Gemini (recommended for free tier)
AI_PROVIDER=gemini
EMBEDDING_PROVIDER=gemini
GEMINI_MODEL=gemini-1.5-flash

# Fallback: OpenAI
OPENAI_CHAT_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002

# Fallback method when APIs unavailable
EMBEDDING_FALLBACK_METHOD=hash
```

### Netflix Settings
```env
# Netflix region (affects available content)
NETFLIX_REGION=US  # US, GB, CA, etc.

# Quality filtering
NETFLIX_MIN_VOTES=50
NETFLIX_INCLUDE_ADULT=false
```

### Performance Tuning
```env
# Batch sizes (smaller = more reliable)
VECTOR_BATCH_SIZE=20
TMDB_BATCH_SIZE=10

# Rate limiting (requests per time period)
GEMINI_REQUESTS_PER_MINUTE=12
TMDB_REQUESTS_PER_10_SECONDS=35
```

## 🐛 Troubleshooting

### Common Issues

#### 1. **Empty Recommendations**
```bash
# Check if data is populated
curl http://localhost:8000/netflix-stats

# If empty, populate first
curl -X POST "http://localhost:8000/populate-netflix?max_pages=3"
```

#### 2. **Timeout During Population**
```bash
# Use smaller batches
curl -X POST "http://localhost:8000/populate-netflix?max_pages=2"

# Or use fast mode with parallel processing
curl -X POST "http://localhost:8000/populate-netflix-fast?max_pages=2"
```

#### 3. **Rate Limit Errors**
- Check your API keys and quotas
- Reduce `max_pages` parameter
- Wait a few minutes and try again
- Check logs: `tail -f logs/app.log`

#### 4. **No Netflix Content Found**
```bash
# Debug what's in your vector store
curl "http://localhost:8000/debug-netflix/Action?query=test"

# Check TMDB API status
curl "https://api.themoviedb.org/3/configuration?api_key=YOUR_KEY"
```

### Performance Tips

1. **Start Small**: Begin with `max_pages=2` to test
2. **Use Gemini**: Free tier with generous limits
3. **Monitor Stats**: Check `/netflix-stats` regularly
4. **Regional Content**: Adjust `NETFLIX_REGION` for your location
5. **Quality First**: Keep vote count filtering for better results

## 📊 API Response Times

| Endpoint | Typical Response Time | Notes |
|----------|----------------------|-------|
| `/health` | < 100ms | Instant health check |
| `/recommendations` | 1-3 seconds | Depends on AI provider |
| `/populate-netflix` | 5-15 minutes | Background processing |
| `/populate-netflix-fast` | 2-8 minutes | Parallel processing |
| `/netflix-stats` | < 500ms | Vector store query |

## 🔒 Security & Privacy

- **No User Data Storage**: API is stateless
- **API Key Protection**: Keys stored in environment variables
- **Rate Limiting**: Built-in protection against abuse
- **Error Logging**: No sensitive data in logs
- **CORS Enabled**: For web application integration

## 🚀 Production Deployment

### Environment Variables for Production
```env
# Production settings
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO

# Increased limits for production
VECTOR_BATCH_SIZE=50
MAX_PAGES_DEFAULT=5

# Monitoring
ENABLE_METRICS=true
HEALTH_CHECK_INTERVAL=60
```

### Docker Deployment (Optional)
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

## 📈 Scaling Considerations

- **Vector Store**: Pinecone scales automatically
- **Rate Limits**: Upgrade API plans for higher throughput
- **Memory Usage**: Monitor during large populations
- **Regional Deployment**: Deploy closer to users for better latency

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/netflix-enhancement`
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational purposes. Please respect the terms of service for:
- TMDB API
- Pinecone
- Google Gemini AI
- OpenAI (if used)
- Netflix content usage policies

## 🙋‍♂️ Support

- **Issues**: Create a GitHub issue
- **Questions**: Check the API documentation at `/docs`
- **Performance**: Monitor `/netflix-stats` endpoint
- **Logs**: Check application logs for detailed error information

---

**🎬 Happy Netflix Binge-Watching with AI-Powered Recommendations! 🍿**