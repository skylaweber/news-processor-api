# News Processor API

AI-powered news processing pipeline that converts URLs or topics into professional articles with SEO metadata.

## Features

- **Multi-model AI pipeline**: GPT-4.1 (extraction + SEO) → o3-mini (verification) → Claude Sonnet 4 (writing)
- **Smart caching**: Redis-based caching with 24-hour TTL
- **Webhook support**: Async processing with callback notifications
- **Flexible input**: Process URLs (via Firecrawl) or topics (via Brave Search)

## Quick Start

1. **Install dependencies:**
```bash
pip3 install -r requirements.txt
```

2. **Set environment variables:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. **Run locally:**
```bash
python3 main.py
```

4. **Test the API:**
Visit `http://localhost:8000/docs` for interactive documentation.

## API Endpoints

### Synchronous Processing
```bash
POST /news-processor
{
  "url": "https://example.com/article"
}
# OR
{
  "topic": "AI developments 2025"
}
```

### Asynchronous Processing
```bash
POST /news-processor/async
{
  "topic": "breaking news",
  "webhook_url": "https://your-app.com/callback"
}
```

### Webhook Endpoint
```bash
POST /webhook/news-processor
{
  "url": "https://example.com/article"
}
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-4.1 and o3-mini |
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for Claude Sonnet 4 |
| `FIRECRAWL_API_KEY` | Yes | Firecrawl API key for URL scraping |
| `BRAVE_API_KEY` | Yes | Brave Search API key for topic search |
| `REDIS_URL` | No | Redis connection URL (defaults to localhost) |
| `CACHE_TTL_HOURS` | No | Cache time-to-live in hours (default: 24) |

## Deployment

### Render (Recommended - Free Tier)
1. Connect this GitHub repo to Render
2. **Build Command:** `pip3 install -r requirements.txt`
3. **Start Command:** `python3 -m uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables in Render dashboard
5. Deploy automatically on git push to main

### Railway/Vercel/Fly.io
Also supported - adjust build commands to use `python3` and `pip3`.

## Response Format

```json
{
  "success": true,
  "article": "Generated news article...",
  "seo": "SEO metadata...",
  "models_used": "GPT-4.1 (facts + SEO), o3-mini (verification), Claude Sonnet 4 (writing)",
  "processing_time": 12.34,
  "input_type": "URL",
  "cached": false,
  "extracted_facts": "Raw facts...",
  "verified_facts": "Verified facts..."
}
```

## Architecture

```
Input (URL/Topic) 
    ↓
Route & Extract Content (Firecrawl/Brave)
    ↓
Extract Facts (GPT-4.1)
    ↓
Verify & Organize (o3-mini)
    ↓
Write Article (Claude Sonnet 4)
    ↓
Generate SEO (GPT-4.1)
    ↓
Return/Webhook Result
```

## License

MIT License