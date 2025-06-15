from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import httpx
import openai
import anthropic
import asyncio
from datetime import datetime, timedelta
import logging
import os
import hashlib
import json
from functools import lru_cache
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("News Processor API starting up...")
    yield
    # Shutdown
    logger.info("News Processor API shutting down...")

app = FastAPI(
    title="News Processor API", 
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic Models
class NewsRequest(BaseModel):
    url: Optional[str] = None
    topic: Optional[str] = None
    webhook_url: Optional[str] = None  # Optional callback URL
    
    class Config:
        schema_extra = {
            "example": {
                "url": "https://example.com/news-article",
                "topic": "AI developments in 2025",
                "webhook_url": "https://your-app.com/webhook/news-complete"
            }
        }

class NewsResponse(BaseModel):
    success: bool
    article: str
    seo: str
    models_used: str
    processing_time: float
    input_type: str
    extracted_facts: Optional[str] = None
    verified_facts: Optional[str] = None
    cached: bool = False  # Always False without Redis
    job_id: Optional[str] = None  # For async processing

class WebhookPayload(BaseModel):
    job_id: str
    status: str  # "completed", "failed"
    result: Optional[NewsResponse] = None
    error: Optional[str] = None
    timestamp: str

# Initialize AI clients
openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class NewsProcessor:
    def __init__(self):
        self.firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
        self.brave_api_key = os.getenv("BRAVE_API_KEY")
    
    async def _send_webhook(self, webhook_url: str, payload: WebhookPayload):
        """Send webhook notification"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    webhook_url,
                    json=payload.dict(),
                    headers={"Content-Type": "application/json"},
                    timeout=30.0
                )
                logger.info(f"Webhook sent to {webhook_url}, status: {response.status_code}")
        except Exception as e:
            logger.error(f"Webhook failed for {webhook_url}: {e}")
    
    async def scrape_url(self, url: str) -> str:
        """Scrape content from URL using Firecrawl"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.firecrawl.dev/v1/scrape",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.firecrawl_api_key}"
                    },
                    json={
                        "url": url,
                        "pageOptions": {
                            "onlyMainContent": True
                        }
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                return data.get("data", {}).get("markdown", "No content available")
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to scrape URL: {str(e)}")
    
    async def search_topic(self, topic: str) -> str:
        """Search for topic using Brave Search"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip",
                        "X-Subscription-Token": self.brave_api_key
                    },
                    params={"q": topic},
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                
                # Extract content from search results
                results = data.get("web", {}).get("results", [])
                if results:
                    # Combine top results
                    content_parts = []
                    for result in results[:3]:  # Top 3 results
                        if result.get("description"):
                            content_parts.append(result["description"])
                    return " ".join(content_parts) if content_parts else "No search results available"
                return "No search results available"
        except Exception as e:
            logger.error(f"Error searching topic {topic}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to search topic: {str(e)}")
    
    async def extract_facts_gpt4(self, content: str) -> str:
        """Extract facts using GPT-4.1"""
        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Extract ONLY verifiable, concrete facts from the provided content. Focus on WHO, WHAT, WHEN, WHERE, HOW MUCH, QUOTES, and DETAILS. EXCLUDE opinions, speculation, and analysis. FORMAT: Structured list of facts only.

Content: {content}"""
                    }
                ],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in GPT-4 fact extraction: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to extract facts: {str(e)}")
    
    async def verify_facts_o3mini(self, facts: str) -> str:
        """Verify and organize facts using o3-mini"""
        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Use advanced reasoning to verify and organize these extracted facts for journalism. Apply chain-of-thought reasoning to identify inconsistencies, rank by importance, and prepare for professional news writing.

Facts to verify: {facts}"""
                    }
                ],
                temperature=0.2,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in o3-mini fact verification: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to verify facts: {str(e)}")
    
    async def write_article_claude(self, verified_facts: str) -> str:
        """Write article using Claude Sonnet 4"""
        try:
            message = await anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Write a professional AP Style news article from these verified facts:

{verified_facts}

Requirements: 300-400 words, lead paragraph with most newsworthy fact, inverted pyramid structure, objective tone, proper attribution."""
                    }
                ]
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"Error in Claude article writing: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to write article: {str(e)}")
    
    async def generate_seo_gpt4(self, article: str) -> str:
        """Generate SEO metadata using GPT-4.1"""
        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Generate SEO metadata for this news article:

{article}

Provide: 5 SEO titles (60 chars), 5 meta descriptions (130 chars), 5 relevant keywords, and a suggested slug."""
                    }
                ],
                temperature=0.4,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in GPT-4 SEO generation: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate SEO: {str(e)}")

    async def process_news_pipeline(self, request: NewsRequest, job_id: str = None) -> NewsResponse:
        """Main processing pipeline"""
        start_time = datetime.now()
        
        # Process content (no caching for now)
        logger.info(f"Processing {'URL' if request.url else 'topic'}: {request.url or request.topic}")
        
        if request.url:
            content = await self.scrape_url(request.url)
            input_type = "URL"
        else:
            content = await self.search_topic(request.topic)
            input_type = "Topic"
        
        logger.info(f"Content extracted: {len(content)} characters")
        
        # Validate content length
        min_content_length = int(os.getenv("MIN_CONTENT_LENGTH", "200"))
        if len(content.strip()) < min_content_length:
            raise HTTPException(
                status_code=422, 
                detail=f"Insufficient content: {len(content)} characters (minimum: {min_content_length}). Content may be paywalled, blocked, or empty."
            )
        
        # AI Pipeline
        logger.info("Extracting facts with GPT-4.1...")
        extracted_facts = await self.extract_facts_gpt4(content)
        
        logger.info("Verifying facts with o3-mini...")
        verified_facts = await self.verify_facts_o3mini(extracted_facts)
        
        logger.info("Writing article with Claude Sonnet 4...")
        article = await self.write_article_claude(verified_facts)
        
        logger.info("Generating SEO with GPT-4.1...")
        seo = await self.generate_seo_gpt4(article)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = NewsResponse(
            success=True,
            article=article,
            seo=seo,
            models_used="GPT-4.1 (facts + SEO), o3-mini (verification), Claude Sonnet 4 (writing)",
            processing_time=processing_time,
            input_type=input_type,
            extracted_facts=extracted_facts,
            verified_facts=verified_facts,
            cached=False,
            job_id=job_id
        )
        
        logger.info(f"Pipeline completed in {processing_time:.2f} seconds")
        return result

# Initialize processor
processor = NewsProcessor()

# Background task for async processing
async def process_news_background(request: NewsRequest, job_id: str):
    """Background task for async news processing"""
    try:
        result = await processor.process_news_pipeline(request, job_id)
        
        # Send webhook if provided
        if request.webhook_url:
            payload = WebhookPayload(
                job_id=job_id,
                status="completed",
                result=result,
                timestamp=datetime.now().isoformat()
            )
            await processor._send_webhook(request.webhook_url, payload)
            
    except Exception as e:
        logger.error(f"Background processing failed for job {job_id}: {e}")
        
        # Send error webhook if provided
        if request.webhook_url:
            payload = WebhookPayload(
                job_id=job_id,
                status="failed",
                error=str(e),
                timestamp=datetime.now().isoformat()
            )
            await processor._send_webhook(request.webhook_url, payload)

@app.post("/news-processor", response_model=NewsResponse)
async def process_news_sync(request: NewsRequest):
    """
    Synchronous news processing endpoint.
    Returns result immediately after processing.
    """
    # Validate input
    if not request.url and not request.topic:
        raise HTTPException(status_code=400, detail="Either 'url' or 'topic' must be provided")
    
    if request.url and request.topic:
        raise HTTPException(status_code=400, detail="Provide either 'url' OR 'topic', not both")
    
    try:
        return await processor.process_news_pipeline(request)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in news processing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/news-processor/async")
async def process_news_async(request: NewsRequest, background_tasks: BackgroundTasks):
    """
    Asynchronous news processing endpoint.
    Returns job_id immediately, sends result via webhook when complete.
    Requires webhook_url in request.
    """
    # Validate input
    if not request.url and not request.topic:
        raise HTTPException(status_code=400, detail="Either 'url' or 'topic' must be provided")
    
    if request.url and request.topic:
        raise HTTPException(status_code=400, detail="Provide either 'url' OR 'topic', not both")
    
    if not request.webhook_url:
        raise HTTPException(status_code=400, detail="webhook_url is required for async processing")
    
    # Generate job ID
    job_id = hashlib.md5(f"{request.url or request.topic}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
    
    # Start background processing
    background_tasks.add_task(process_news_background, request, job_id)
    
    return {
        "success": True,
        "job_id": job_id,
        "status": "processing",
        "message": "Job started. Result will be sent to webhook_url when complete.",
        "webhook_url": request.webhook_url
    }

@app.post("/webhook/news-processor", response_model=NewsResponse)
async def webhook_endpoint(request: Request):
    """
    Webhook endpoint for external systems to trigger news processing.
    Accepts same payload as /news-processor but via webhook.
    """
    try:
        body = await request.json()
        news_request = NewsRequest(**body)
        return await process_news_sync(news_request)
        
    except Exception as e:
        logger.error(f"Webhook processing error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid webhook payload: {str(e)}")

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Check status of async job (if using external job tracking).
    This is a placeholder - you'd implement with your job storage.
    """
    return {
        "job_id": job_id,
        "status": "unknown",
        "message": "Job tracking not implemented. Use webhook for async results."
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "News Processor API",
        "version": "1.0.0",
        "endpoints": {
            "sync_process": "/news-processor",
            "async_process": "/news-processor/async", 
            "webhook": "/webhook/news-processor",
            "job_status": "/job/{job_id}",
            "health": "/health",
            "docs": "/docs"
        },
        "features": [
            "Multi-model AI pipeline",
            "Async processing with webhooks", 
            "Content validation",
            "Comprehensive logging"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)