            # Step 6: SEO generation
            logger.info("Step 5: Generating SEO metadata")
            seo = await self.generate_seo_gpt4(article, request.topic)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Build comprehensive response
            result = NewsResponse(
                success=True,
                article=article,
                seo=seo,
                models_used="GPT-4.1 (facts + SEO), o3-mini (verification), Claude Sonnet 4 (writing)",
                processing_time=processing_time,
                input_type="Topic",
                # Enhanced tracking data
                search_query=search_query,
                search_method=search_method,
                num_results=len(search_results),
                search_results=search_results,
                content_source=content_source,
                raw_content=raw_content[:5000] + "..." if len(raw_content) > 5000 else raw_content,  # Truncate for display
                extracted_facts=extracted_facts,
                facts_count=facts_count,
                verified_facts=verified_facts,
                verification_status=verification_status,
                quality_issues=quality_issues,
                sources_used=sources_used,
                cached=False,
                job_id=job_id
            )
            
            logger.info(f"Enhanced pipeline completed in {processing_time:.2f} seconds")
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in enhanced pipeline: {e}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# Initialize enhanced processor
processor = EnhancedNewsProcessor()

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
    Enhanced synchronous news processing endpoint.
    Now includes comprehensive source tracking and quality control.
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
    Enhanced asynchronous news processing endpoint.
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
    Check status of async job.
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

@app.head("/")
async def root_head():
    """HEAD method for root endpoint"""
    return {}

@app.head("/health")
async def health_head():
    """HEAD method for health endpoint"""
    return {}

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Enhanced News Processor API",
        "version": "2.0.0",
        "endpoints": {
            "sync_process": "/news-processor",
            "async_process": "/news-processor/async", 
            "webhook": "/webhook/news-processor",
            "job_status": "/job/{job_id}",
            "health": "/health",
            "docs": "/docs"
        },
        "features": [
            "Enhanced multi-source pipeline",
            "Quality-controlled source filtering",
            "Comprehensive fact verification", 
            "Source attribution and tracking",
            "Detailed processing transparency",
            "Quality gates and issue detection"
        ],
        "new_in_v2": [
            "Multi-source content aggregation",
            "Source credibility scoring",
            "Enhanced fact verification", 
            "Quality issue detection",
            "Detailed pipeline tracking",
            "Date filtering for recent content"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
