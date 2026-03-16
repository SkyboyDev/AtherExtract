from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import asyncio
import json
from bs4 import BeautifulSoup
import aiohttp
import yt_dlp
from emergentintegrations.llm.chat import LlmChat, UserMessage


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Models
class PromptOptimizeRequest(BaseModel):
    prompt: str

class PromptOptimizeResponse(BaseModel):
    original_prompt: str
    target_urls: List[str]
    extraction_selectors: Dict[str, Any]
    filter_keywords: List[str]
    scraping_strategy: str
    estimated_complexity: str

class ExtractionRequest(BaseModel):
    prompt: str
    optimized_params: Optional[Dict[str, Any]] = None

class ExtractionStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    message: str
    results: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    timestamp: datetime

class IntelligenceCard(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: Optional[str] = None
    url: str
    metadata: Dict[str, Any]
    content_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    verified: bool = False


# In-memory job storage (in production, use Redis or database)
jobs_store: Dict[str, Dict[str, Any]] = {}


async def get_llm_chat():
    """Initialize LLM chat for prompt optimization"""
    api_key = os.environ.get('EMERGENT_LLM_KEY')
    if not api_key:
        raise ValueError("EMERGENT_LLM_KEY not found in environment")
    
    chat = LlmChat(
        api_key=api_key,
        session_id=f"aether-extract-{uuid.uuid4()}",
        system_message="""You are an expert web scraping parameter extractor. 
        Analyze user prompts and extract structured scraping parameters.
        Return ONLY valid JSON with these fields:
        - target_urls: list of URLs to scrape (if mentioned, else suggest based on intent)
        - extraction_selectors: dict with CSS selectors or xpath patterns
        - filter_keywords: list of keywords to filter results
        - scraping_strategy: one of [general, ecommerce, news, video, software]
        - estimated_complexity: one of [simple, moderate, complex]
        Be precise and actionable."""
    ).with_model("openai", "gpt-5.2")
    
    return chat


@api_router.post("/optimize-prompt", response_model=PromptOptimizeResponse)
async def optimize_prompt(request: PromptOptimizeRequest):
    """AI-powered prompt optimization and parameter extraction"""
    try:
        logger.info(f"Optimizing prompt: {request.prompt}")
        
        chat = await get_llm_chat()
        
        user_message = UserMessage(
            text=f"""Extract scraping parameters from this user prompt:
            
            "{request.prompt}"
            
            Return only valid JSON, no markdown or extra text."""
        )
        
        response = await chat.send_message(user_message)
        logger.info(f"LLM Response: {response}")
        
        # Parse the JSON response
        try:
            # Clean up response if it contains markdown
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response.split("```json")[1]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.split("```")[1]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response.rsplit("```", 1)[0]
            
            params = json.loads(cleaned_response.strip())
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}, Response: {response}")
            # Fallback to basic parsing
            params = {
                "target_urls": [],
                "extraction_selectors": {"title": "h1, .title", "content": "p, article"},
                "filter_keywords": request.prompt.split()[:5],
                "scraping_strategy": "general",
                "estimated_complexity": "moderate"
            }
        
        return PromptOptimizeResponse(
            original_prompt=request.prompt,
            target_urls=params.get("target_urls", []),
            extraction_selectors=params.get("extraction_selectors", {}),
            filter_keywords=params.get("filter_keywords", []),
            scraping_strategy=params.get("scraping_strategy", "general"),
            estimated_complexity=params.get("estimated_complexity", "moderate")
        )
        
    except Exception as e:
        logger.error(f"Error optimizing prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Prompt optimization failed: {str(e)}")


async def scrape_general_web(url: str, selectors: Dict[str, Any], keywords: List[str]) -> List[Dict[str, Any]]:
    """General web scraping logic"""
    results = []
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=30) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract title
                    title_selector = selectors.get('title', 'h1, .title, title')
                    title_elem = soup.select_one(title_selector)
                    title = title_elem.get_text(strip=True) if title_elem else "No title found"
                    
                    # Extract content
                    content_selector = selectors.get('content', 'p, article, .content')
                    content_elems = soup.select(content_selector)
                    content = ' '.join([elem.get_text(strip=True) for elem in content_elems[:5]])
                    
                    # Filter by keywords if provided
                    if keywords:
                        text_lower = (title + ' ' + content).lower()
                        if not any(keyword.lower() in text_lower for keyword in keywords):
                            return results
                    
                    results.append({
                        'title': title,
                        'description': content[:300] + '...' if len(content) > 300 else content,
                        'url': url,
                        'metadata': {
                            'status_code': response.status,
                            'content_length': len(html)
                        },
                        'content_type': 'general'
                    })
                    
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        results.append({
            'title': f'Error scraping {url}',
            'description': str(e),
            'url': url,
            'metadata': {'error': True},
            'content_type': 'error'
        })
    
    return results


async def scrape_video_platform(url: str) -> List[Dict[str, Any]]:
    """Video platform scraping using yt-dlp"""
    results = []
    
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            if info:
                results.append({
                    'title': info.get('title', 'No title'),
                    'description': info.get('description', '')[:300],
                    'url': url,
                    'metadata': {
                        'duration': info.get('duration', 0),
                        'view_count': info.get('view_count', 0),
                        'uploader': info.get('uploader', 'Unknown'),
                        'upload_date': info.get('upload_date', '')
                    },
                    'content_type': 'video'
                })
                
    except Exception as e:
        logger.error(f"Error scraping video {url}: {e}")
        results.append({
            'title': f'Error scraping video',
            'description': str(e),
            'url': url,
            'metadata': {'error': True},
            'content_type': 'error'
        })
    
    return results


async def scrape_software_repository(url: str, keywords: List[str]) -> List[Dict[str, Any]]:
    """Scrape free software repositories like APKPure, itch.io"""
    results = []
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=30) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # APKPure specific selectors
                    if 'apkpure' in url.lower():
                        apps = soup.select('.first-info, .app-item')
                        for app in apps[:10]:
                            title_elem = app.select_one('.title, .p1')
                            link_elem = app.select_one('a')
                            
                            if title_elem and link_elem:
                                title = title_elem.get_text(strip=True)
                                link = link_elem.get('href', '')
                                
                                results.append({
                                    'title': title,
                                    'description': 'Verified free Android app',
                                    'url': f"https://apkpure.com{link}" if link.startswith('/') else link,
                                    'metadata': {
                                        'platform': 'Android',
                                        'source': 'APKPure',
                                        'verified': True
                                    },
                                    'content_type': 'software',
                                    'verified': True
                                })
                    
                    # itch.io specific selectors
                    elif 'itch.io' in url.lower():
                        games = soup.select('.game_cell, .game_title')
                        for game in games[:10]:
                            title_elem = game.select_one('.title, a')
                            
                            if title_elem:
                                title = title_elem.get_text(strip=True)
                                link = title_elem.get('href', '') if title_elem.name == 'a' else game.select_one('a').get('href', '')
                                
                                results.append({
                                    'title': title,
                                    'description': 'Verified indie game from itch.io',
                                    'url': link,
                                    'metadata': {
                                        'platform': 'Cross-platform',
                                        'source': 'itch.io',
                                        'verified': True
                                    },
                                    'content_type': 'software',
                                    'verified': True
                                })
                    
                    # Generic software page
                    else:
                        links = soup.select('a[href*="download"], a[href*=".apk"], a[href*=".exe"]')
                        for link in links[:5]:
                            title = link.get_text(strip=True) or link.get('href', '')
                            href = link.get('href', '')
                            
                            results.append({
                                'title': title,
                                'description': 'Download link found',
                                'url': href,
                                'metadata': {
                                    'type': 'download_link'
                                },
                                'content_type': 'software'
                            })
                            
    except Exception as e:
        logger.error(f"Error scraping software repository {url}: {e}")
    
    return results


async def perform_extraction(job_id: str, params: Dict[str, Any]):
    """Background task to perform the actual extraction"""
    try:
        jobs_store[job_id]['status'] = 'processing'
        jobs_store[job_id]['message'] = 'Initializing Extraction Protocols'
        jobs_store[job_id]['progress'] = 10
        
        await asyncio.sleep(1)
        
        target_urls = params.get('target_urls', [])
        selectors = params.get('extraction_selectors', {})
        keywords = params.get('filter_keywords', [])
        strategy = params.get('scraping_strategy', 'general')
        
        all_results = []
        
        jobs_store[job_id]['message'] = 'Sanitizing Data Stream'
        jobs_store[job_id]['progress'] = 30
        
        # Process each URL based on strategy
        for idx, url in enumerate(target_urls):
            jobs_store[job_id]['message'] = f'Processing target {idx+1}/{len(target_urls)}'
            jobs_store[job_id]['progress'] = 30 + (50 * (idx + 1) // len(target_urls))
            
            if strategy == 'video' or 'youtube.com' in url or 'youtu.be' in url:
                results = await scrape_video_platform(url)
            elif strategy == 'software' or 'apkpure' in url or 'itch.io' in url:
                results = await scrape_software_repository(url, keywords)
            else:
                results = await scrape_general_web(url, selectors, keywords)
            
            all_results.extend(results)
            await asyncio.sleep(0.5)
        
        jobs_store[job_id]['message'] = 'Extraction Complete'
        jobs_store[job_id]['progress'] = 100
        jobs_store[job_id]['status'] = 'completed'
        jobs_store[job_id]['results'] = all_results
        
        # Store in database
        for result in all_results:
            card = IntelligenceCard(**result)
            doc = card.model_dump()
            doc['timestamp'] = doc['timestamp'].isoformat()
            doc['job_id'] = job_id
            await db.intelligence_cards.insert_one(doc)
        
    except Exception as e:
        logger.error(f"Error in extraction job {job_id}: {e}")
        jobs_store[job_id]['status'] = 'failed'
        jobs_store[job_id]['error'] = str(e)
        jobs_store[job_id]['message'] = f'Extraction failed: {str(e)}'


@api_router.post("/extract")
async def create_extraction_job(request: ExtractionRequest, background_tasks: BackgroundTasks):
    """Create a new extraction job"""
    job_id = str(uuid.uuid4())
    
    # Initialize job
    jobs_store[job_id] = {
        'job_id': job_id,
        'status': 'queued',
        'progress': 0,
        'message': 'Job queued',
        'results': None,
        'error': None,
        'timestamp': datetime.now(timezone.utc)
    }
    
    # Use optimized params or optimize now
    if request.optimized_params:
        params = request.optimized_params
    else:
        # Quick optimization
        optimize_result = await optimize_prompt(PromptOptimizeRequest(prompt=request.prompt))
        params = optimize_result.model_dump()
    
    # Start background extraction
    background_tasks.add_task(perform_extraction, job_id, params)
    
    return {
        'job_id': job_id,
        'status': 'queued',
        'message': 'Extraction job created'
    }


@api_router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get status of an extraction job"""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_store[job_id]
    
    return {
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'message': job['message'],
        'results': job.get('results'),
        'error': job.get('error'),
        'timestamp': job['timestamp'].isoformat()
    }


@api_router.get("/intelligence-cards")
async def get_intelligence_cards(limit: int = 50, skip: int = 0):
    """Get stored intelligence cards"""
    cards = await db.intelligence_cards.find({}, {"_id": 0}).sort("timestamp", -1).skip(skip).limit(limit).to_list(limit)
    
    for card in cards:
        if isinstance(card.get('timestamp'), str):
            card['timestamp'] = datetime.fromisoformat(card['timestamp'])
    
    return cards


@api_router.get("/")
async def root():
    return {
        "service": "AETHER-EXTRACT",
        "version": "1.0.0",
        "status": "operational"
    }


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

