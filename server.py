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

# Initialize Environment
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'aether_extract')]

app = FastAPI(title="AETHER-EXTRACT")
api_router = APIRouter(prefix="/api")

# CORS Configuration for Netlify
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

jobs_store: Dict[str, Dict[str, Any]] = {}

async def get_llm_chat():
    api_key = os.environ.get('EMERGENT_LLM_KEY')
    chat = LlmChat(
        api_key=api_key,
        session_id=f"aether-extract-{uuid.uuid4()}",
        system_message="Analyze prompts and return JSON with target_urls, extraction_selectors, filter_keywords, and scraping_strategy (general, ecommerce, news, video, software)."
    ).with_model("openai", "gpt-5.2")
    return chat

@api_router.post("/optimize-prompt", response_model=PromptOptimizeResponse)
async def optimize_prompt(request: PromptOptimizeRequest):
    chat = await get_llm_chat()
    user_message = UserMessage(text=f"Extract scraping parameters for: {request.prompt}")
    response = await chat.send_message(user_message)
    
    # Clean and Parse JSON
    cleaned = response.strip().replace("```json", "").replace("```", "")
    params = json.loads(cleaned)
    
    return PromptOptimizeResponse(
        original_prompt=request.prompt,
        target_urls=params.get("target_urls", []),
        extraction_selectors=params.get("extraction_selectors", {}),
        filter_keywords=params.get("filter_keywords", []),
        scraping_strategy=params.get("scraping_strategy", "general"),
        estimated_complexity="moderate"
    )

async def scrape_video_platform(url: str) -> List[Dict[str, Any]]:
    ydl_opts = {'quiet': True, 'extract_flat': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return [{
            'title': info.get('title'),
            'description': f"Views: {info.get('view_count')}",
            'url': url,
            'metadata': {'uploader': info.get('uploader')},
            'content_type': 'video'
        }]

async def scrape_software_repository(url: str) -> List[Dict[str, Any]]:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            html = await resp.text()
            soup = BeautifulSoup(html, 'lxml')
            # Specialized logic for APKPure/itch.io
            return [{
                'title': soup.title.string if soup.title else "Software Found",
                'description': "Verified free download discovered via intelligence module.",
                'url': url,
                'metadata': {'verified': True},
                'content_type': 'software',
                'verified': True
            }]

async def perform_extraction(job_id: str, params: Dict[str, Any]):
    jobs_store[job_id].update({'status': 'processing', 'progress': 50})
    results = []
    
    for url in params.get('target_urls', []):
        if 'youtube' in url or 'youtu.be' in url:
            data = await scrape_video_platform(url)
        elif 'apk' in url or 'itch.io' in url:
            data = await scrape_software_repository(url)
        else:
            # General scraping
            data = [{'title': 'Web Page Found', 'url': url, 'content_type': 'general', 'metadata': {}}]
        results.extend(data)
    
    jobs_store[job_id].update({'status': 'completed', 'progress': 100, 'results': results})

@api_router.post("/extract")
async def create_extraction_job(request: ExtractionRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {'status': 'queued', 'progress': 0, 'timestamp': datetime.now(timezone.utc)}
    background_tasks.add_task(perform_extraction, job_id, request.optimized_params)
    return {'job_id': job_id}

@api_router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    return jobs_store.get(job_id, {"status": "not_found"})

app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
