"""External service integrations: web scraping and YouTube search"""

import requests
from typing import List, Dict
from src.config import (
    JINA_URL_TEMPLATE,
    SCRAPE_TIMEOUT,
    MAX_SCRAPE_CHARS,
    YOUTUBE_API_KEY,
    YOUTUBE_API_URL,
    YOUTUBE_MAX_RESULTS
)


def scrape_url(url: str) -> str:
    """
    Scrape website content using Jina AI reader
    
    Args:
        url: Website URL to scrape
        
    Returns:
        Scraped content (up to MAX_SCRAPE_CHARS) or empty string on failure
    """
    try:
        jina_url = JINA_URL_TEMPLATE.format(url=url)
        response = requests.get(jina_url, timeout=SCRAPE_TIMEOUT)
        if response.status_code == 200:
            return response.text[:MAX_SCRAPE_CHARS]
        return ""
    except Exception:
        return ""


def search_youtube(company: str, role: str) -> List[Dict[str, str]]:
    """
    Search YouTube for interview preparation videos
    
    Args:
        company: Company name
        role: Job role/title
        
    Returns:
        List of video dictionaries with title, url, channel, and thumbnail
    """
    try:
        if not YOUTUBE_API_KEY:
            return []
        
        query = f"{company} {role} interview tips preparation"
        params = {
            'part': 'snippet',
            'q': query,
            'type': 'video',
            'maxResults': YOUTUBE_MAX_RESULTS,
            'key': YOUTUBE_API_KEY,
            'order': 'relevance'
        }
        
        response = requests.get(YOUTUBE_API_URL, params=params, timeout=SCRAPE_TIMEOUT)
        data = response.json()
        
        videos = []
        for item in data.get('items', []):
            videos.append({
                'title': item['snippet']['title'],
                'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                'channel': item['snippet']['channelTitle'],
                'thumbnail': item['snippet']['thumbnails']['medium']['url']
            })
        
        return videos
        
    except Exception as e:
        print(f"YouTube search error: {e}")
        return []

