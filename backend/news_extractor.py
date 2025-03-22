from typing import List, Dict, Any, Set
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import random
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from search_engines import search_google_news, search_bing_news, search_yahoo_news

    
def extract_company_news(company_name: str, num_articles: int = 10) -> List[Dict[str, Any]]:
    """
    Extract news articles about a company from multiple search engines in parallel.
    
    Args:
        company_name: The name of the company to search for
        num_articles: Maximum number of articles to return
        
    Returns:
        List of extracted news article dictionaries
    """
    # List to store extracted news articles
    news_articles = []
    processed_urls = set()
    
    # Configure retry strategy directly within this function
    session = requests.Session()
    
    # Add retry logic directly (no separate function needed)
    retry_strategy = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # User agent to mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

    articles_lock = Lock()
    remaining_count_lock = Lock()
    remaining_count = num_articles
    
    def search_and_process(search_function, search_name, args):
        """Helper function to run a search and collect results"""
        nonlocal remaining_count

        try:
            print(f"Starting {search_name} search for '{company_name}'")
            with remaining_count_lock:
                if remaining_count <= 0:
                    print(f"Already have enough articles, skipping {search_name}")
                    return 0

                target_count = min(remaining_count + 2, num_articles)  # +2 as buffer
            
            # Inject session into the args if it's GoogleNews and needs pagination
            if "Google News" in search_name:
                # If we have a page param (tuple of 5 elements), add session as 6th arg
                if len(args) == 5:
                    # Since args is immutable tuple, create a new one
                    args = args + (session,)
            
            results = search_function(*args)
            
            # Thread-safe update of news_articles and remaining count
            articles_to_add = []
            with articles_lock:
                # Add articles that aren't duplicates
                for article in results:
                    url = article.get('url', '')
                    # Skip if URL already exists in news_articles
                    if url and url not in [a.get('url', '') for a in news_articles]:
                        articles_to_add.append(article)
                        with remaining_count_lock:
                            remaining_count -= 1
                
                news_articles.extend(articles_to_add)
                print(f"Added {len(articles_to_add)} articles from {search_name}. Total: {len(news_articles)}/{num_articles}")
                
            return len(articles_to_add)
        except Exception as e:
            print(f"Error during {search_name} search: {e}")
            return 0
    
    # Define search tasks to run in parallel - with more Google pages
    search_tasks = [
        (search_google_news, "Google News page 1", (company_name, num_articles, headers, processed_urls, 1)),
        (search_google_news, "Google News page 2", (company_name, num_articles, headers, processed_urls, 2)),
        (search_google_news, "Google News page 3", (company_name, num_articles, headers, processed_urls, 3)),
        (search_google_news, "Google News page 4", (company_name, num_articles, headers, processed_urls, 4)),
        (search_google_news, "Google News page 5", (company_name, num_articles, headers, processed_urls, 5)),
        (search_google_news, "Google News page 6", (company_name, num_articles, headers, processed_urls, 6)),
        (search_bing_news, "Bing News", (company_name, num_articles, headers, processed_urls, session)),
        (search_yahoo_news, "Yahoo News", (company_name, num_articles, headers, processed_urls, session))
    ]
    
    # Process all search engines in parallel
    print(f"Starting parallel searches for news about '{company_name}'")
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all search tasks
        future_to_search = {
            executor.submit(search_and_process, func, name, args): name 
            for func, name, args in search_tasks
        }
        
        # Process results as they complete
        for future in as_completed(future_to_search):
            search_name = future_to_search[future]
            try:
                articles_found = future.result()
                print(f"{search_name} search completed, found {articles_found} articles")
                
                # Check if we have enough articles already
                if len(news_articles) >= num_articles * 1.5:
                    # Cancel any remaining tasks
                    for f in future_to_search:
                        if not f.done():
                            f.cancel()
                    print(f"Found {len(news_articles)} articles, stopping remaining searches")
                    break
                    
            except Exception as e:
                print(f"Error in {search_name} search: {e}")
    
    # Limit to requested number of articles and remove duplicates if any slipped through
    unique_articles = []
    seen_urls = set()
    
    for article in news_articles:
        url = article.get('url', '')
        if url and url not in seen_urls and len(unique_articles) < num_articles:
            seen_urls.add(url)
            unique_articles.append(article)
    
    print(f"Total unique articles found: {len(unique_articles)}/{num_articles} requested")
    
    # Process the articles and return
    combined_article_info = display_news(unique_articles)
    return combined_article_info


def display_news(news_articles: List[Dict[str, Any]]) -> None:
    """
    Display the extracted news articles in a readable format.
    
    Args:
        news_articles: List of news article dictionaries
    """
    if not news_articles:
        print("No news articles found.")
        return
        
    print(f"\n{'=' * 80}")
    print(f"Found {len(news_articles)} news articles about the company:")
    print(f"{'=' * 80}\n")
    
    combined = []
    for i, article in enumerate(news_articles, 1):
        # Extract data from the article
        url = article.get('url', 'No url available')
        title = article.get('title', 'No title available')
        source = article.get('source', 'Unknown')
        date = article.get('date', 'Unknown')
        original_content = article.get('raw_content', 'No content available')
        author = article.get('author', 'Unknown')
        read_time = article.get('read_time', 'Unknown')
        industry = article.get('industry', 'Unknown')
        
        # Extract metadata from the article
        summary = article.get('summary', 'No summary available')
        keywords = article.get('keywords', [])
        relevance = article.get('relevance', 'Unknown')
        main_topic = article.get('main_topic', 'Uncategorized')
        
        # Combine all data into a single dictionary
        combined_info = {
            'url': url,
            'title': title,
            'source': source,
            'date': date,
            'summary': summary,
            'keywords': keywords,
            'relevance': relevance,
            'raw_content': original_content,
            'author': author,
            'read_time': read_time,
            'industry': industry,
            'main_topic': main_topic
        }

        combined.append(combined_info)

    return combined



