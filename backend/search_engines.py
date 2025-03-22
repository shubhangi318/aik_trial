import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Any, Set
import time
import random

from article_processor import process_urls_in_parallel

def search_google_news(company_name: str, num_articles: int, headers: Dict[str, str], processed_urls: Set[str], page: int = 1, session=None) -> List[Dict[str, Any]]:
    """Search for news articles on Google News"""
    news_articles = []
    all_article_urls = []
    
    # Create a session if not provided
    if session is None:
        session = requests.Session()
    
    # Search query for Google News
    query = f"{company_name} bistro news"
    encoded_query = query.replace(' ', '+')

    # Use just the single page that was requested (pagination handled by parallel calls)
    start_param = 0 if page == 1 else (page - 1) * 10
    url = f"https://www.google.com/search?q={encoded_query}&tbm=nws&start={start_param}"
    
    print(f"Searching Google News page {page} for '{company_name}'")
    
    try:
        # Progressive delay for higher page numbers
        if page > 1:
            delay = min(5 + (page * 0.5), 15)  # Start with 5.5s for page 2, max 15s
            print(f"Waiting {delay:.1f}s before requesting Google page {page}")
            time.sleep(delay)
        
        # Rotate user agent for each request
        if page > 1:
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
            ]
            headers = headers.copy()
            headers['User-Agent'] = random.choice(user_agents)
        
        # Send request to Google News
        response = session.get(url, headers=headers, timeout=15)

        if response.status_code != 200:
            print(f"Google returned status code {response.status_code} for page {page}")
            time.sleep(random.uniform(10, 20))  # Longer pause on error
            return []
    
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Check for CAPTCHA or blocking indicators
        if "unusual traffic" in response.text.lower() or "captcha" in response.text.lower():
            print(f"Google has detected unusual traffic on page {page}, backing off")
            time.sleep(random.uniform(30, 60))  # Very long pause if detected
            return []
        
        # Find news article elements
        article_elements = soup.find('div', id='center_col')

        if not article_elements:
            print(f"No article container found on page {page}")
            return []
    
        divs = article_elements.find_all('div', class_='SoaBEf')
        if not divs:
            divs = article_elements.find_all('g-card')
        if not divs:
            divs = article_elements.find_all('div', class_=['dbsr', 'WlydOe'])
            
        print(f"Found {len(divs)} article elements on page {page}")

        # Extract URLs from the found articles
        page_urls = []
        for div in divs:
            try:
                # Extract link
                link_element = div.find('a')
                url = link_element['href'] if link_element and 'href' in link_element.attrs else ""
                
                # Check if URL is valid
                if not url or not url.startswith('http'):
                    if url.startswith('/url?'):
                        url_match = re.search(r'url\?q=([^&]+)', url)
                        if url_match:
                            url = url_match.group(1)
                    else:
                        continue
                
                # Skip if already processed
                if url in processed_urls:
                    continue
                
                page_urls.append(url)
                processed_urls.add(url)
                
            except Exception as e:
                print(f"Error extracting URL: {e}")
                continue
        
        all_article_urls.extend(page_urls)
        
        # Use parallel processing to extract content from URLs
        if page_urls:
            print(f"Processing {len(page_urls)} URLs from Google page {page}")
            # Add small delays between article processing
            modified_headers = headers.copy()
            modified_headers['X-Add-Delay'] = 'true'
            news_articles.extend(process_urls_in_parallel(page_urls, modified_headers, company_name, num_articles))
            
    except Exception as e:
        print(f"Error during Google News extraction (page {page}): {e}")
        time.sleep(random.uniform(5, 10))  # Pause on error
    
    print(f"Extracted {len(news_articles)} articles from Google News page {page}")
    return news_articles

def search_bing_news(company_name: str, num_articles: int, headers: Dict[str, str], processed_urls: Set[str], session=None) -> List[Dict[str, Any]]:
    """Search for news articles on Bing News"""
    news_articles = []
    all_article_urls = []
    
    # Create a session if not provided
    if session is None:
        session = requests.Session()
    
    # Search query for Bing News
    query = f"{company_name} bistro news"
    encoded_query = query.replace(' ', '+')
    
    # Use pagination for Bing too (pages 1-3)
    for page in range(1, 4):
        if len(all_article_urls) >= num_articles * 2:
            break
        
        # Bing uses 'first' parameter for pagination
        first_param = 1 + ((page - 1) * 10) if page > 1 else 1
        url = f"https://www.bing.com/news/search?q={encoded_query}&first={first_param}"
        
        print(f"Searching Bing News page {page} for '{company_name}'")
        
        try:
            # Add delay between pages
            if page > 1:
                delay = random.uniform(3, 7)
                print(f"Waiting {delay:.1f}s before requesting Bing page {page}")
                time.sleep(delay)
            
            # Rotate user agent
            headers = headers.copy()
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
            ]
            headers['User-Agent'] = random.choice(user_agents)
            
            # Send request to Bing News
            response = session.get(url, headers=headers, timeout=15)
            
            if response.status_code != 200:
                print(f"Bing returned status code {response.status_code} for page {page}")
                time.sleep(random.uniform(10, 20))
                continue
            
            # Parse HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news article elements (Bing's structure)
            news_cards = soup.find_all('div', class_='news-card')
            if not news_cards:
                news_cards = soup.find_all('div', class_='card-with-cluster')
            if not news_cards:
                news_cards = soup.find_all('div', class_=['newsitem', 'cardcommon'])
                
            print(f"Found {len(news_cards)} article elements on Bing page {page}")
            
            # Extract URLs
            page_urls = []
            for card in news_cards:
                try:
                    # Extract link
                    link_element = card.find('a')
                    if not link_element or 'href' not in link_element.attrs:
                        continue
                        
                    url = link_element['href']
                    
                    # Skip if already processed or invalid
                    if not url or not url.startswith('http') or url in processed_urls:
                        continue
                    
                    page_urls.append(url)
                    processed_urls.add(url)
                    
                except Exception as e:
                    print(f"Error extracting URL from Bing: {e}")
                    continue
            
            all_article_urls.extend(page_urls)
            
            # If no URLs found on this page and it's not page 1, stop
            if len(page_urls) == 0 and page > 1:
                break
        
        except Exception as e:
            print(f"Error during Bing News extraction (page {page}): {e}")
            time.sleep(random.uniform(5, 10))
    
    # Process the gathered URLs
    if all_article_urls:
        # Add small delays between article processing
        modified_headers = headers.copy()
        modified_headers['X-Add-Delay'] = 'true'
        news_articles.extend(process_urls_in_parallel(all_article_urls[:num_articles*2], modified_headers, company_name, num_articles))
    
    print(f"Extracted {len(news_articles)} articles from Bing News")
    return news_articles

def search_yahoo_news(company_name: str, num_articles: int, headers: Dict[str, str], processed_urls: Set[str], session=None) -> List[Dict[str, Any]]:
    """Search for news articles on Yahoo News"""
    news_articles = []
    all_article_urls = []
    
    # Create a session if not provided
    if session is None:
        session = requests.Session()
    
    # Search query for Yahoo News
    query = f"{company_name} bistronews"
    encoded_query = query.replace(' ', '+')
    
    # Use pagination for Yahoo too (pages 1-3)
    for page in range(1, 4):
        if len(all_article_urls) >= num_articles * 2:
            break
        
        # Yahoo uses 'b' parameter for pagination
        b_param = (page - 1) * 10 + 1 if page > 1 else 1
        url = f"https://news.search.yahoo.com/search?p={encoded_query}&b={b_param}"
        
        print(f"Searching Yahoo News page {page} for '{company_name}'")
        
        try:
            # Add delay between pages
            if page > 1:
                delay = random.uniform(4, 8)
                print(f"Waiting {delay:.1f}s before requesting Yahoo page {page}")
                time.sleep(delay)
            
            # Rotate user agent
            headers = headers.copy()
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
            ]
            headers['User-Agent'] = random.choice(user_agents)
            
            # Send request to Yahoo News
            response = session.get(url, headers=headers, timeout=15)
            
            if response.status_code != 200:
                print(f"Yahoo returned status code {response.status_code} for page {page}")
                time.sleep(random.uniform(10, 20))
                continue
            
            # Parse HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news article elements (Yahoo's structure)
            news_items = soup.find('div', id='web')
            
            if not news_items:
                print(f"No article container found on Yahoo page {page}")
                continue
                
            ol_element = news_items.find('ol')
            if not ol_element:
                print(f"No list element found on Yahoo page {page}")
                continue
                
            # Get all list items
            list_items = ol_element.find_all('li')
            print(f"Found {len(list_items)} article elements on Yahoo page {page}")
            
            # Extract URLs
            page_urls = []
            for li in list_items:
                try:
                    # Extract link
                    link_element = li.find('a')
                    if not link_element or 'href' not in link_element.attrs:
                        continue
                        
                    url = link_element['href']
                    
                    # Skip if already processed or invalid
                    if not url or not url.startswith('http') or url in processed_urls:
                        continue
                    
                    page_urls.append(url)
                    processed_urls.add(url)
                    
                except Exception as e:
                    print(f"Error extracting URL from Yahoo: {e}")
                    continue
            
            all_article_urls.extend(page_urls)
            
            # If no URLs found on this page and it's not page 1, stop
            if len(page_urls) == 0 and page > 1:
                break
        
        except Exception as e:
            print(f"Error during Yahoo News extraction (page {page}): {e}")
            time.sleep(random.uniform(5, 10))
    
    # Process the gathered URLs
    if all_article_urls:
        # Add small delays between article processing
        modified_headers = headers.copy()
        modified_headers['X-Add-Delay'] = 'true'
        news_articles.extend(process_urls_in_parallel(all_article_urls[:num_articles*2], modified_headers, company_name, num_articles))
    
    print(f"Extracted {len(news_articles)} articles from Yahoo News")
    return news_articles