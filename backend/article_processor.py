import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Any, Set
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from urllib.parse import urlparse  # Add this line
from openai import OpenAI

client = OpenAI(api_key='key')
from utils import extract_author, extract_date


def generate_article_metadata(article_content: str, company_name: str) -> Dict[str, Any]:
    """
    Generate metadata for an article using OpenAI API.
    
    Args:
        article_content: The content of the article
        company_name: The name of the company
        
    Returns:
        Dictionary with generated metadata
    """
    try:


        # Truncate content if too long (OpenAI has token limits)
        max_content_length = 4000  # Adjust based on your OpenAI model
        truncated_content = article_content[:max_content_length] if len(article_content) > max_content_length else article_content

        prompt = f"""
                You are an experienced journalist and news analyst specializing in corporate and financial reporting. Your task is to analyze the following news article about {company_name} and extract key insights in a structured format. Ensure accuracy, relevance, and completeness.

                ### **Required Analysis:**

                1. **Comprehensive Summary (at least 10 lines):**  
                - Provide a **detailed** summary capturing the **key points, context, and implications** of the article.  
                - Highlight **significant events, company announcements, industry trends, or market impacts**.  
                - If applicable, include any **financial data, product updates, executive changes, or regulatory issues** that affect {company_name}.  

                2. **Key Topics & Keywords (at least 6 items):**  
                - Extract **six** important topics, themes, or keywords that define the article's content.  
                - Focus on **company-specific aspects, industry trends, and major news elements**.  
                - **Do NOT include '{company_name}' as a keyword.**  

                3. **Relevance to {company_name} (High, Medium, Low):**  
                - Assess how **directly and significantly** the article pertains to {company_name}.  
                - Consider factors such as **direct mentions, business impact, strategic importance, and industry context**.  

                4. **Industry Classification:**  
                - Identify the industry **{company_name} belongs to**, choosing from the following standard categories:  
                    **Energy, Materials, Industrials, Consumer Discretionary, Consumer Staples, Healthcare, Financials, Information Technology, Communication Services, Utilities, Real Estate.**  

                5. **Estimated Read Time:**  
                - Calculate the **approximate read time** based on an average reading speed of **200 words per minute**.  

                6. **Main Topic Categorization:**  
                - Identify the **primary focus** of the article from the following **news categories**:  
                    **Financial Results, Mergers & Acquisitions, Stock Market, Corporate Strategy, Layoffs & Restructuring, Competitor & Market Trends, Regulatory & Legal Issues, Technology & Innovation, Sustainability & ESG, Executive Leadership, Employee Culture, Product Launch, Crisis Management, Government & Geopolitics.**  
                - Select the **best fit** based on the core subject of the article.  

                ### **Article Content:**  
                {truncated_content}

                ### **Output Format (JSON):**  
                Respond in **structured JSON** format as follows:  

                ```json
                {{
                    "summary": "Detailed summary of the article",
                    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5", "keyword6"],
                    "relevance": "High/Medium/Low",
                    "industry": "Industry name",
                    "read_time": "X min read",
                    "main_topic": "Primary news category"
        }}"""
        # # Create prompt for OpenAI with new fields
        # prompt = f"""
        # You are an experienced journalist with expertise in analyzing news articles. Analyze the following news article about {company_name} and extract detailed insights, ensuring accuracy and relevance. Provide the following structured analysis:
        # 1. Comprehensive Summary (at least 10 lines): Provide a detailed summary capturing the key points, context, and implications discussed in the article. Highlight significant events, announcements, trends, or any notable impacts on {company_name}.
        # 2. Key Topics & Keywords (at least 6 items): Identify six key topics, themes, or keywords that best represent the core subject matter of the article.
        # 3. Relevance to {company_name} (High, Medium, Low): Assess how directly and significantly the article pertains to {company_name}, considering factors such as direct mentions, business impact, or strategic implications.
        # 4. Industry Classification: Identify which industry {company_name} belongs to from the following categories: Energy, Materials, Industrials, Consumer Discretionary, Consumer Staples, Healthcare, Financials, Information Technology, Communication Services, Utilities, or Real Estate.
        # 5. Estimated Read Time: Calculate the approximate read time based on an average reading speed of 200 words per minute.
        # 6. Main Topic: Identify the single main topic or category that best represents the article's primary focus (e.g., Financial Results, Product Launch, Leadership Change, Regulatory Issue, Market Expansion, Industry Trend, etc.)

        # Article content:
        # {truncated_content}

        # Format your response as JSON with the following structure:
        # {{
        #     "summary": "Brief summary of the article",
        #     "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5", "keyword6"],
        #     "relevance": "high/medium/low",
        #     "industry": "Industry name",
        #     "read_time": "X min read",
        #     "main_topic": "Primary topic or category"
        # }}
        # """

        response = client.chat.completions.create(model="gpt-3.5-turbo",  # or another appropriate model
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500)

        # Extract and parse the response
        result = response.choices[0].message.content.strip()

        # Find JSON in the response
        json_match = re.search(r'({.*})', result, re.DOTALL)
        if json_match:
            import json
            metadata = json.loads(json_match.group(1))
            return metadata
        else:
            print("Failed to parse OpenAI response as JSON")
            return {}

    except Exception as e:
        print(f"Error generating metadata with OpenAI: {e}")
        return {}

def process_urls_in_parallel(urls: List[str], headers: Dict[str, str], company_name: str, max_articles: int) -> List[Dict[str, Any]]:
    """
    Process multiple URLs in parallel using ThreadPoolExecutor
    
    Args:
        urls: List of article URLs to process
        headers: HTTP headers for requests
        company_name: Name of the company for metadata generation
        max_articles: Maximum number of articles to return
        
    Returns:
        List of extracted article dictionaries
    """
    raw_articles = []
    article_count = 0
    total_urls = len(urls)

    # Check if we should add delays between processing
    add_delay = headers.get('X-Add-Delay') == 'true'

    print(f"Processing {total_urls} URLs in parallel" + (" with delays" if add_delay else ""))
    # Add a global variable to track progress for Streamlit
    global extraction_progress
    extraction_progress = {"current": 0, "total": max_articles, "status": "Searching for articles..."}

    # STEP 1: Extract content from URLs in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:  # Adjusted to 5 for better balance
        # Submit all tasks (but don't generate metadata yet)
        future_to_url = {}

        # Submit tasks with controlled delays if requested
        for i, url in enumerate(urls):
            # Add a small delay between submissions if requested
            if add_delay and i > 0:
                time.sleep(random.uniform(0.5, 1.5))

            future = executor.submit(extract_article_content_only, i, url, headers)
            future_to_url[future] = url

        # Process results as they complete
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                article_data = future.result()

                # Skip invalid articles
                if not article_data or not article_data.get('content'):
                    continue

                # Handle paywalled content
                if article_data.get('is_paywall', False):
                    print(f"Article at {url} is behind a paywall. Trying proreader.io...")
                    proreader_url = f"https://proreader.io/search?url={url}"
                    time.sleep(5)  # Reduced wait time
                    proreader_data = extract_article_from_proreader(proreader_url, headers)

                    if proreader_data and proreader_data.get('content'):
                        article_data = proreader_data
                    else:
                        print(f"ProReader also failed for {url}. Skipping this article.")
                        continue

                raw_articles.append(article_data)
                article_count += 1

                # Update progress for Streamlit
                extraction_progress["current"] = article_count
                extraction_progress["status"] = f"Extracted {article_count}/{max_articles} articles"

                print(f"Successfully extracted content from {article_count}/{max_articles}: {article_data.get('title', 'Unknown')}")

                # Stop if we have enough articles
                if article_count >= max_articles:
                    extraction_progress["status"] = "Extraction complete. Generating metadata..."
                    # Cancel remaining futures to stop processing
                    for f in [fut for fut in future_to_url.keys() if not fut.done()]:
                        f.cancel()
                    break

                # Add delay after successful extraction if requested
                if add_delay:
                    time.sleep(random.uniform(1.0, 2.0))

            except Exception as e:
                print(f"Error processing {url}: {e}")

    # STEP 2: Generate metadata for all articles in parallel
    if raw_articles:
        enriched_articles = generate_metadata_in_parallel(raw_articles, company_name)
        extraction_progress["status"] = f"Articles ready: {len(enriched_articles)}"
        return enriched_articles

    return []

def process_single_url(i: int, url: str, headers: Dict[str, str], company_name: str) -> Dict[str, Any]:
    """
    Process a single URL to extract article content and metadata.
    Note: This function is kept for backward compatibility, but the parallel workflow is preferred.
    """
    try:
        # Extract content
        article_data = extract_article_content_only(i, url, headers)

        # Check if the article is accessible
        if article_data.get('is_paywall', True):
            print(f"Article at {url} is behind a paywall. Trying proreader.io...")
            proreader_url = f"https://proreader.io/search?url={url}"
            time.sleep(15)  # Wait for 15 seconds before scraping
            article_data = extract_article_from_proreader(proreader_url, headers)

        # If no content is available, skip this article
        if not article_data or not article_data.get('content'):
            print(f"No content available for {url}. Skipping this article.")
            return {}

        # Generate metadata
        metadata = generate_article_metadata(article_data['content'], company_name)

        # Combine all data
        article_info = {
            'url': url,
            'raw_content': article_data.get('content', 'No content available'),
            'title': metadata.get('title', article_data.get('title', 'No title available')),
            'summary': metadata.get('summary', 'No summary available'),
            'source': article_data.get('source', 'Unknown'),
            'date': article_data.get('date', 'Unknown'),
            'keywords': metadata.get('keywords', []),
            'relevance': metadata.get('relevance', 'Unknown'),
            'industry': metadata.get('industry', 'Unknown'),
            'author': article_data.get('author', 'Unknown'),
            'read_time': metadata.get('read_time', 'Unknown'),
        }

        return article_info

    except Exception as e:
        print(f"Error in process_single_url for {url}: {e}")
        return {}


def extract_article_from_proreader(url: str, headers: Dict[str, str]) -> Dict[str, Any]:
    """Extract article content from proreader.io"""
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the content in the div with id 'root'
        content_div = soup.find('div', id='root')

        # Check if content_div is None
        if content_div is None:
            print(f"ProReader did not return any content div with id 'root' for {url}")
            return {}

        # # Debug output - use str() to get the string representation
        # print(f"ProReader content div found with length: {len(str(content_div))}")

        # Extract title
        title_element = content_div.find(['h1', 'h2'])
        title = title_element.get_text(strip=True) if title_element else "No title available"

        # Extract source from URL
        original_url = url.split('url=')[-1] if 'url=' in url else url
        # Extract source from URL
        original_url = url.split('url=')[-1] if 'url=' in url else url
        parsed_url = urlparse(original_url)
        source = parsed_url.netloc.replace('www.', '')

        # Extract author
        author = "Unknown"
        content = content_div.get_text(strip=True)

        # Extract author with content for LlamaParse fallback
        author = extract_author(soup, content)
        date = extract_date(soup, content)


        # Get content text
        content = content_div.get_text(strip=True)

        print(f"Successfully extracted from ProReader")

        return {
            'content': content,
            'title': title,
            'source': source,
            'date': date,
            'author': author,
            'url': original_url
        }

    except Exception as e:
        print(f"Error extracting content from ProReader: {e}")
        import traceback
        traceback.print_exc()  # Print the full traceback for debugging
        return {}


def extract_article_content(i, url: str, headers: Dict[str, str]) -> Dict[str, Any]:
    """
    Extract content from a news article URL.
    
    Args:
        url: The URL of the article
        headers: HTTP headers for the request
        
    Returns:
        Dictionary with article content and metadata or empty dict if access denied
    """
    try:
        # Enhance headers to appear more like a legitimate browser
        enhanced_headers = headers.copy()
        enhanced_headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        })

        # Add a referer if possible
        parsed_url = urlparse(url)
        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        enhanced_headers['Referer'] = domain

        # Rotate user agents to avoid detection
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        ]
        enhanced_headers['User-Agent'] = random.choice(user_agents)

        session = requests.Session()

        # Try accessing the article once
        try:
            response = session.get(url, headers=enhanced_headers, timeout=15)
            status_code = response.status_code

            # Print status code for debugging
            print(f"Url : {url} - Status Code: {status_code}")

            # If not successful, return empty dict with status info
            if response.status_code != 200:
                print(f"Access failed. Status Code: {response.status_code}")

                # Determine if it might be a paywall
                if response.status_code in [403, 451, 401, 429]:
                    return {'is_paywall': True, 'status_code': response.status_code}

                return {'is_paywall': False, 'status_code': response.status_code}

        except requests.exceptions.RequestException as e:
            print(f"Failed to access {url}. Error: {str(e)}")
            return {'is_paywall': False, 'error': str(e)}

        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        paywall_indicators = [
            'premium article', 'sign up', 'sign in', 'member', 'membership',
            'continue reading', 'limited access', 'unlock'
        ]

        if any(indicator in soup.text.lower() for indicator in paywall_indicators):
            return {'is_paywall': True, 'status_code': status_code}
        # Check for paywall indicators in the content

        # Extract title (improved fallback)
        title = "No title available"
        if soup.title:
            title = soup.title.text.strip()
        elif soup.find('h1'):
            title = soup.find('h1').text.strip()
        elif soup.find('h2'):
            title = soup.find('h2').text.strip()

        # Extract source from domain
        source = url.split('/')[2] if len(url.split('/')) > 2 else "Unknown"

        # Try to find date in common formats
        


        # Extract main content
        # Strategy 1: Look for article tag
        content = ""
        article_tag = soup.find('article')

        if article_tag:
            # Get all paragraphs within the article
            paragraphs = article_tag.find_all('p')
            content = ' '.join([p.text.strip() for p in paragraphs])

        # Strategy 2: Look for common content containers
        if not content:
            content_containers = soup.find_all(['div', 'section'], class_=lambda c: c and any(
                content_term in c.lower() for content_term in ['content', 'article', 'story', 'body', 'text', 'main']
            ))

            for container in content_containers:
                paragraphs = container.find_all('p')
                if paragraphs:
                    content = ' '.join([p.text.strip() for p in paragraphs])
                    break


        # Strategy 3: Just get all paragraphs if nothing else worked
        if not content or len(content) < 500:
            # Exclude navigation, footer, sidebar elements
            for nav in soup.find_all(['nav', 'footer', 'aside']):
                nav.decompose()
            #get all the text from the soup
            content = soup.get_text(strip=True)
            #ensure that the content is valid english and not corrupted
        # Clean up content
        content = re.sub(r'\s+', ' ', content).strip()


        if not is_valid_content(content):
            print(f"Invalid content detected. Skipping this article.")
            return {'is_paywall': False, 'status_code': status_code}

        author = "Unknown"
        author = extract_author(soup, content)

        date = "Unknown"
        date = extract_date(soup, content)

        # Check if content appears to be truncated (potential paywall indicator)
        paywall_phrases = ['subscribe to continue reading', 'to continue reading', 'register to continue', 
                          'create an account', 'create a free account', 'sign up to read more',
                          'subscribe for full access', 'subscribers can read']

        content_lower = content.lower()
        if any(phrase in content_lower for phrase in paywall_phrases):
            print(f"Paywall indicator detected in content for {url}")
            return {'is_paywall': True, 'status_code': status_code}

        return {
            'url': str(url),
            'title': title,
            'source': source,
            'date': date,
            'author': author,  # Added author
            'content': content,
            'is_paywall': False,
            'status_code': status_code
        }

    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return {'is_paywall': False, 'error': str(e)}  # Return error info¸


def is_valid_content(content: str) -> bool:
    """
    Check if the content is valid text and not just random corrupted characters.
    
    Args:
        content: The extracted text content
        
    Returns:
        Boolean indicating if content is valid
    """
    # Check if content is empty or too short
    if not content or len(content) < 100:
        print("Content is too short.")
        return False

    # Check for excessive special characters
    special_chars = sum(1 for c in content if c in '!@#$%^&*()_+={}[]|\\:;"<>?~`')
    if len(content) > 0 and special_chars / len(content) > 0.15:  # More than 15% special chars
        print("Content contains too many special characters.")
        return False

    # Check for proper sentence structure (at least some periods)
    if len(content) > 300 and content.count('.') < 3:
        print("Content lacks proper sentence structure.")
        return False

    # Check for character distribution (detect encoding issues)
    # Normal text should have a reasonable distribution of characters
    letter_count = sum(c.isalpha() for c in content)
    space_count = sum(c.isspace() for c in content)
    digit_count = sum(c.isdigit() for c in content)

    # Valid text should be mostly letters and spaces
    if len(content) > 0:
        text_ratio = (letter_count + space_count) / len(content)
        if text_ratio < 0.7:  # Less than 70% letters and spaces
            print("Content has unusual character distribution.")
            return False

    # Check for repeated patterns (often a sign of corrupted content)
    # If the same character repeats too many times, it's likely corrupted
    for i in range(len(content) - 10):
        if content[i:i+10] == content[i] * 10:
            print("Content contains suspicious repeated patterns.")
            return False

    # Look for nonsensical character sequences (common in encoding issues)
    suspicious_sequences = ['����', '■■■', '□□□', '###', '\u0000', '\uFFFD']
    if any(seq in content for seq in suspicious_sequences):
        print("Content contains suspicious character sequences.")
        return False

    # Verify the content has a reasonable mix of consonants and vowels
    # (valid text in English should have vowels)
    vowels = sum(c.lower() in 'aeiou' for c in content if c.isalpha())
    if letter_count > 0 and vowels / letter_count < 0.1:  # Less than 10% vowels
        print("Content has unusual vowel/consonant ratio.")
        return False

    # Check if content has a reasonable word length distribution
    words = content.split()
    if words:
        avg_word_length = sum(len(word) for word in words) / len(words)
        if avg_word_length < 2 or avg_word_length > 15:
            print(f"Content has unusual average word length: {avg_word_length:.2f}")
            return False

    return True

def generate_metadata_in_parallel(articles: List[Dict[str, Any]], company_name: str) -> List[Dict[str, Any]]:
    """
    Generate metadata for multiple articles in parallel using ThreadPoolExecutor.
    
    Args:
        articles: List of article dictionaries with content
        company_name: Name of the company for metadata generation
        
    Returns:
        List of articles with metadata added
    """
    print(f"Generating metadata for {len(articles)} articles in parallel...")

    def process_article_metadata(article):
        """Process metadata for a single article"""
        try:
            content = article.get('content', '')
            if content:
                metadata = generate_article_metadata(content, company_name)
                article.update({
                    'summary': metadata.get('summary', 'No summary available'),
                    'keywords': metadata.get('keywords', []),
                    'relevance': metadata.get('relevance', 'Unknown'),
                    'industry': metadata.get('industry', 'Unknown'),
                    'read_time': metadata.get('read_time', 'Unknown'),
                    'main_topic': metadata.get('main_topic', 'Uncategorized'),  # Add main_topic here!
                    'raw_content': content  # Rename content to raw_content for consistency
                })
                # Print for debugging
                print(f"Article '{article.get('title', 'Untitled')[:30]}...' assigned main topic: {article.get('main_topic', 'None')}")

                # Remove the original content key to avoid duplication
                if 'content' in article:
                    del article['content']
                return article
            return article
        except Exception as e:
            print(f"Error generating metadata for article: {e}")
            # Return article without metadata
            article.update({
                'summary': 'Error generating summary',
                'keywords': [],
                'relevance': 'Unknown',
                'industry': 'Unknown',
                'read_time': 'Unknown',
                'main_topic': 'Uncategorized',  # Add default main_topic here too
                'raw_content': article.get('content', 'No content available')
            })
            if 'content' in article:
                del article['content']
            return article
    # Process all articles in parallel
    enriched_articles = []
    with ThreadPoolExecutor(max_workers=5) as executor:  # Limit to 5 concurrent API calls
        # Submit all tasks
        future_to_article = {
            executor.submit(process_article_metadata, article): i 
            for i, article in enumerate(articles)
        }

        # Process results as they complete
        for future in as_completed(future_to_article):
            try:
                enriched_article = future.result()
                if enriched_article:
                    enriched_articles.append(enriched_article)
                    print(f"Successfully generated metadata for article {future_to_article[future] + 1}/{len(articles)}")
            except Exception as e:
                print(f"Error processing article metadata: {e}")

    return enriched_articles

def extract_article_content_only(i, url: str, headers: Dict[str, str]) -> Dict[str, Any]:
    """
    Extract just the content from a news article URL without generating metadata.
    This is a modified version of extract_article_content that doesn't include metadata generation.
    
    Args:
        i: Article index for debugging
        url: The URL of the article
        headers: HTTP headers for the request
        
    Returns:
        Dictionary with article content and basic info
    """
    # This is essentially the same as extract_article_content, 
    # but returns the article data without generating metadata

    try:
        # Use the existing extract_article_content function but only for content extraction
        article_data = extract_article_content(i, url, headers)

        # Return the raw data without metadata
        return article_data

    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return {'is_paywall': False, 'error': str(e)}