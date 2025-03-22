import re
import json
from lxml import etree
from dateutil import parser
from openai import OpenAI

client = OpenAI()
import requests
import tempfile

import os 


from llama_index.readers.llama_parse import LlamaParse  # ✅ Another possible location
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader


from openai import OpenAI

client = OpenAI()
openai_api_key = os.getenv('OPENAI_API_KEY')


def extract_author_with_openai(article_text):
    """
    Extracts the author from article text using OpenAI's GPT API.
    
    Args:
        article_text: Full article text
        
    Returns:
        String containing author name or "Unknown"
    """

    if not openai_api_key:
        print("OPENAI_API_KEY not found in environment.")
        return "Unknown"

    try:
        response = client.chat.completions.create(model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant skilled in extracting structured data from text."},
            {"role": "user", "content": f"Extract the author's name from the following news article. "
                                        f"The author's name usually follows 'By', 'Written by', 'Published by', or 'Author:'. "
                                        f"If no author is mentioned, return 'Unknown'. Provide only the name: {article_text}"}
        ])

        author = response.choices[0].message.content.strip()

        return author if author.lower() != "unknown" else "Unknown"

    except Exception as e:
        print(f"OpenAI author extraction failed: {e}")
        return "Unknown"


def extract_author_with_llamaparse(article_text):
    """
    Extracts the author from article text using LlamaParse and LlamaIndex.
    
    Args:
        article_text: Full article text
        
    Returns:
        String containing author name or "Unknown"
    """
    try:
        # Create a temporary file to store the article
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(article_text)

        # Get API key from environment
        llama_api_key = os.getenv('LLAMA_CLOUD_API_KEY')

        # Initialize LlamaParse and parse the file
        parser = LlamaParse(api_key=llama_api_key, result_type="markdown")
        file_extractor = {".txt": parser}
        documents = SimpleDirectoryReader(input_files=[temp_file_path], file_extractor=file_extractor).load_data()

        # Create an index for querying
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()

        # Query the document for author extraction
        query = (
            "Identify the author's name from this news article. "
            "The author's name is typically found in a byline, following phrases like 'By', 'Written by', 'Published by', or 'Author:'."
            "It may also appear at the beginning of the article, near the title, or at the end. "
            "If no clear author is mentioned, return exactly 'Unknown'. "
            "Provide only the author's name, nothing else.")    
        response = query_engine.query(query)

        author = str(response).strip()

        # Clean up the extracted author name
        author = re.sub(r'^(by|author|written by|published by)\s+', '', author, flags=re.IGNORECASE)
        author = author.strip()

        # Validate author (not too long, not Unknown)
        if author and len(author.split()) < 5 and author.lower() != "unknown":
            return author

        # If LlamaParse didn't find an author, try OpenAI
        if author.lower() == "unknown":
            print("LlamaParse returned Unknown, trying OpenAI fallback...")
            openai_author = extract_author_with_openai(article_text)
            return openai_author  # Return whatever OpenAI found

        return author

    except Exception as e:
        print(f"LlamaParse author extraction failed: {e}")

        # Try OpenAI as fallback when LlamaParse fails completely
        print("LlamaParse failed, trying OpenAI fallback...")
        return extract_author_with_openai(article_text)

    finally:
        # Clean up the temporary file
        try:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        except:
            pass  # Ignore cleanup errors

def extract_author(soup, article_content):
    author = "Unknown"

    # Check common author elements
    author_patterns = [
        soup.find(['a', 'span', 'div', 'p'], attrs={'class': lambda c: c and any(author_term in c.lower()
                                                                                for author_term in ['author', 'byline', 'writer', 'creator'])}),
        soup.find('meta', attrs={'property': 'article:author'}),
        soup.find('meta', attrs={'name': 'author'})
    ]

    for pattern in author_patterns:
        if pattern:
            text = pattern.get('content', '').strip() if pattern.name == 'meta' else pattern.text.strip()
            # Filter out "Written by" or unwanted phrases
            text = re.sub(r'(?i)written by\s+', '', text).strip()
            if text and len(text.split()) < 5:  # Avoid long bios
                author = text
                break

    # # Check JSON-LD (structured data)
    # json_ld = soup.find("script", type="application/ld+json")
    # if json_ld:
    #     try:
    #         data = json.loads(json_ld.string)
    #         if isinstance(data.get("author"), dict):
    #             json_author = data.get("author", {}).get("name", "").strip()
    #             if json_author and len(json_author.split()) < 5:  # Avoid long bios
    #                 author = json_author
    #     except json.JSONDecodeError:
    #         pass  # Ignore JSON parsing errors

    # Backup regex method for "By John Doe"
    # if author == "Unknown":
    #     text = soup.get_text()
    #     match = re.search(r'(?i)\bby\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', text)
    #     if match:
    #         author = match.group(1)

    # # Handle multiple authors
    # if isinstance(author, str) and ',' in author:
    #     author = [a.strip() for a in author.split(',') if len(a.split()) < 5]

    # Filter out publication names mistakenly detected as authors
    bad_authors = {"business standard", "latest entertainment", "editorial team"}
    if isinstance(author, str) and author.lower() in bad_authors:
        author = "Unknown"

    if author == "Unknown" and article_content:
        try:
            llama_author = extract_author_with_llamaparse(article_content)
            if llama_author and llama_author != "Unknown":
                author = llama_author
        except Exception as e:
            print(f"LlamaParse author extraction failed: {e}")

    return author

def extract_date_with_openai(article_text):
    """
    Extracts the publication date from article text using OpenAI's GPT API.
    
    Args:
        article_text: Full article text
        
    Returns:
        String containing date in YYYY-MM-DD format or "Unknown"
    """
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if not openai_api_key:
        print("OPENAI_API_KEY not found in environment.")
        return "Unknown"

    try:
        response = client.chat.completions.create(model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant skilled in extracting dates from text."},
            {"role": "user", "content": f"Extract the publication date from the following news article. "
                                       f"Look for date formats like 'Published on', 'Posted on', or standalone dates. "
                                       f"Return the date in YYYY-MM-DD format only. "
                                       f"If no date is found, return 'Unknown': {article_text[:4000]}"}
        ])

        date = response.choices[0].message.content.strip()

        # Validate date format or parse if possible
        if date.lower() != "unknown":
            try:
                # Try to parse and standardize the date format
                date = parser.parse(date).strftime('%Y-%m-%d')
            except (ValueError, TypeError):
                print(f"Couldn't parse date: {date}")
                return "Unknown"

        return date

    except Exception as e:
        print(f"OpenAI date extraction failed: {e}")
        return "Unknown"


def extract_date_with_llamaparse(article_text):
    """
    Extracts the publication date from article text using LlamaParse and LlamaIndex.
    
    Args:
        article_text: Full article text
        
    Returns:
        String containing date in YYYY-MM-DD format or "Unknown"
    """
    try:
        # Create a temporary file to store the article
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(article_text)

        # Get API key from environment
        llama_api_key = os.getenv('LLAMA_CLOUD_API_KEY')

        # Initialize LlamaParse and parse the file
        parser = LlamaParse(api_key=llama_api_key, result_type="markdown")
        file_extractor = {".txt": parser}
        documents = SimpleDirectoryReader(input_files=[temp_file_path], file_extractor=file_extractor).load_data()

        # Create an index for querying
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()

        # Query the document for date extraction
        query = (
            "Identify the publication date of this news article. "
            "The date is typically found near the title or at the beginning of the article, "
            "often preceded by 'Published on', 'Posted on', or similar phrases. "
            "Return the date in YYYY-MM-DD format only. "
            "If no clear date is mentioned, return exactly 'Unknown'."
        )
        response = query_engine.query(query)

        date = str(response).strip()

        print("--------------------------------")
        print(f"Extracted date from LlamaParse: {date}")
        print("--------------------------------")

        # Validate date format or parse if possible
        if date.lower() != "unknown":
            try:
                # Try to parse and standardize the date format
                date = parser.parse(date).strftime('%Y-%m-%d')
            except (ValueError, TypeError):
                print(f"Couldn't parse date: {date}")
                # If LlamaParse returns a date but it can't be parsed, try OpenAI
                openai_date = extract_date_with_openai(article_text)
                return openai_date

        # If LlamaParse didn't find a date, try OpenAI
        if date.lower() == "unknown":
            print("LlamaParse returned Unknown for date, trying OpenAI fallback...")
            openai_date = extract_date_with_openai(article_text)
            return openai_date

        return date

    except Exception as e:
        print(f"LlamaParse date extraction failed: {e}")

        # Try OpenAI as fallback when LlamaParse fails completely
        print("LlamaParse failed for date extraction, trying OpenAI fallback...")
        return extract_date_with_openai(article_text)

    finally:
        # Clean up the temporary file
        try:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        except:
            pass  # Ignore cleanup errors

def extract_date(soup, article_content):
    """Extracts the publication date from a news article using multiple methods."""

    date = "Unknown"

    ## **1. Check `<time>` Elements**
    time_tag = soup.find("time")
    if time_tag and time_tag.text:
        date = time_tag.text.strip()

    ## **2. Check `<meta>` Tags (Common Date Fields)**
    if date == "Unknown":
        meta_tags = [
            soup.find("meta", attrs={"property": "article:published_time"}),
            soup.find("meta", attrs={"property": "og:published_time"}),
            soup.find("meta", attrs={"name": "date"}),
            soup.find("meta", attrs={"name": "dc.date"}),
            soup.find("meta", attrs={"name": "dc.date.issued"}),
            soup.find("meta", attrs={"itemprop": "datePublished"}),
        ]
        for tag in meta_tags:
            if tag and tag.get("content"):
                date = tag["content"].strip()
                break  # Stop if found

    ## **3. Check HTML Elements with Date-related Classes**
    if date == "Unknown":
        date_classes = ['date', 'time', 'published', 'datetime', 'post-date', 'entry-date']
        date_element = soup.find(['span', 'div', 'p'], attrs={'class': lambda c: c and any(d in c.lower() for d in date_classes)})
        if date_element:
            date = date_element.text.strip()

    ## **4. Check JSON-LD Structured Data (if available)**
    if date == "Unknown":
        json_ld = soup.find("script", type="application/ld+json")
        if json_ld:
            try:
                data = json.loads(json_ld.string)
                if isinstance(data, list):  # Some sites use an array of JSON-LD objects
                    data = data[0]

                if "datePublished" in data:
                    date = data["datePublished"]
                elif "dateCreated" in data:
                    date = data["dateCreated"]
            except json.JSONDecodeError:
                pass  # Ignore parsing errors

    ## **5. Use XPath for Hidden Elements**
    if date == "Unknown":
        tree = etree.HTML(str(soup))
        date_xpath = tree.xpath("//*[contains(@class, 'date') or contains(@class, 'time')]//text()")
        if date_xpath:
            date = date_xpath[0].strip()

    ## **6. Backup Regex for Dates in Article Text**
    if date == "Unknown":
        text = soup.get_text()
        match = re.search(r'(\d{1,2}\s+\w+\s+\d{4})|(\d{4}-\d{2}-\d{2})', text)  # Supports formats like "12 March 2024" or "2024-03-12"
        if match:
            date = match.group(0)

    if date == "Unknown" and article_content:
        try:
            llama_date = extract_date_with_llamaparse(article_content)
            if llama_date and llama_date != "Unknown":
                date = llama_date
        except Exception as e:
            print(f"LlamaParse date extraction failed: {e}")

    ## **7. Normalize Date Format**
    if date and date != "Unknown":
        try:
            date = parser.parse(date).strftime('%Y-%m-%d')  # Convert to YYYY-MM-DD format
        except (ValueError, TypeError):
            date = "Unknown"  # If parsing fails, reset to Unknown

    return date
