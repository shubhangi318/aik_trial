from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from news_extractor import extract_company_news
from sentiment import analyze_sentiment, compare_sentiments
import os
import base64
from io import BytesIO
from gtts import gTTS
import time 
import re
import json
from openai import OpenAI

client = OpenAI()
from topic_utils import find_common_topics, get_article_specific_topics

app = FastAPI(title="Company News Analysis API")

class CompanyRequest(BaseModel):
    company_name: str
    num_articles: Optional[int] = 10
    force_refresh: Optional[bool] = False  # This can be removed if not needed

@app.post("/api/extract-news", response_model=List[Dict[str, Any]])
async def get_news(request: CompanyRequest):
    try:
        company_name = request.company_name
        num_articles = request.num_articles

        print(f"Received request to extract news for: {company_name}, articles: {num_articles}")

        # Extract new articles directly
        print(f"Extracting data for {company_name}")
        articles = extract_company_news(company_name, num_articles)

        if not articles:
            print("No articles found. Returning 404 error.")
            raise HTTPException(status_code=404, detail="No news articles found for this company")

        # Add sentiment analysis for each article
        from sentiment import analyze_sentiment

        print(f"Analyzing sentiment for {len(articles)} articles...")
        for i, article in enumerate(articles):
            if 'raw_content' in article:
                print(f"Analyzing sentiment for article {i+1}/{len(articles)}")
                sentiment_result = analyze_sentiment(
                    article['raw_content'], 
                    article.get('title', '')  # Pass the title
                )

                # Add basic sentiment classification
                article['sentiment'] = sentiment_result['sentiment']

        return articles

    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error extracting news: {str(e)}")

@app.post("/api/analyze-sentiment")
async def analyze_news_sentiment(request: CompanyRequest):
    try:
        company_name = request.company_name
        num_articles = request.num_articles

        # Extract articles directly (no database check)
        articles = extract_company_news(company_name, num_articles)
        if not articles:
            raise HTTPException(status_code=404, detail="No news articles found for this company")

        # No sentiment analysis, just ensure each article has a main_topic
        for article in articles:
            # Ensure we have a main_topic (if missing for some reason)
            if 'main_topic' not in article:
                # Default to first keyword if available
                # keywords = article.get('keywords', [])
                # article['main_topic'] = keywords[0] if keywords else "Uncategorized"
                article['main_topic'] = "Uncategorized"

        return {"articles": articles}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error analyzing articles: {str(e)}")

@app.post("/api/compare-sentiment")
async def compare_sentiment(request: CompanyRequest):
    try:
        company_name = request.company_name
        num_articles = request.num_articles

        # Extract articles directly (no database check)
        articles = extract_company_news(company_name, num_articles)
        if not articles:
            raise HTTPException(status_code=404, detail="No news articles found for this company")

        # Analyze sentiment for each article
        for article in articles:
            if 'raw_content' in article:
                print(f"Analyzing article: {article.get('title', 'Unknown')}")
                sentiment_result = analyze_sentiment(article['raw_content'])
                article.update(sentiment_result)

                # Ensure we have a main_topic (if missing for some reason)
                if 'main_topic' not in article:
                    # Default to first keyword if available
                    # keywords = article.get('keywords', [])
                    # article['main_topic'] = keywords[0] if keywords else "Uncategorized"
                    article['main_topic'] = "Uncategorized"
            else:
                print(f"Missing raw_content for article: {article.get('title', 'Unknown')}")
                article.update({
                    'sentiment': 'Unknown',
                    'polarity': 0,
                    'vader_compound': 0,
                    'main_topic': 'Uncategorized'
                })

        # Generate comparative analysis with the chart
        comparison_results = compare_sentiments(articles)

        # Return both the articles and the comparison results
        return {
            "articles": articles,
            "charts": comparison_results.get("charts", {}),
            "average_sentiment": comparison_results.get("average_sentiment", "neutral"),
            "sentiment_distribution": comparison_results.get("sentiment_distribution", {})
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")

def generate_hindi_audio_content(text: str) -> str:
    """Generate Hindi audio and return base64 encoded content"""
    try:
        audio_bytes = BytesIO()
        tts = gTTS(text=text, lang='hi', slow=False)
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)

        # Convert to base64
        audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
        return audio_base64
    except Exception as e:
        print(f"Error generating audio content: {e}")
        return None


@app.post("/api/final-analysis")
async def generate_final_analysis(request: CompanyRequest):
    try:
        company_name = request.company_name
        num_articles = request.num_articles

        # Extract articles
        articles = extract_company_news(company_name, num_articles)
        if not articles:
            raise HTTPException(status_code=404, detail="No news articles found for this company")

        # Analyze sentiment for each article
        analyzed_articles = []
        for article in articles:
            if 'raw_content' in article:
                sentiment_result = analyze_sentiment(article['raw_content'])
                article.update(sentiment_result)
            else:
                article.update({
                    'sentiment': 'Unknown',
                    'polarity': 0,
                    # 'subjectivity': 0,
                    'vader_compound': 0,
                    'speculation_score': 0,
                    # 'certainty_type': 'Unknown',
                    # 'intensity': 'Unknown'
                })
            analyzed_articles.append(article)

        # Generate comparative analysis
        comparison_results = compare_sentiments(analyzed_articles)

        # Create the final report structure
        final_report = create_final_report(company_name, analyzed_articles, comparison_results)

        # Generate Hindi TTS for the final sentiment analysis
        try:
            hindi_summary = translate_to_hindi(final_report["Final Sentiment Analysis"])
            audio_content = generate_hindi_audio_content(hindi_summary)
            final_report["AudioContent"] = audio_content  # This contains the actual audio bytes
            final_report["Audio"] = "Generated"  # Just a marker
        except Exception as e:
            print(f"Error generating Hindi TTS: {e}")
            final_report["Audio"] = "Audio generation failed"
            final_report["AudioContent"] = None

        return final_report

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating final analysis: {str(e)}")


def generate_article_comparisons(articles: List[Dict[str, Any]], company_name: str) -> List[Dict[str, Any]]:
    """
    Generate in-depth comparisons between articles using OpenAI.
    
    Args:
        articles: List of article dictionaries
        company_name: Name of the company for context
    
    Returns:
        List of comparison dictionaries with "Comparison" and "Impact" keys
    """
    if len(articles) < 2:
        return []

    try:
        # Prepare article summaries for the prompt
        article_summaries = []
        for i, article in enumerate(articles, 1):
            title = article.get('title', 'No title')
            summary = article.get('summary', 'No summary available')
            sentiment = article.get('sentiment', 'unknown')
            article_summaries.append(f"Article {i}: '{title}' - {summary} (Sentiment: {sentiment})")

        # Combine the summaries
        all_summaries = "\n\n".join(article_summaries)



        # Create the prompt for OpenAI
        prompt = f"""
        You are an experienced financial journalist specializing in sentiment analysis of news articles. Compare the following article summaries about {company_name}, 
        focusing specifically on differences in **tone, sentiment, and overall narrative**. Ensure comparisons highlight how sentiment varies across articles 
        (e.g., positive vs. negative framing, optimistic vs. skeptical outlook, risk-emphasizing vs. opportunity-driven perspectives). You can go beyond these examples as well.

        {all_summaries}
        ### **Output Guidelines**  

        - **Generate 2-3 comparison statements** that contrast sentiment differences between articles.  
        - Ensure **each comparison references article numbers** (e.g., "Article 1..., while Article 2...").  
        - Use **varied sentence structures** to avoid repetitive phrasing.  

        - **For each comparison, provide an impact analysis** explaining how these sentiment differences may influence:  
        - **Investor sentiment** (e.g., confidence, risk perception).  
        - **Stakeholder decisions** (e.g., business partnerships, customer trust).  
        - **Market perception** (e.g., brand reputation, competitive positioning).  
        - Ensure **each impact analysis is nuanced** and not generic.  

        ### **Structured JSON Output (Schema Enforced)**  

        Use the following **JSON format** to structure the response:  

        ```json
        {{
        "type": "array",
        "items": {{
            "type": "object",
            "properties": {{
            "Comparison": {{
                "type": "string",
                "description": "A concise comparison of sentiment differences between two or more articles."
            }},
            "Impact": {{
                "type": "string",
                "description": "Analysis of how the sentiment differences might influence investor sentiment, stakeholder decisions, or market perception."
            }}
            }},
            "required": ["Comparison", "Impact"]
        }}
        }}
        """

        # Make the OpenAI API call
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial analyst providing article comparisons."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=800,
        response_format = {"type": "json_object"})

        # Extract and parse the response
        result = response.choices[0].message.content.strip()


        json_match = re.search(r'(\[.*\])', result, re.DOTALL)
        if json_match:
            comparisons = json.loads(json_match.group(1))
            return comparisons
        else:
            print("Failed to parse OpenAI comparison response as JSON")
            return []

    except Exception as e:
        print(f"Error generating article comparisons with OpenAI: {e}")
        return []


def normalize_keyword(keyword: str) -> str:
    """
    Normalize a keyword to facilitate better matching.
    Removes hyphens, converts to lowercase, etc.
    
    Args:
        keyword: The original keyword string
        
    Returns:
        Normalized keyword string
    """
    # Convert to lowercase
    normalized = keyword.lower()

    # Remove hyphens and replace with spaces
    normalized = normalized.replace('-', ' ')

    # Remove any extra whitespace
    normalized = ' '.join(normalized.split())

    return normalized

def generate_comprehensive_summary(company_name: str, articles: List[Dict[str, Any]], sentiment_counts: Dict[str, int]) -> str:
    """
    Generate a comprehensive summary of all articles using OpenAI.
    This summary covers sentiment, impact, context and key points across all articles.
    
    Args:
        company_name: Name of the company
        articles: List of article dictionaries
        sentiment_counts: Dictionary with sentiment distribution counts
        
    Returns:
        String containing the comprehensive summary
    """
    try:
        # Prepare article data for the prompt
        article_data = []

        for i, article in enumerate(articles, 1):
            title = article.get('title', 'No title')
            summary = article.get('summary', 'No summary available')
            sentiment = article.get('sentiment', 'neutral')
            keywords = article.get('keywords', [])
            # main_topic = article.get('main_topic', 'Uncategorized')

            article_data.append(f"Article {i}:\n"
                              f"Title: {title}\n"
                              f"Sentiment: {sentiment}\n"
                            #   f"Main Topic: {main_topic}\n"
                              f"Keywords: {', '.join(keywords)}\n"
                              f"Summary: {summary}")

        combined_data = "\n\n".join(article_data)

        # Calculate overall sentiment distribution percentages
        total_articles = sum(sentiment_counts.values())
        sentiment_percentages = {
            key: (count / total_articles) * 100 if total_articles > 0 else 0 
            for key, count in sentiment_counts.items()
        }

        # Create the prompt for OpenAI
        prompt = f"""
        As a financial analyst and news summarizer, create a comprehensive summary paragraph about {company_name} based on these {len(articles)} news articles:
        
        {combined_data}
        
        Overall Sentiment Distribution:
        - Positive: {sentiment_percentages['Positive']:.1f}% ({sentiment_counts['Positive']} articles)
        - Negative: {sentiment_percentages['Negative']:.1f}% ({sentiment_counts['Negative']} articles)
        - Neutral: {sentiment_percentages['Neutral']:.1f}% ({sentiment_counts['Neutral']} articles)
        
        Create a single comprehensive paragraph (roughly 150-200 words) that:
        1. Summarizes the key news about {company_name} from all articles
        2. Mentions major themes, developments, or events
        3. Integrates the overall sentiment landscape 
        4. Notes potential impacts or implications for the company
        5. Provides a holistic view that covers both positive and negative aspects if present
        
        The summary should be factual, balanced, and reader-friendly, suitable for an investor or general audience. 
        Do not use bullet points or numbered lists - create a flowing narrative paragraph.
        """

        # Make the OpenAI API call
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a skilled financial journalist creating comprehensive news summaries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=400)

        # Extract the response
        comprehensive_summary = response.choices[0].message.content.strip()
        return comprehensive_summary

    except Exception as e:
        print(f"Error generating comprehensive summary with OpenAI: {e}")


def create_final_report(company_name: str, articles: List[Dict[str, Any]], comparison_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a structured final report with comparative analysis.
    """
    # Format articles for the report
    formatted_articles = []
    for article in articles:
        formatted_article = {
            "Title": article.get('title', 'No title available'),
            "Summary": article.get('summary', 'No summary available'),
            "Sentiment": article.get('sentiment', 'neutral').capitalize(),
            "Keywords": article.get('keywords', [])
        }

        formatted_article.update({
            "Main_Topic": article.get('main_topic', 'Uncategorized'),
            "Industry": article.get('industry', 'Unknown'),
            "URL": article.get('url', '#')
        })

        formatted_articles.append(formatted_article)

    # Count sentiment distribution
    sentiment_counts = {
        "Positive": sum(1 for a in articles if a.get('sentiment').lower() == 'positive'),
        "Negative": sum(1 for a in articles if a.get('sentiment').lower() == 'negative'),
        "Neutral": sum(1 for a in articles if a.get('sentiment').lower() == 'neutral')
    }

    # Generate coverage differences (pairwise comparisons of articles)
    coverage_differences = generate_article_comparisons(articles, company_name)

    article_keywords = [article.get('keywords', []) for article in articles]

    # NEW: Find common topics using sentence embeddings
    common_topics = find_common_topics(article_keywords)

    # NEW: Find article-specific topics
    article_specific_topics = get_article_specific_topics(article_keywords, common_topics)

    # Create a dictionary of unique topics per article
    unique_topics = {}
    for i, topics in enumerate(article_specific_topics):
        unique_topic_list = []
        for rep, similar in topics.items():
            unique_topic_list.append(rep)
            unique_topic_list.extend(similar)

        if unique_topic_list:
            unique_topics[f"Unique Topics in Article {i+1}"] = unique_topic_list

    # Convert common topics to a flat list
    common_topics_list = []
    for rep, similar in common_topics.items():
        common_topics_list.append(rep)
        common_topics_list.extend(similar)


    # Generate final sentiment analysis
    dominant_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0] if sentiment_counts else "Neutral"
    final_analysis = f"{company_name}'s latest news coverage is mostly {dominant_sentiment.lower()}. "

    if dominant_sentiment == "Positive":
        final_analysis += "The articles generally highlight strengths and opportunities for growth."
    elif dominant_sentiment == "Negative":
        final_analysis += "The articles generally highlight challenges and concerns facing the company."
    else:
        final_analysis += "The articles present a balanced view of the company's current situation."

    comprehensive_summary = generate_comprehensive_summary(company_name, articles, sentiment_counts)

    # Create the final report structure
    final_report = {
        "Company": company_name,
        "Articles": formatted_articles,
        "Comparative Sentiment Score": {
            "Sentiment Distribution": sentiment_counts,
            "Coverage Differences": coverage_differences,
            "Keyword Overlap": {
                "Common Keywords": common_topics_list,
                **unique_topics
            }
        },
        "Final Sentiment Analysis": comprehensive_summary,
        "Audio": "Not yet generated"
    }

    final_report["Report_Metadata"] = {
        "Generated_At": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Total_Articles_Analyzed": len(articles),
        "Dominant_Sentiment": dominant_sentiment
    }

    return final_report

def translate_to_hindi(text: str) -> str:
    """
    Translate English text to Hindi using OpenAI.
    """
    try:
        from openai import OpenAI
        
        client = OpenAI()

        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a translator. Translate the following text from English to Hindi."},
            {"role": "user", "content": text}
        ],
        temperature=0.3,
        max_tokens=500)

        hindi_text = response.choices[0].message.content.strip()
        return hindi_text
    except Exception as e:
        print(f"Error translating to Hindi: {e}")
        return text  # Return original text if translation fails

# Add a simple root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Company News Analysis API"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)