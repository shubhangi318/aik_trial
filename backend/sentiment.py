from typing import Dict, List, Any, Tuple
import re
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from textblob import TextBlob
import statistics
import os
from transformers import pipeline
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from urllib.parse import urlparse

# Load FinBERT once at the start - this happens when the module is imported
print("Loading FinBERT model...")
finbert_pipeline = pipeline("text-classification", model="ProsusAI/finbert", return_all_scores=True)
print("FinBERT model loaded successfully")

# Download VADER lexicon if not already installed
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')

def preprocess_article(text: str) -> str:
    """
    Preprocess article by removing irrelevant content like ads, 
    navigation elements, and standardizing whitespace.
    """
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove extra whitespace, tabs, newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common article footers
    footers = [
        "This article was produced by", "Follow us on", "Copyright ©",
        "All rights reserved", "Terms of Service", "Privacy Policy"
    ]
    for footer in footers:
        if footer in text:
            text = text.split(footer)[0]
    
    # Remove common ad or metadata phrases
    ad_patterns = [
        r'ADVERTISEMENT', r'SPONSORED CONTENT', r'Read more:',
        r'Click here to subscribe', r'Share this article', r'Read more at', r'Also read'
    ]
    for pattern in ad_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text

def split_into_paragraphs(text: str, max_length: int = 512) -> List[str]:
    """
    Split article text into meaningful paragraphs for analysis.
    """
    # First try to split by newlines (natural paragraph breaks)
    paragraphs = [p for p in text.split('\n') if p.strip()]
    
    # If no natural paragraphs or very few, split by sentences
    if len(paragraphs) <= 1:
        sentences = nltk.sent_tokenize(text)
        paragraphs = []
        current_paragraph = ""
        
        for sentence in sentences:
            if len(current_paragraph) + len(sentence) > max_length:
                if current_paragraph:
                    paragraphs.append(current_paragraph)
                current_paragraph = sentence
            else:
                if current_paragraph:
                    current_paragraph += " " + sentence
                else:
                    current_paragraph = sentence
        
        if current_paragraph:
            paragraphs.append(current_paragraph)
    
    # Handle very long paragraphs that exceed model limits
    result_paragraphs = []
    for paragraph in paragraphs:
        if len(paragraph) > max_length:
            # Split by sentence and recombine to keep under limit
            sentences = nltk.sent_tokenize(paragraph)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > max_length:
                    result_paragraphs.append(current_chunk)
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
            
            if current_chunk:
                result_paragraphs.append(current_chunk)
        else:
            result_paragraphs.append(paragraph)
    
    return result_paragraphs

def get_finbert_sentiment(text: str) -> Dict[str, float]:
    """
    Use FinBERT to get financial sentiment scores using the pre-loaded pipeline.
    """
    # Truncate if needed (pipeline handles this, but good to be explicit)
    if len(text) > 512:
        text = text[:512]
    
    # Get sentiment using pre-loaded pipeline
    result = finbert_pipeline(text)[0]
    
    # Convert results to the expected format
    # The pipeline returns a list of dictionaries with label and score
    sentiment_scores = {
        item['label']: item['score'] for item in result
    }
    
    # Ensure we have all three keys
    for label in ['positive', 'negative', 'neutral']:
        if label not in sentiment_scores:
            sentiment_scores[label] = 0.0
    
    return sentiment_scores

def get_vader_sentiment(text: str) -> Dict[str, float]:
    """
    Get VADER sentiment scores for text.
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment

def analyze_sentiment(text: str, title: str = None) -> Dict[str, Any]:
    """
    Comprehensive sentiment analysis with hybrid approach.
    """
    # Step 1: Preprocess the article
    processed_text = preprocess_article(text)

    title_sentiment_score = 0
    if title:
        title_vader = get_vader_sentiment(title)
        title_finbert = get_finbert_sentiment(title)
        
        # Calculate title sentiment score (stronger weight on title)
        title_sentiment_score = (
            (title_finbert.get("positive", 0) - title_finbert.get("negative", 0)) * 0.6 +
            (title_vader['compound']) * 0.4
        )
    
    # Step 2: Analyze with FinBERT (financial domain model)
    finbert_scores = get_finbert_sentiment(processed_text)
    
    # Get the predominant FinBERT sentiment
    finbert_sentiment = max(finbert_scores.items(), key=lambda x: x[1])[0]
    finbert_confidence = max(finbert_scores.values())
    
    # Step 3: Break into paragraphs and analyze with VADER
    paragraphs = split_into_paragraphs(processed_text)
    
    # Analyze each paragraph with VADER
    vader_paragraph_scores = []
    for paragraph in paragraphs:
        if len(paragraph.strip()) > 20:  # Only analyze substantial paragraphs
            vader_score = get_vader_sentiment(paragraph)
            vader_paragraph_scores.append(vader_score)
    
    # Calculate aggregate VADER scores
    if vader_paragraph_scores:
        vader_compound = sum(score['compound'] for score in vader_paragraph_scores) / len(vader_paragraph_scores)
        vader_pos = sum(score['pos'] for score in vader_paragraph_scores) / len(vader_paragraph_scores)
        vader_neg = sum(score['neg'] for score in vader_paragraph_scores) / len(vader_paragraph_scores)
        vader_neu = sum(score['neu'] for score in vader_paragraph_scores) / len(vader_paragraph_scores)
    else:
        # Fallback if no substantial paragraphs
        vader_full = get_vader_sentiment(processed_text)
        vader_compound = vader_full['compound']
        vader_pos = vader_full['pos']
        vader_neg = vader_full['neg']
        vader_neu = vader_full['neu']
    
    # Determine VADER sentiment category
    # if vader_compound >= 0.05:
    #     vader_sentiment = "positive"
    # elif vader_compound <= -0.05:
    #     vader_sentiment = "negative"
    # else:
    #     vader_sentiment = "neutral"
    
    # Step 4: Determine final sentiment using a weighted approach
    # Define weights: FinBERT has domain knowledge, VADER has linguistic rules
    finbert_weight = 0.6  # Higher weight for financial domain expertise
    vader_weight = 0.4
    
    # Calculate weighted sentiment scores for each category
    weighted_scores = {
        "positive": (finbert_scores.get("positive", 0) * finbert_weight) + 
                   (vader_pos * vader_weight),
        "negative": (finbert_scores.get("negative", 0) * finbert_weight) + 
                   (vader_neg * vader_weight),
        "neutral": (finbert_scores.get("neutral", 0) * finbert_weight) + 
                  (vader_neu * vader_weight)
    }
    
    # Get final sentiment category
    final_sentiment = max(weighted_scores.items(), key=lambda x: x[1])[0]
    
    # Check for specific negative business terms
    business_negative_terms = [
        'bankruptcy', 'bankrupt', 'arbitration', 'lawsuit', 'legal battle', 
        'financial struggle', 'debt', 'litigation', 'layoffs', 'downsizing',
        'cautious', 'unsustainable', 'decline', 'struggles', 'struggling',
        'failed'
    ]

    reputation_negative_terms = [
        'criticizes', 'accuses', 'exploiting', 'fraud', 'controversy',
        'backlash', 'outrage', 'complaint', 'protest', 'scandal', 'investigation',
        'attack', 'slam', 'blame', 'condemn', 'dispute', 'anger',
        'skeptical', 'cautious about'
    ]
    
    # NEW: Add positive terms
    reputation_positive_terms = [
        'launches', 'improves', 'enhances', 'expands',
        'success', 'achievement', 'innovation', 'growth',  'award',
        'breakthrough', 'milestone', 'leadership', 'significant growth'
    ]
    
    # NEW: Apply contextual analysis for negative terms
    negative_term_found = False
    for term in business_negative_terms:
        if term in processed_text.lower():
            negative_term_found = True
            print(f"Negative financial term found: {term}")
            break
    
    # NEW: Check reputation negative terms
    for term in reputation_negative_terms:
        if term in processed_text.lower() or (title and term in title.lower()):
            negative_term_found = True
            print(f"Negative reputation term found: {term}")
            break
    
    # NEW: Check for positive terms that might override
    positive_term_found = False
    for term in reputation_positive_terms:
        # Give more weight if in title
        if title and term in title.lower():
            positive_term_found = True
            print(f"Positive term found in title: {term}")
            break
        elif term in processed_text.lower():
            # Count occurrences in text
            occurrences = len(re.findall(fr'\b{term}\b', processed_text.lower()))
            if occurrences >= 2:  # Multiple occurrences strengthen positive signal
                positive_term_found = True
                print(f"Multiple positive terms found: {term} ({occurrences} times)")
                break
    
    # Calculate initial sentiment score
    finbert_factor = (finbert_scores.get("positive", 0) - finbert_scores.get("negative", 0)) 
    vader_factor = vader_compound
    sentiment_score = (finbert_factor * finbert_weight) + (vader_factor * vader_weight)
    
    # When applying sentiment overrides, also adjust the polarity score
    if negative_term_found and not positive_term_found:
        if weighted_scores["positive"] < 0.6:
            final_sentiment = "negative"
            # ADDED: Ensure polarity is negative
            sentiment_score = -abs(sentiment_score)
    elif positive_term_found and weighted_scores["negative"] < 0.6:
        final_sentiment = "positive"
        # ADDED: Ensure polarity is positive
        sentiment_score = abs(sentiment_score)
    
    # NEW: Apply title influence if significantly different from body
    if title and abs(title_sentiment_score) > 0.5:
        # If title has strong sentiment different from current analysis
        if (title_sentiment_score > 0.5 and final_sentiment != "positive"):
            if weighted_scores["negative"] < 0.7:  # Not overwhelmingly negative
                final_sentiment = "positive"
                print(f"Sentiment adjusted to positive based on strong positive title")
        elif (title_sentiment_score < -0.5 and final_sentiment != "negative"):
            if weighted_scores["positive"] < 0.7:  # Not overwhelmingly positive
                final_sentiment = "negative"
                print(f"Sentiment adjusted to negative based on strong negative title")
    
    # After all sentiment determination logic, right before return
    if final_sentiment == "neutral":
        # Scale down polarity for neutral sentiment - keep it closer to zero
        sentiment_score = sentiment_score * 0.25  # Dampen the polarity

    # Return comprehensive results while maintaining backward compatibility
    return {
        "sentiment": final_sentiment,
        "polarity": sentiment_score,
        "vader_compound": vader_compound,  # For backward compatibility
        "vader_pos": vader_pos,
        "vader_neg": vader_neg,
        "vader_neu": vader_neu,
        "finbert_sentiment": finbert_sentiment,
    }



# def generate_sentiment_diverging_chart(articles: List[Dict[str, Any]], max_articles: int = 20) -> str:
#     if not articles:
#         print("No articles provided.")
#         return None
    
#     # Limit to max_articles most recent articles
#     articles = articles[-max_articles:] if len(articles) > max_articles else articles
    
#     # Extract article data
#     data = []
#     for article in articles:
#         url = article.get('url', 'Unknown')
#         polarity = article.get('polarity', 0)
#         sentiment = article.get('sentiment', 'neutral')
        
#         if polarity is None:
#             print(f"Missing polarity for article: {url}")
#             continue
        
#         data.append({
#             'url': url,
#             'domain': urlparse(url).netloc,
#             'polarity': polarity,
#             'sentiment': sentiment
#         })
    
#     if not data:
#         print("No valid data to plot.")
#         return None
    
#     # Create DataFrame
#     df = pd.DataFrame(data)
    
#     # Sort by polarity
#     df = df.sort_values('polarity')
    
#     # Check if DataFrame is empty after sorting
#     if df.empty:
#         print("DataFrame is empty after sorting.")
#         return None
    
#     # Set up the figure
#     fig, ax = plt.subplots(figsize=(12, max(8, len(df) * 0.4)))
    
#     # Create colors based on sentiment
#     colors = ['#B90E0A' if s == 'negative' else '#32CD32' if s == 'positive' else '#b0bce5' 
#               for s in df['sentiment']] 
    

#     # Create bars
#     y_pos = np.arange(len(df))
#     bars = ax.barh(y_pos, df['polarity'], color=colors, alpha=0.8)
    
#     # Customize x-axis
#     ax.set_xlim(-1, 1)
#     ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    
#     # Add center line
#     ax.axvline(0, color='black', alpha=0.3, ls='--', zorder=0)
    
#     # Set labels
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels(df['domain'])
#     ax.set_xlabel('Sentiment Score (-1: Highly Negative, 0: Neutral, 1: Highly Positive)')
#     ax.set_title('News Article Sentiment Analysis')
    
#     # Save figure
#     output_path = os.path.abspath("../frontend/sentiment_chart.png")
#     plt.savefig(output_path, dpi=100, bbox_inches='tight')
#     plt.close()
    
#     print(f"Chart saved to {output_path}")
#     return output_path


def compare_sentiments(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Perform comparative analysis across multiple articles.
    
    Args:
        articles: List of article dictionaries with sentiment data
        
    Returns:
        Dictionary with comparative analysis results
    """
    if not articles:
        return {
            "average_sentiment": "N/A",
            "charts": {}
        }
    
    # Extract sentiment data
    polarities = [article.get('polarity', 0) for article in articles]
    sentiments = [article.get('sentiment', 'neutral') for article in articles]
    
    # Calculate statistics
    avg_polarity = statistics.mean(polarities) if polarities else 0
    
    # Determine overall sentiment
    if avg_polarity > 0.1:
        overall_sentiment = "positive"
    elif avg_polarity < -0.1:
        overall_sentiment = "negative"
    else:
        overall_sentiment = "neutral"
    
    # Count sentiment distribution
    sentiment_count = {
        "positive": sentiments.count("positive"),
        "neutral": sentiments.count("neutral"),
        "negative": sentiments.count("negative")
    }
    
    # Generate only the news categories bubble chart
    charts = {}
    # sentiment_chart_path = generate_sentiment_diverging_chart(articles)
    # if sentiment_chart_path:
    #     charts["sentiment_diverging_chart"] = sentiment_chart_path
    
    frequency_chart_path = generate_sentiment_frequency_chart(articles)
    if frequency_chart_path:
        charts["sentiment_frequency_chart"] = frequency_chart_path
    
    return {
        "average_sentiment": overall_sentiment,
        "average_polarity": avg_polarity,
        "sentiment_distribution": sentiment_count,
        "charts": charts
    }

# def generate_sentiment_frequency_chart(articles: List[Dict[str, Any]], max_sources: int = 15) -> str:
#     """
#     Create a diverging chart showing the frequency of positive, negative, and neutral
#     sentiments for each news source, with neutral centered on a midpoint axis.
#     """
#     if not articles:
#         print("No articles provided for frequency chart.")
#         return None
    
#     # Extract source and sentiment data
#     data = []
#     for article in articles:
#         source = article.get('source', 'Unknown')
#         # Extract the domain if full URL is provided
#         if source.startswith('http'):
#             source = urlparse(source).netloc
#         # Use domain from URL if source not available
#         if source == 'Unknown' and 'url' in article:
#             source = urlparse(article['url']).netloc
            
#         sentiment = article.get('sentiment', 'neutral')
        
#         data.append({
#             'source': source,
#             'sentiment': sentiment
#         })
    
#     # Create DataFrame
#     df = pd.DataFrame(data)
    
#     # Group by source and sentiment, count occurrences
#     sentiment_counts = df.groupby(['source', 'sentiment']).size().unstack(fill_value=0)
    
#     # Ensure all sentiment columns exist
#     for sentiment in ['positive', 'negative', 'neutral']:
#         if sentiment not in sentiment_counts.columns:
#             sentiment_counts[sentiment] = 0
    
#     # Sort by total number of articles per source (descending)
#     sentiment_counts['total'] = sentiment_counts.sum(axis=1)
#     sentiment_counts = sentiment_counts.sort_values('total', ascending=False).drop('total', axis=1)
    
#     # Limit to top sources to avoid overcrowding
#     if len(sentiment_counts) > max_sources:
#         sentiment_counts = sentiment_counts.iloc[:max_sources]
    
#     # Create figure with more appropriate size
#     fig, ax = plt.subplots(figsize=(16, max(10, len(sentiment_counts) * 0.8)))
    
#     # Calculate maximum count for any sentiment type across all sources
#     max_count = sentiment_counts.values.max()
    
#     # Calculate the midpoint (this will be our center line)
#     midpoint = max_count/2
    
#     # Set up positions for y-axis
#     y_pos = np.arange(len(sentiment_counts.index))

#     def process_domain(domain):
#         parts = domain.split('.')
#         return parts[1] if len(parts) > 2 else parts[0]
    
#     # Process the index (domain names)
#     processed_domains = [process_domain(domain) for domain in sentiment_counts.index]
    
#     # Create empty bars for the legend (but make them invisible)
#     # This ensures all three categories appear in the legend
#     ax.barh(-1, 0, color='#32CD32', label='Positive')
#     ax.barh(-1, 0, color='#b0bce5', label='Neutral')
#     ax.barh(-1, 0, color='#B90E0A', label='Negative')
    
#     # For each source, plot the sentiments
#     for i, source in enumerate(sentiment_counts.index):
#         pos_count = sentiment_counts.loc[source, 'positive']
#         neg_count = sentiment_counts.loc[source, 'negative']
#         neu_count = sentiment_counts.loc[source, 'neutral']
        
#         # Calculate starting positions
#         # Neutral is centered on the midpoint
#         neutral_start = midpoint - (neu_count / 2)
#         # Negative extends left from where neutral starts
#         negative_start = neutral_start - neg_count
#         # Positive extends right from where neutral ends
#         positive_start = neutral_start + neu_count
        
#         # Plot the bars
#         if neg_count > 0:
#             neg_bar = ax.barh(i, neg_count, left=negative_start, color='#B90E0A', alpha=0.8)
#             # Add count label
#             ax.text(negative_start + neg_count/2, i, str(int(neg_count)), 
#                    ha='center', va='center', color='white', fontweight='bold')
        
#         if neu_count > 0:
#             neu_bar = ax.barh(i, neu_count, left=neutral_start, color='#b0bce5', alpha=0.8)
#             # Add count label
#             ax.text(neutral_start + neu_count/2, i, str(int(neu_count)), 
#                    ha='center', va='center', color='black', fontweight='bold')
        
#         if pos_count > 0:
#             pos_bar = ax.barh(i, pos_count, left=positive_start, color='#32CD32', alpha=0.8)
#             # Add count label
#             ax.text(positive_start + pos_count/2, i, str(int(pos_count)), 
#                    ha='center', va='center', color='white', fontweight='bold')
    
#     # Add the midpoint line - this is critical
#     ax.axvline(midpoint, color='black', alpha=0.15, ls='-', zorder=10, linewidth=1.5)
    
#     # Set axes limits with significantly more padding to prevent cutoff
#     # Calculate the total width needed (including the furthest right bar)
#     total_width_needed = midpoint + sentiment_counts['positive'].max() * 1.5
#     ax.set_xlim(0, total_width_needed)
    
#     # Set up y-axis labels
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels(processed_domains)
    
#     # Set up x-axis with appropriate tick marks
#     x_ticks = np.arange(0, total_width_needed + 1, max_count/2)
#     ax.set_xticks(x_ticks)
    
#     # Label axes
#     ax.set_xlabel('Number of Articles')
#     ax.set_title('Article Sentiment Distribution by Source', fontsize=14, pad=15)
    
#     # Add legend with all three sentiment types
#     # Place the legend outside the plot area to avoid overlapping with bars
#     ax.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99))
    
#     # Adjust layout
#     plt.tight_layout(pad=3)
    
#     # Save figure
#     output_path = os.path.abspath("../frontend/sentiment_frequency_chart.png")
#     plt.savefig(output_path, dpi=100, bbox_inches='tight')
#     plt.close()
    
#     print(f"Frequency chart saved to {output_path}")
#     return output_path


def generate_sentiment_frequency_chart(articles: List[Dict[str, Any]], max_sources: int = 15) -> str:
    """
    Create a diverging chart showing the frequency of positive, negative, and neutral
    sentiments for each news source, with neutral centered on a midpoint axis.
    """
    if not articles:
        print("No articles provided for frequency chart.")
        return None
    
    # Extract source and sentiment data
    data = []
    for article in articles:
        source = article.get('source', 'Unknown')
        # Extract the domain if full URL is provided
        if source.startswith('http'):
            source = urlparse(source).netloc
        # Use domain from URL if source not available
        if source == 'Unknown' and 'url' in article:
            source = urlparse(article['url']).netloc
            
        sentiment = article.get('sentiment', 'neutral')
        
        data.append({
            'source': source,
            'sentiment': sentiment
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Group by source and sentiment, count occurrences
    sentiment_counts = df.groupby(['source', 'sentiment']).size().unstack(fill_value=0)
    
    # Ensure all sentiment columns exist
    for sentiment in ['positive', 'negative', 'neutral']:
        if sentiment not in sentiment_counts.columns:
            sentiment_counts[sentiment] = 0
    
    # Sort by total number of articles per source (descending)
    sentiment_counts['total'] = sentiment_counts.sum(axis=1)
    sentiment_counts = sentiment_counts.sort_values('total', ascending=False).drop('total', axis=1)
    
    # Limit to top sources to avoid overcrowding
    if len(sentiment_counts) > max_sources:
        sentiment_counts = sentiment_counts.iloc[:max_sources]
    
    # Process domain names for cleaner display
    def process_domain(domain):
        parts = domain.split('.')
        return parts[1] if len(parts) > 2 else parts[0]
    
    # Process the index (domain names)
    processed_domains = [process_domain(domain) for domain in sentiment_counts.index]
    
    # Create figure with more appropriate size and tight aspect ratio
    # Use height based on number of sources with less padding
    fig_height = max(5, len(sentiment_counts) * 0.6)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    # Calculate maximum count for any sentiment type across all sources
    max_count = sentiment_counts.values.max()
    
    # Calculate the midpoint (this will be our center line)
    # Use a smaller scale factor to prevent bars from appearing too stretched
    scale_factor = 3.0  # This will compress the x-axis scale
    midpoint = max_count / scale_factor
    
    # Set up positions for y-axis with reduced spacing
    y_pos = np.arange(len(sentiment_counts.index))
    
    # Create empty bars for the legend
    ax.barh(-1, 0, color='#32CD32', label='Positive')
    ax.barh(-1, 0, color='#b0bce5', label='Neutral')
    ax.barh(-1, 0, color='#B90E0A', label='Negative')
    
    # For each source, plot the sentiments
    for i, source in enumerate(sentiment_counts.index):
        pos_count = sentiment_counts.loc[source, 'positive']
        neg_count = sentiment_counts.loc[source, 'negative']
        neu_count = sentiment_counts.loc[source, 'neutral']
        
        # Calculate starting positions
        # Neutral is centered on the midpoint
        neutral_start = midpoint - (neu_count / (scale_factor * 2))
        # Negative extends left from where neutral starts
        negative_start = neutral_start - (neg_count / scale_factor)
        # Positive extends right from where neutral ends
        positive_start = neutral_start + (neu_count / scale_factor)
        
        # Plot the bars
        if neg_count > 0:
            neg_bar = ax.barh(i, neg_count/scale_factor, left=negative_start, color='#B90E0A', alpha=0.8)
            # Add count label
            ax.text(negative_start + (neg_count/(scale_factor*2)), i, str(int(neg_count)), 
                   ha='center', va='center', color='white', fontweight='bold')
        
        if neu_count > 0:
            neu_bar = ax.barh(i, neu_count/scale_factor, left=neutral_start, color='#b0bce5', alpha=0.8)
            # Add count label
            ax.text(neutral_start + (neu_count/(scale_factor*2)), i, str(int(neu_count)), 
                   ha='center', va='center', color='black', fontweight='bold')
        
        if pos_count > 0:
            pos_bar = ax.barh(i, pos_count/scale_factor, left=positive_start, color='#32CD32', alpha=0.8)
            # Add count label
            ax.text(positive_start + (pos_count/(scale_factor*2)), i, str(int(pos_count)), 
                   ha='center', va='center', color='white', fontweight='bold')
    
    # Add the midpoint line
    ax.axvline(midpoint, color='black', alpha=0.2, ls='-', zorder=10, linewidth=1.0)
    
    # Set axes limits with proper padding to prevent cutoff
    max_range = max(sentiment_counts['positive'].max(), 
                    sentiment_counts['negative'].max(), 
                    sentiment_counts['neutral'].max()) / scale_factor
    
    # Set x-axis limits with padding to prevent cutoff
    x_min = midpoint - max_range * 1.3
    x_max = midpoint + max_range * 1.3
    
    # Ensure we don't go below 0 on the left
    x_min = max(0, x_min)
    ax.set_xlim(x_min, x_max)
    
    # Set up y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(processed_domains)
    
    # Remove x-axis labels to reduce clutter
    ax.set_xticks([])
    ax.set_xlabel('')
    
    # Set title
    ax.set_title('Count of Positive, Negative, and Neutral Articles by Source', fontsize=12, pad=10)
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Remove excess whitespace and spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Reduce whitespace
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.05)
    
    # Save figure
    output_path = os.path.abspath("../frontend/sentiment_frequency_chart.png")
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"Frequency chart saved to {output_path}")
    return output_path