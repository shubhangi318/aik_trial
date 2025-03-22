import streamlit as st
import requests
import pandas as pd
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import os
from PIL import Image
import time
from pywaffle import Waffle 
import plotly.io as pio
from streamlit import components
import mpld3
from typing import List, Dict, Any
import numpy as np

import nltk
nltk.download('vader_lexicon')

import plotly.express as px
import pandas as pd

# API endpoints
API_URL = "http://localhost:8000/api"

def get_news(company_name, num_articles=10):
    response = requests.post(
        f"{API_URL}/extract-news",
        json={"company_name": company_name, "num_articles": num_articles}
    )
    return response.json()


def generate_speech(text):
    response = requests.post(
        f"{API_URL}/text-to-speech",
        json={"text": text}
    )
    return response.json()




def get_audio_player(audio_path):
    if audio_path and audio_path != "Audio generation failed":
        # If the audio path is a URL, use it directly
        if audio_path.startswith('http'):
            st.audio(audio_path)
        # If it's a local file path, read the file
        else:
            try:
                with open(audio_path, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/mp3')
            except Exception as e:
                st.error(f"Error playing audio: {e}")
    else:
        st.error("No audio available")

def get_sentiment_color(sentiment):
    """Return the color for sentiment indicators"""
    if sentiment == "positive":
        return "#32CD32"  # Green
    elif sentiment == "negative":
        return "#B90E0A"  # Red
    else:
        return "#b0bce5"  # Blue for neutral


def main():
    st.set_page_config(
        page_title="Company News Sentiment Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    # Initialize session state for storing articles
    if 'articles' not in st.session_state:
        st.session_state.articles = None
    if 'last_company' not in st.session_state:
        st.session_state.last_company = ""
    if 'last_article_count' not in st.session_state:
        st.session_state.last_article_count = 0
    
    st.title("📊 Company News Sentiment Analyzer")
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["News & Sentiment", "Comparative Analysis", "Final Report"])
    
    # Common inputs that will be used in both tabs
    company_name = st.sidebar.text_input("Enter Company Name", "Company Name")
    num_articles = st.sidebar.number_input("Number of Articles", min_value=1, max_value=30, value=10)
    
    # Check if company name or article count has changed - if so, clear session state
    if (company_name != st.session_state.last_company or 
        num_articles != st.session_state.last_article_count):
        st.session_state.articles = None
        st.session_state.last_company = company_name
        st.session_state.last_article_count = num_articles
    

    # Tab 1: News & Sentiment Analysis
    with tab1:        
        analyze_button = st.button("Extract & Analyze News")

        if analyze_button:
            with st.spinner(f"Extracting and analyzing news for {company_name}..."):
                try:
                    # Get analysis results (includes both extraction and analysis)
                    results = get_news(company_name, num_articles)
                    
                    # Now that extraction is complete, display the results
                    if isinstance(results, list) and len(results) > 0:
                        articles = results
                        # Store in session state for reuse
                        st.session_state.articles = articles
                        st.success(f"Extraction completed! Found {len(articles)} articles. Analyzing...")
                        
                        # Calculate sentiment distribution
                        sentiment_counts = {
                            "Positive": sum(1 for a in articles if a.get('sentiment').lower() == 'positive'),
                            "Neutral": sum(1 for a in articles if a.get('sentiment').lower() == 'neutral'),
                            "Negative": sum(1 for a in articles if a.get('sentiment').lower() == 'negative')
                        }
                        
                        # Filter out zero-value sentiments
                        filtered_sentiment_counts = {k: v for k, v in sentiment_counts.items() if v > 0}
                        
                        # Create a DataFrame for the pie chart - only using non-zero values
                        sentiment_df = pd.DataFrame({
                            'Sentiment': list(filtered_sentiment_counts.keys()),
                            'Count': list(filtered_sentiment_counts.values())
                        })

                        sentiment_df['Count'] = sentiment_df['Count'].astype(int)
                        
                        # NEW: Generate the final analysis report for the summary
                        try:
                            # Call the API endpoint for final analysis
                            final_report_response = requests.post(
                                f"{API_URL}/final-analysis",
                                json={"company_name": company_name, "num_articles": num_articles}
                            )
                            final_report = final_report_response.json()
                        except Exception as e:
                            st.error(f"Error generating comprehensive report: {str(e)}")
                            final_report = {"Final Sentiment Analysis": "Unable to generate comprehensive analysis."}
                        
                        # Create a two-column layout
                        left_col, right_col = st.columns([0.65, 0.35])
                        
                        # Left column: Final Analysis, Audio, and Download button
                        with left_col:
                            st.info(final_report.get("Final Sentiment Analysis", "Analysis not available"))

                            # Audio player
                            if "AudioContent" in final_report and final_report["AudioContent"]:
                                try:
                                    audio_bytes = base64.b64decode(final_report["AudioContent"])
                                    st.audio(audio_bytes, format='audio/mp3')
                                except Exception as e:
                                    st.error(f"Error playing audio: {e}")
                            else:
                                st.error("Audio not available")
                            
                            # Download JSON button
                            json_str = json.dumps(final_report, indent=2)
                            st.download_button(
                                label="Download JSON Report",
                                data=json_str,
                                file_name=f"{company_name}_analysis_report.json",
                                mime="application/json",
                                key="download_json_tab1"
                            )
                        
                        # Add this after retrieving the articles (around line 176)

                        def sort_by_date(article):
                            date = article.get('date', 'Unknown')
                            if date == 'Unknown':
                                # Return a tuple with high value to sort Unknown dates at the end
                                return (1, '0000-00-00')
                            else:
                                # Return a tuple with 0 first to sort known dates before Unknown
                                # Use the date string as second element for sorting (most recent first)
                                return (0, date)

                        # Sort the articles (most recent first, Unknown dates at the end)
                        articles.sort(key=sort_by_date, reverse=True)
                        
                        # Right column: Donut chart
                        with right_col:
                            
                            # Calculate the percentages manually
                            total = sentiment_df['Count'].sum()
                            sentiment_df['Percentage'] = (sentiment_df['Count'] / total * 100).round(1)
                            
                            # Create a custom DataFrame with repeated rows based on count
                            expanded_df = pd.DataFrame({
                                'Sentiment': np.repeat(sentiment_df['Sentiment'].values, sentiment_df['Count'].values)
                            })
                            
                            # Create donut chart with the expanded DataFrame
                            fig = px.pie(
                                expanded_df,
                                names='Sentiment',
                                color='Sentiment',
                                color_discrete_map={
                                    'Positive': '#32CD32',  # Green
                                    'Neutral': '#b0bce5',   # Blue
                                    'Negative': '#B90E0A'   # Red
                                },
                                hole=0.4,
                                # title=f"Sentiment Analysis of {len(articles)} Articles about {company_name}"
                            )
                            fig.update_layout(
                                legend_orientation="h",  # Horizontal legend
                                legend_yanchor="top",   # Changed from "bottom" to "top"
                                legend_y=-0.1,         # Negative value to move below the chart
                                legend_x=0.5,          # Center horizontally
                                legend_xanchor="center",
                                height=350, 
                                margin=dict(l=58, r=40, t=5, b=5)  # Increased bottom margin to accommodate legend
                            )

                            # Add the custom text showing actual counts and percentages
                            fig.update_traces(
                                texttemplate='%{percent:.1f}%',
                                hovertemplate='%{label}: %{count} articles (%{percent:.1f}%)<extra></extra>',
                                textinfo='percent',
                                textposition='inside',
                            )
                            
                            # Display the donut chart
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Add a horizontal divider after the two columns
                        st.markdown("---")
                        
                        # Display articles in custom card layout (existing code)
                        for i, article in enumerate(articles):
                            # Create container for the card
                            with st.container():
                                # Add sentiment badge at the top
                                sentiment = article.get('sentiment', 'neutral')
                                sentiment_color = get_sentiment_color(sentiment)
                                st.markdown(f"""
                                <div style="display: inline-block; padding: 5px 10px; background-color:{sentiment_color}; 
                                color:white; border-radius:20px; font-weight:bold; margin-bottom:10px;">
                                {sentiment.upper()}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Show the title and date in a card layout
                                st.subheader(article['title'])
        
                                # Show the title and date in a card layout
                                st.write(f"**Date:** {article.get('date', 'Unknown')}")
                                st.write(f"**Source:** {article.get('source', 'Unknown')}")
                                st.write(f"**Industry:** {article.get('industry', 'Unknown')}")
                                st.write(f"**Author:** {article.get('author', 'Unknown')}")
                                st.write(f"**Read Time:** {article.get('read_time', 'Unknown')}")
                                st.write(f"**Relevance:** {article.get('relevance', 'Unknown')}")
                                st.write(f"**Summary:** {article.get('summary', 'Unknown')}")
                                st.write(f"**Original Link:** [View Article]({article['url']})")

                                # Extract keywords and display them
                                keywords = article.get('keywords', [])
                                if keywords:
                                    keyword_html = ""
                                    for kw in keywords:
                                        # Grey background and black text
                                        keyword_html += f'<span style="background-color:#E0E0E0; color:#000000  ; padding:5px; border-radius:10px; margin-right:5px; margin-bottom:5px; display:inline-block;">{kw}</span>'
                                    st.markdown("**All Keywords:**", unsafe_allow_html=True)
                                    st.markdown(keyword_html, unsafe_allow_html=True)

                                st.markdown("---")
                    else:
                        st.error(f"No articles found for {company_name}")
                
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
    with tab2:

        st.header("Article Sentiment Comparison")
        compare_button = st.button("Generate Comparative Analysis")
        
        if compare_button:
            with st.spinner("Generating comparative visualization..."):
                try:
                    if (st.session_state.articles is not None and 
                    len(st.session_state.articles) >= num_articles and 
                    st.session_state.last_company == company_name):
                    
                        articles = st.session_state.articles
                    # Use stored articles if available and if we have enough
                    # if (st.session_state.articles and 
                    #     len(st.session_state.articles) >= num_articles and
                    #     st.session_state.last_company == company_name and
                    #     st.session_state.last_article_count == num_articles):
                        
                    #     articles = st.session_state.articles
                        st.info(f"Using {len(articles)} articles from previous extraction")
                    else:
                        # Fetch fresh articles if parameters changed or we don't have enough
                        st.info(f"Fetching {num_articles} articles for {company_name}...")
                        response = requests.post(
                            f"{API_URL}/extract-news",
                            json={"company_name": company_name, "num_articles": num_articles}
                        )
                        articles = response.json()
                        
                        # Update session state with new articles
                        st.session_state.articles = articles
                        st.session_state.last_company = company_name
                        st.session_state.last_article_count = num_articles
                        
                        st.success(f"Fetched {len(articles)} new articles")
                    
                    # Display dataframe with article info (for debugging)
                    # call a treemap here    
                    
                    # Call the API endpoint for sentiment comparison
                    response = requests.post(
                        f"{API_URL}/compare-sentiment",
                                                json={"company_name": company_name, "num_articles": num_articles}
                    )
                    comparison_results = response.json()
                    
                    # # Extract articles from the comparison results
                    articles = comparison_results.get("articles", [])
           
                    if "charts" in comparison_results and "sentiment_frequency_chart" in comparison_results["charts"]:
                        frequency_chart_path = comparison_results["charts"]["sentiment_frequency_chart"]
                        if os.path.exists(frequency_chart_path):
                            st.subheader("Article Sentiment Frequency by Source")
                            st.image(frequency_chart_path, caption="Count of Positive, Negative, and Neutral Articles by Source", use_column_width=True)
                            st.info("This chart shows the number of positive (green), negative (red), and neutral (blue) articles for each news source.")
                        else:
                            st.warning("Sentiment frequency chart image file not found.")
                    
                except Exception as e:
                    st.error(f"Error generating comparison: {str(e)}")
                    st.exception(e)

    # Tab 3: Final Report
    with tab3:
        st.header("Final Analysis Report")
     
        report_button = st.button("Generate Final Report")
        
        if report_button:
            with st.spinner("Generating analysis..."):
                try:
                    # Use stored articles if available and if we have enough
                    if (st.session_state.articles and 
                        len(st.session_state.articles) >= num_articles and
                        st.session_state.last_company == company_name and
                        st.session_state.last_article_count == num_articles):
                        
                        articles = st.session_state.articles
                        st.info(f"Using {len(articles)} articles from previous extraction")
                    else:
                        # Fetch fresh articles if parameters changed or we don't have enough
                        st.info(f"Fetching {num_articles} articles for {company_name}...")
                        response = requests.post(
                            f"{API_URL}/extract-news",
                            json={"company_name": company_name, "num_articles": num_articles}
                        )
                        articles = response.json()
                        
                        # Update session state with new articles
                        st.session_state.articles = articles
                        st.session_state.last_company = company_name
                        st.session_state.last_article_count = num_articles
                        
                        st.success(f"Fetched {len(articles)} new articles")

                    response = requests.post(
                        f"{API_URL}/compare-sentiment",
                                                json={"company_name": company_name, "num_articles": num_articles}
                    )
                    comparison_results = response.json()
                    
                    # Extract articles from the comparison results
                    articles = comparison_results.get("articles", [])
                    
                    # Count main topics and prepare data for treemap
                    topic_counts = {}
                    for article in articles:
                        main_topic = article.get('main_topic', 'Uncategorized')
                        topic_counts[main_topic] = topic_counts.get(main_topic, 0) + 1

                    # Create dataframe for the treemap - FIXED VERSION
                    topic_df = pd.DataFrame({
                        'topic': list(topic_counts.keys()),
                        'count': list(topic_counts.values())
                    })
                    
                    # Explicitly convert Count to numeric type
                    topic_df['count'] = pd.to_numeric(topic_df['count'])
                    # Streamlit App Title
                    st.title("📊 News Article Distribution by Topic")
                    # Create Treemap using Plotly
                    fig = px.treemap(topic_df, 
                                    path=["topic"], 
                                    values="count", 
                                    title="Treemap of News Topics",
                                    color="count",
                                    color_continuous_scale='Viridis')  # Dynamic color scheme

                    # Display Treemap in Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("This treemap shows the distribution of main topics across the analyzed articles. Larger blocks represent more frequent topics.")
                    

                    fig = plt.figure(
                        FigureClass=Waffle, 
                        rows=3,
                        values=topic_df['count'].tolist(),  # Use the actual count values
                        labels=topic_df['topic'].tolist(),
                        legend={'loc': 'upper left', 'bbox_to_anchor': (1.05, 1), 'prop': {'size': 5}},
                        figsize=(4, 2)   # Use the actual topic labels
                    )
                    
                    # Display the waffle chart
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error generating comparison: {str(e)}")
                    import traceback
                    st.exception(e)   


if __name__ == "__main__":
    main()