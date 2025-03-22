import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from typing import List, Dict, Any, Tuple
import re

# Initialize sentence transformer model for embeddings
model = SentenceTransformer('all-mpnet-base-v2')  # Better for business domain

def preprocess_keyword(keyword: str) -> str:
    """Clean and normalize a keyword to handle format variations."""
    # Convert to lowercase
    keyword = keyword.lower()
    
    # Replace hyphens and similar punctuation with spaces
    keyword = re.sub(r'[-–—]', ' ', keyword)  # Handle different types of hyphens
    
    # Remove other punctuation that might cause variations
    keyword = re.sub(r'[^\w\s]', '', keyword)
    
    # Normalize whitespace (including multiple spaces)
    keyword = ' '.join(keyword.split())
    
    # Sort words for consistent ordering
    # This ensures "food delivery quick" and "quick food delivery" match
    if len(keyword.split()) > 1:
        words = keyword.split()
        # Keep common phrases intact (e.g., "food delivery")
        common_phrases = ["food delivery", "quick commerce"]
        for phrase in common_phrases:
            if phrase in keyword:
                # Remove phrase words from sorting
                for word in phrase.split():
                    if word in words:
                        words.remove(word)
                # Add phrase at the beginning
                words = phrase.split() + sorted(words)
                break
        else:
            # If no common phrase found, sort all words
            words = sorted(words)
        keyword = ' '.join(words)
    
    return keyword.strip() # Minimal preprocessing to preserve meaning

def get_normalized_to_original_mapping(keywords: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Create mapping between normalized and original forms of keywords."""
    normalized_keywords = []
    normalized_to_original = {}
    
    for keyword in keywords:
        normalized = preprocess_keyword(keyword)
        normalized_keywords.append(normalized)
        
        # Keep the original form (preferring shortest if multiple originals map to same normalized)
        if normalized not in normalized_to_original or len(keyword) < len(normalized_to_original[normalized]):
            normalized_to_original[normalized] = keyword
    
    return normalized_keywords, normalized_to_original

def cluster_keywords(keywords: List[str], threshold: float = 0.3) -> Dict[str, List[str]]:
    """Cluster similar keywords using embeddings and DBSCAN clustering."""
    if not keywords or len(keywords) < 2:
        return {k: [] for k in keywords} if keywords else {}
    
    # Normalize keywords and create mapping to original forms
    normalized_keywords, normalized_to_original = get_normalized_to_original_mapping(keywords)
    
    # First, group exact matches after normalization
    unique_normalized = []
    unique_to_duplicates = {}
    
    seen = set()
    for i, norm_kw in enumerate(normalized_keywords):
        if norm_kw not in seen:
            seen.add(norm_kw)
            unique_normalized.append(norm_kw)
            unique_to_duplicates[norm_kw] = [keywords[i]]
        else:
            # Add to the duplicates list if we've seen this normalized form before
            for unique_kw in unique_normalized:
                if preprocess_keyword(unique_kw) == norm_kw:
                    unique_to_duplicates[unique_kw].append(keywords[i])
                    break
    
    # If only exact duplicates or no keywords, return early
    if len(unique_normalized) <= 1:
        if not unique_normalized:
            return {}
        representative = normalized_to_original[unique_normalized[0]]
        duplicates = [k for k in unique_to_duplicates[unique_normalized[0]] if k != representative]
        return {representative: duplicates}
    
    # Generate embeddings
    embeddings = model.encode(unique_normalized)
    
    # Cluster with DBSCAN (using cosine distance for better semantic matching)
    clustering = DBSCAN(eps=threshold, min_samples=1, metric='cosine').fit(embeddings)
    labels = clustering.labels_
    
    # Create clusters dictionary
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    # Create final output with representative keywords
    result = {}
    for label, indices in clusters.items():
        if len(indices) == 1:
            # Single keyword cluster
            norm_keyword = unique_normalized[indices[0]]
            representative = normalized_to_original[norm_keyword]
            
            # Include any duplicates from the normalization step
            duplicates = [k for k in unique_to_duplicates[norm_keyword] if k != representative]
            result[representative] = duplicates
        else:
            # Multi-keyword cluster
            # Find the most central keyword to use as representative
            centroid = np.mean([embeddings[idx] for idx in indices], axis=0)
            distances = [np.linalg.norm(embeddings[idx] - centroid) for idx in indices]
            representative_idx = indices[np.argmin(distances)]
            representative_norm = unique_normalized[representative_idx]
            representative = normalized_to_original[representative_norm]
            
            # Collect all keywords in this cluster, including duplicates
            all_similar = []
            for idx in indices:
                if idx != representative_idx:
                    norm_kw = unique_normalized[idx]
                    all_similar.extend(unique_to_duplicates[norm_kw])
            
            # Also add duplicates of the representative itself
            for dup in unique_to_duplicates[representative_norm]:
                if dup != representative:
                    all_similar.append(dup)
            
            result[representative] = all_similar
    
    return result

    #     if len(indices) == 1:
    #         # Single keyword cluster
    #         representative = keywords[indices[0]]
    #         result[representative] = []
    #     else:
    #         # Multi-keyword cluster
    #         # Find the most central keyword to use as representative
    #         centroid = np.mean([embeddings[idx] for idx in indices], axis=0)
    #         distances = [np.linalg.norm(embeddings[idx] - centroid) for idx in indices]
    #         representative_idx = indices[np.argmin(distances)]
    #         representative = keywords[representative_idx]
            
    #         similar = [keywords[idx] for idx in indices if idx != representative_idx]
    #         result[representative] = similar
    
    # return result

def find_common_topics(article_keywords: List[List[str]], min_articles: int = 2) -> Dict[str, List[str]]:
    """Find common topics across multiple articles using embeddings."""
    # Flatten all keywords
    all_keywords = []
    for keywords in article_keywords:
        all_keywords.extend(keywords)
    
    # Remove duplicates while preserving order
    unique_keywords = []
    seen = set()
    for kw in all_keywords:
        if kw.lower() not in seen:
            unique_keywords.append(kw)
            seen.add(kw.lower())
    
    # Cluster similar keywords
    clusters = cluster_keywords(unique_keywords)
    
    # Track which article each keyword appears in
    keyword_to_articles = {}
    for i, keywords in enumerate(article_keywords):
        for kw in keywords:
            if kw not in keyword_to_articles:
                keyword_to_articles[kw] = set()
            keyword_to_articles[kw].add(i)
    
    # Find common topics (clusters that appear in multiple articles)
    common_topics = {}
    for representative, similar_keywords in clusters.items():
        # Get all articles that contain any keyword in this cluster
        articles_with_topic = keyword_to_articles.get(representative, set())
        for similar in similar_keywords:
            articles_with_topic.update(keyword_to_articles[similar])
        
        # If topic appears in minimum number of articles, add to common topics
        if len(articles_with_topic) >= min_articles:
            common_topics[representative] = similar_keywords
    
    return common_topics

def get_article_specific_topics(article_keywords: List[List[str]], common_topics: Dict[str, List[str]]) -> List[Dict[str, List[str]]]:
    """Find topics unique to each article."""
    # Create a set of all common topic keywords
    all_common_keywords = set()
    for rep, similar in common_topics.items():
        all_common_keywords.add(rep.lower())
        all_common_keywords.update(kw.lower() for kw in similar)
    
    # Find unique topics for each article
    article_specific_topics = []
    
    for i, keywords in enumerate(article_keywords):
        # Cluster the article's keywords
        article_clusters = cluster_keywords(keywords)
        
        # Keep only clusters that don't overlap with common topics
        unique_clusters = {}
        for rep, similar in article_clusters.items():
            if rep.lower() not in all_common_keywords and not any(s.lower() in all_common_keywords for s in similar):
                unique_clusters[rep] = similar
        
        article_specific_topics.append(unique_clusters)
    
    return article_specific_topics