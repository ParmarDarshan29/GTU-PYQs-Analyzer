"""NLP analysis utilities for semantic processing and clustering."""

import numpy as np
import pandas as pd
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from textstat import flesch_reading_ease, flesch_kincaid_grade
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import streamlit as st
from config.config import MAX_KEYWORDS, DEFAULT_CLUSTERS


class NLPAnalyzer:
    """Advanced NLP analysis for question processing."""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = self._get_stop_words()
        self.vectorizer = None
    
    def _get_stop_words(self):
        """Get stop words with error handling."""
        try:
            return set(stopwords.words('english'))
        except LookupError:
            # Fallback to basic stop words if NLTK data not available
            return {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    
    def extract_keywords(self, questions: List[Dict]) -> List[Tuple[str, float]]:
        """
        Extract important keywords using TF-IDF.
        
        Args:
            questions: List of question dictionaries
            
        Returns:
            List of (keyword, importance_score) tuples
        """
        if not questions:
            return []
        
        try:
            # Preprocess texts
            processed_texts = []
            for q in questions:
                tokens = self._tokenize_and_clean(q['text'])
                processed_texts.append(' '.join(tokens))
            
            # TF-IDF Analysis
            self.vectorizer = TfidfVectorizer(
                max_features=MAX_KEYWORDS,
                ngram_range=(1, 3),
                stop_words='english',
                min_df=1,
                max_df=0.95
            )
            
            tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Calculate mean TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            keyword_scores = list(zip(feature_names, mean_scores))
            
            # Sort by importance and return top keywords
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            return keyword_scores
            
        except Exception as e:
            st.warning(f"Keyword extraction failed: {str(e)}")
            return []
    
    def _tokenize_and_clean(self, text: str) -> List[str]:
        """Tokenize and clean text for analysis."""
        try:
            tokens = word_tokenize(text.lower())
        except LookupError:
            # Fallback tokenization if NLTK data not available
            tokens = text.lower().split()
        
        # Clean and stem tokens
        cleaned_tokens = []
        for word in tokens:
            if (word.isalnum() and 
                word not in self.stop_words and 
                len(word) > 2):
                try:
                    stemmed = self.stemmer.stem(word)
                    cleaned_tokens.append(stemmed)
                except:
                    cleaned_tokens.append(word)
        
        return cleaned_tokens
    
    def find_semantic_clusters(self, questions: List[Dict], n_clusters: int = None) -> Dict[int, List[Dict]]:
        """
        Find semantically similar questions using clustering.
        
        Args:
            questions: List of question dictionaries
            n_clusters: Number of clusters (auto-determined if None)
            
        Returns:
            Dictionary mapping cluster_id to list of questions
        """
        if len(questions) < 2:
            return {}
        
        try:
            # Prepare texts
            texts = [self._clean_text_for_clustering(q['text']) for q in questions]
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Determine optimal number of clusters
            if n_clusters is None:
                n_clusters = min(DEFAULT_CLUSTERS, len(questions) // 2)
                n_clusters = max(2, n_clusters)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Group questions by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[label].append({
                    'question': questions[i],
                    'cluster_id': label,
                    'similarity_score': self._calculate_cluster_similarity(
                        tfidf_matrix[i], kmeans.cluster_centers_[label]
                    )
                })
            
            return dict(clusters)
            
        except Exception as e:
            st.warning(f"Clustering failed: {str(e)}")
            return {}
    
    def _clean_text_for_clustering(self, text: str) -> str:
        """Clean text specifically for clustering analysis."""
        # Remove question markers and numbers
        text = text.lower()
        text = re.sub(r'^\d+[).]?\s*', '', text)
        text = re.sub(r'^[a-z][).]?\s*', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _calculate_cluster_similarity(self, doc_vector, cluster_center) -> float:
        """Calculate similarity between document and cluster center."""
        try:
            similarity = cosine_similarity(doc_vector.reshape(1, -1), 
                                         cluster_center.reshape(1, -1))[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def analyze_question_complexity(self, questions: List[Dict]) -> List[Dict]:
        """
        Analyze question complexity and readability.
        
        Args:
            questions: List of question dictionaries
            
        Returns:
            List of complexity analysis results
        """
        complexity_scores = []
        
        for q in questions:
            text = q['text']
            
            # Skip very short texts
            if len(text) < 20:
                continue
            
            try:
                # Readability scores
                flesch_score = flesch_reading_ease(text)
                fk_grade = flesch_kincaid_grade(text)
                
                # Additional complexity metrics
                word_count = len(text.split())
                sentence_count = len([s for s in text.split('.') if s.strip()])
                avg_word_length = np.mean([len(word) for word in text.split()])
                
                # Custom complexity index
                complexity_index = (100 - flesch_score) + fk_grade + (avg_word_length * 2)
                
                complexity_scores.append({
                    'question_id': q['id'],
                    'flesch_score': round(flesch_score, 2),
                    'fk_grade': round(fk_grade, 2),
                    'word_count': word_count,
                    'sentence_count': max(1, sentence_count),
                    'avg_word_length': round(avg_word_length, 2),
                    'complexity_index': round(complexity_index, 2),
                    'difficulty_level': self._categorize_difficulty(complexity_index)
                })
                
            except Exception as e:
                # Skip questions that cause errors in readability analysis
                continue
        
        return complexity_scores
    
    def _categorize_difficulty(self, complexity_index: float) -> str:
        """Categorize difficulty based on complexity index."""
        if complexity_index < 30:
            return "Easy"
        elif complexity_index < 50:
            return "Medium"
        elif complexity_index < 70:
            return "Hard"
        else:
            return "Very Hard"
    
    def find_frequent_questions(self, questions: List[Dict], similarity_threshold: float = 0.8) -> Dict:
        """
        Find frequently occurring questions using semantic similarity.
        
        Args:
            questions: List of question dictionaries
            similarity_threshold: Threshold for considering questions similar
            
        Returns:
            Dictionary with frequency analysis
        """
        if len(questions) < 2:
            return {}
        
        try:
            # Create TF-IDF matrix
            texts = [q['text'] for q in questions]
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find similar question groups
            question_groups = []
            used_indices = set()
            
            for i in range(len(questions)):
                if i in used_indices:
                    continue
                
                # Find similar questions
                similar_indices = [i]
                for j in range(i + 1, len(questions)):
                    if j not in used_indices and similarity_matrix[i][j] > similarity_threshold:
                        similar_indices.append(j)
                        used_indices.add(j)
                
                if len(similar_indices) > 1:
                    group_questions = [questions[idx] for idx in similar_indices]
                    question_groups.append({
                        'questions': group_questions,
                        'frequency': len(similar_indices),
                        'representative': questions[i],  # First question as representative
                        'similarity_scores': [similarity_matrix[i][j] for j in similar_indices[1:]]
                    })
                
                used_indices.add(i)
            
            # Sort by frequency
            question_groups.sort(key=lambda x: x['frequency'], reverse=True)
            
            return {
                'frequent_groups': question_groups,
                'unique_questions': len(questions) - sum(group['frequency'] for group in question_groups),
                'total_groups': len(question_groups)
            }
            
        except Exception as e:
            st.warning(f"Frequency analysis failed: {str(e)}")
            return {}