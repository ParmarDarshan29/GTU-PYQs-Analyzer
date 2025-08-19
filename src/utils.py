"""Utility functions for the GTU PYQs Analyzer."""

import json
import hashlib
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import re
import nltk


def download_nltk_data():
    """Download required NLTK data with error handling."""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        return True
    except Exception as e:
        st.warning(f"NLTK download failed: {str(e)}. Some features may not work optimally.")
        return False


def validate_api_key(api_key: str) -> bool:
    """
    Validate API key format.
    
    Args:
        api_key: The API key to validate
        
    Returns:
        bool: True if format looks valid
    """
    if not api_key:
        return False
    
    # Basic validation - should be at least 20 characters
    if len(api_key) < 20:
        return False
    
    # Should contain alphanumeric characters and possibly dashes/underscores
    if not re.match(r'^[a-zA-Z0-9_-]+$', api_key):
        return False
    
    return True


def create_session_id() -> str:
    """
    Create a unique session ID for tracking.
    
    Returns:
        str: Unique session identifier
    """
    timestamp = datetime.now().isoformat()
    return hashlib.md5(timestamp.encode()).hexdigest()[:8]


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"


def export_to_json(data: Any, filename_prefix: str = "gtu_analysis") -> str:
    """
    Export data to JSON format.
    
    Args:
        data: Data to export
        filename_prefix: Prefix for filename
        
    Returns:
        str: JSON string
    """
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'export_type': filename_prefix,
        'data': data
    }
    
    return json.dumps(export_data, indent=2, ensure_ascii=False)


def create_download_link(data: str, filename: str, link_text: str) -> str:
    """
    Create a download link for data.
    
    Args:
        data: String data to download
        filename: Filename for download
        link_text: Text to display for link
        
    Returns:
        str: HTML download link
    """
    import base64
    
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}" style="text-decoration: none; color: #1f77b4;">{link_text}</a>'
    return href


def clean_text_for_analysis(text: str) -> str:
    """
    Clean text for better analysis.
    
    Args:
        text: Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E]', '', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[.]{3,}', '...', text)
    
    return text.strip()


def extract_numbers_from_text(text: str) -> List[int]:
    """
    Extract all numbers from text.
    
    Args:
        text: Text to extract numbers from
        
    Returns:
        List[int]: List of numbers found
    """
    numbers = re.findall(r'\b\d+\b', text)
    return [int(num) for num in numbers]


def calculate_similarity_score(text1: str, text2: str) -> float:
    """
    Calculate simple similarity score between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Simple word-based similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    Split list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers.
    
    Args:
        numerator: Numerator
        denominator: Denominator  
        default: Default value if division by zero
        
    Returns:
        float: Result of division or default
    """
    return numerator / denominator if denominator != 0 else default


def get_color_palette(n_colors: int) -> List[str]:
    """
    Generate color palette for visualizations.
    
    Args:
        n_colors: Number of colors needed
        
    Returns:
        List of color hex codes
    """
    import colorsys
    
    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    
    return colors


def format_analysis_report(analysis_data: Dict) -> str:
    """
    Format analysis data into a readable report.
    
    Args:
        analysis_data: Dictionary containing analysis results
        
    Returns:
        str: Formatted report
    """
    report_lines = [
        "# GTU PYQs Analysis Report",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        f"- Total Questions: {analysis_data.get('total_questions', 0)}",
        f"- Files Processed: {analysis_data.get('files_processed', 0)}",
        f"- Unique Topics: {len(analysis_data.get('keywords', []))}",
        f"- Question Clusters: {analysis_data.get('clusters', 0)}",
        "",
    ]
    
    # Add top topics
    keywords = analysis_data.get('keywords', [])
    if keywords:
        report_lines.extend([
            "## Top Topics",
            ""
        ])
        for i, (topic, score) in enumerate(keywords[:10], 1):
            report_lines.append(f"{i}. {topic} (Score: {score:.4f})")
        report_lines.append("")
    
    # Add complexity analysis if available
    complexity = analysis_data.get('complexity_analysis')
    if complexity:
        report_lines.extend([
            "## Complexity Analysis",
            f"- Average Complexity Score: {complexity.get('avg_complexity', 0):.2f}",
            f"- Difficulty Distribution: {complexity.get('difficulty_distribution', {})}",
            ""
        ])
    
    return "\n".join(report_lines)


def validate_question_quality(question: Dict) -> Dict[str, Any]:
    """
    Validate and score question quality.
    
    Args:
        question: Question dictionary
        
    Returns:
        Dict with quality metrics
    """
    text = question.get('text', '')
    quality_score = 0
    issues = []
    
    # Check minimum length
    if len(text) < 10:
        issues.append("Question too short")
    else:
        quality_score += 20
    
    # Check for proper sentence structure
    if '?' in text or text.endswith('.'):
        quality_score += 15
    else:
        issues.append("Missing question mark or proper ending")
    
    # Check for meaningful content
    word_count = len(text.split())
    if word_count >= 5:
        quality_score += 20
    else:
        issues.append("Too few words")
    
    # Check for numbers/technical terms (good for technical questions)
    if any(char.isdigit() for char in text):
        quality_score += 10
    
    # Check for technical keywords
    technical_keywords = ['algorithm', 'function', 'method', 'system', 'process', 'analyze', 'calculate', 'implement', 'design', 'explain']
    if any(keyword in text.lower() for keyword in technical_keywords):
        quality_score += 15
    
    # Normalize score to 0-100
    quality_score = min(100, quality_score)
    
    return {
        'quality_score': quality_score,
        'issues': issues,
        'is_valid': quality_score >= 50 and len(issues) <= 2
    }


def create_summary_statistics(questions: List[Dict]) -> Dict[str, Any]:
    """
    Create comprehensive summary statistics.
    
    Args:
        questions: List of question dictionaries
        
    Returns:
        Dict with summary statistics
    """
    if not questions:
        return {}
    
    word_counts = [q.get('word_count', 0) for q in questions]
    char_counts = [q.get('char_count', 0) for q in questions]
    
    # Calculate basic statistics
    stats = {
        'total_questions': len(questions),
        'word_count_stats': {
            'mean': sum(word_counts) / len(word_counts),
            'min': min(word_counts),
            'max': max(word_counts),
            'median': sorted(word_counts)[len(word_counts) // 2]
        },
        'char_count_stats': {
            'mean': sum(char_counts) / len(char_counts),
            'min': min(char_counts),
            'max': max(char_counts),
        }
    }
    
    # Count questions by length category
    short_questions = sum(1 for wc in word_counts if wc <= 10)
    medium_questions = sum(1 for wc in word_counts if 10 < wc <= 50)
    long_questions = sum(1 for wc in word_counts if wc > 50)
    
    stats['length_distribution'] = {
        'short': short_questions,
        'medium': medium_questions,
        'long': long_questions
    }
    
    # Count questions with years
    questions_with_years = sum(1 for q in questions if q.get('year'))
    stats['questions_with_years'] = questions_with_years
    
    # Count questions with options (MCQs)
    mcq_questions = sum(1 for q in questions if q.get('has_options'))
    stats['mcq_questions'] = mcq_questions
    
    return stats


def log_analysis_session(session_data: Dict, log_file: str = "analysis_log.json"):
    """
    Log analysis session for debugging/analytics.
    
    Args:
        session_data: Data to log
        log_file: Log file path
    """
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': create_session_id(),
            'data': session_data
        }
        
        # In a real application, you might want to save this to a file
        # For Streamlit, we'll just store in session state
        if 'analysis_log' not in st.session_state:
            st.session_state.analysis_log = []
        
        st.session_state.analysis_log.append(log_entry)
        
    except Exception as e:
        st.warning(f"Logging failed: {str(e)}")


def handle_api_error(error: Exception, context: str = "") -> str:
    """
    Handle API errors gracefully.
    
    Args:
        error: Exception that occurred
        context: Additional context about where error occurred
        
    Returns:
        str: User-friendly error message
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    # Common API error patterns
    if "timeout" in error_msg.lower():
        return f"⏱️ Request timeout. Please try again. {context}"
    elif "401" in error_msg or "unauthorized" in error_msg.lower():
        return f"🔑 API key authentication failed. Please check your key. {context}"
    elif "429" in error_msg or "rate limit" in error_msg.lower():
        return f"🚦 Rate limit exceeded. Please wait a moment before retrying. {context}"
    elif "500" in error_msg or "502" in error_msg or "503" in error_msg:
        return f"🔧 Server error. The API service may be temporarily unavailable. {context}"
    else:
        return f"❌ API Error ({error_type}): {error_msg} {context}"


def create_export_package(questions: List[Dict], analysis_results: Dict, metadata: Dict) -> Dict:
    """
    Create comprehensive export package.
    
    Args:
        questions: List of questions
        analysis_results: Analysis results
        metadata: File metadata
        
    Returns:
        Dict: Complete export package
    """
    return {
        'export_info': {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0',
            'total_questions': len(questions),
            'files_processed': len(metadata) if isinstance(metadata, list) else 1
        },
        'questions': questions,
        'analysis_results': analysis_results,
        'metadata': metadata,
        'summary_statistics': create_summary_statistics(questions)
    }


def estimate_processing_time(file_count: int, total_pages: int) -> str:
    """
    Estimate processing time based on file characteristics.
    
    Args:
        file_count: Number of files
        total_pages: Total pages across files
        
    Returns:
        str: Estimated time string
    """
    # Rough estimates based on typical processing times
    base_time = file_count * 5  # 5 seconds per file
    page_time = total_pages * 0.5  # 0.5 seconds per page
    api_time = 30  # 30 seconds for LLM analysis
    
    total_seconds = base_time + page_time + api_time
    
    if total_seconds < 60:
        return f"~{int(total_seconds)} seconds"
    else:
        minutes = int(total_seconds / 60)
        return f"~{minutes} minute{'s' if minutes > 1 else ''}"


class AnalysisCache:
    """Simple caching mechanism for analysis results."""
    
    def __init__(self):
        if 'analysis_cache' not in st.session_state:
            st.session_state.analysis_cache = {}
    
    def get_cache_key(self, files: List, settings: Dict) -> str:
        """Generate cache key based on files and settings."""
        file_info = []
        for file in files:
            if hasattr(file, 'name') and hasattr(file, 'size'):
                file_info.append(f"{file.name}_{file.size}")
        
        cache_string = f"{'_'.join(file_info)}_{json.dumps(settings, sort_keys=True)}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get(self, cache_key: str):
        """Get cached result."""
        return st.session_state.analysis_cache.get(cache_key)
    
    def set(self, cache_key: str, result: Any):
        """Set cached result."""
        st.session_state.analysis_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now(),
            'cache_key': cache_key
        }
        
        # Limit cache size to prevent memory issues
        if len(st.session_state.analysis_cache) > 10:
            # Remove oldest entries
            oldest_key = min(
                st.session_state.analysis_cache.keys(),
                key=lambda k: st.session_state.analysis_cache[k]['timestamp']
            )
            del st.session_state.analysis_cache[oldest_key]
    
    def clear(self):
        """Clear all cached results."""
        st.session_state.analysis_cache = {}


def format_duration(seconds: float) -> str:
    """
    Format duration in human readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        str: Formatted duration
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.0f}s"
    else:
        hours = int(seconds / 3600)
        remaining_minutes = int((seconds % 3600) / 60)
        return f"{hours}h {remaining_minutes}m"