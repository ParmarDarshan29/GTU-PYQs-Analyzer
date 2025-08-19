"""Question segmentation, cleaning, and preprocessing utilities."""

import re
from typing import List, Dict
from config.config import QUESTION_PATTERNS, MIN_QUESTION_LENGTH


class QuestionProcessor:
    """Handles question segmentation and preprocessing."""
    
    def __init__(self):
        self.patterns = QUESTION_PATTERNS
        self.min_length = MIN_QUESTION_LENGTH
    
    def segment_questions(self, text: str) -> List[Dict]:
        """
        Advanced question segmentation with multiple patterns.
        
        Args:
            text: Raw text extracted from PDF
            
        Returns:
            List of question dictionaries with metadata
        """
        questions = []
        
        # Try different patterns and use the one that gives the most questions
        best_segmentation = []
        
        for pattern in self.patterns:
            potential_questions = re.split(pattern, text, flags=re.IGNORECASE)
            if len(potential_questions) > len(best_segmentation):
                best_segmentation = potential_questions
        
        # Clean and filter questions
        for i, q in enumerate(best_segmentation):
            cleaned_q = q.strip()
            
            # Filter out very short or empty questions
            if len(cleaned_q) > self.min_length:
                question_data = {
                    'id': i,
                    'text': cleaned_q,
                    'word_count': len(cleaned_q.split()),
                    'char_count': len(cleaned_q),
                    'year': self._extract_year(cleaned_q),
                    'has_options': self._has_multiple_choice_options(cleaned_q),
                    'estimated_marks': self._estimate_marks(cleaned_q)
                }
                questions.append(question_data)
        
        return questions
    
    def clean_question_text(self, text: str) -> str:
        """
        Clean and normalize question text.
        
        Args:
            text: Raw question text
            
        Returns:
            Cleaned question text
        """
        # Remove question numbers and markers
        text = re.sub(r"^\d+[).]?\s*", "", text)
        text = re.sub(r"^[a-z][).]?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\([a-z0-9]+\)\s*", "", text, flags=re.IGNORECASE)
        
        # Normalize spaces and remove excessive punctuation
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s\?\.\,\;\:\!\-\(\)]", "", text)
        
        # Remove trailing periods and spaces
        text = text.strip().rstrip('.')
        
        return text
    
    def _extract_year(self, text: str) -> str:
        """Extract year from question text if present."""
        year_pattern = r"(19|20)\d{2}"
        match = re.search(year_pattern, text)
        return match.group() if match else None
    
    def _has_multiple_choice_options(self, text: str) -> bool:
        """Check if question has multiple choice options."""
        # Look for patterns like (a), (b), (c) or a), b), c)
        option_patterns = [
            r"\([a-d]\)",  # (a) (b) (c) (d)
            r"[a-d]\)",    # a) b) c) d)
            r"\b[a-d]\.?", # a. b. c. d.
        ]
        
        for pattern in option_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if len(matches) >= 3:  # At least 3 options
                return True
        
        return False
    
    def _estimate_marks(self, text: str) -> int:
        """Estimate question marks based on text patterns."""
        # Look for explicit marks mention
        marks_pattern = r"(\d+)\s*marks?"
        match = re.search(marks_pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # Estimate based on question length and complexity
        word_count = len(text.split())
        
        if word_count < 20:
            return 2  # Short answer
        elif word_count < 50:
            return 5  # Medium answer
        elif word_count < 100:
            return 10  # Long answer
        else:
            return 15  # Very long answer/essay
    
    def categorize_questions(self, questions: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Categorize questions by type and characteristics.
        
        Args:
            questions: List of question dictionaries
            
        Returns:
            Dictionary with categorized questions
        """
        categories = {
            'multiple_choice': [],
            'short_answer': [],
            'long_answer': [],
            'numerical': [],
            'theoretical': [],
            'practical': []
        }
        
        for q in questions:
            text = q['text'].lower()
            
            # Categorize based on content patterns
            if q['has_options']:
                categories['multiple_choice'].append(q)
            elif any(keyword in text for keyword in ['calculate', 'compute', 'find', 'solve']):
                categories['numerical'].append(q)
            elif any(keyword in text for keyword in ['explain', 'describe', 'discuss', 'define']):
                categories['theoretical'].append(q)
            elif any(keyword in text for keyword in ['implement', 'code', 'program', 'algorithm']):
                categories['practical'].append(q)
            elif q['word_count'] < 30:
                categories['short_answer'].append(q)
            else:
                categories['long_answer'].append(q)
        
        return categories
    
    def get_question_statistics(self, questions: List[Dict]) -> Dict:
        """
        Generate statistical summary of questions.
        
        Args:
            questions: List of question dictionaries
            
        Returns:
            Dictionary with statistics
        """
        if not questions:
            return {}
        
        word_counts = [q['word_count'] for q in questions]
        char_counts = [q['char_count'] for q in questions]
        
        # Calculate year distribution
        years = [q['year'] for q in questions if q['year']]
        year_distribution = {}
        for year in years:
            year_distribution[year] = year_distribution.get(year, 0) + 1
        
        # Calculate question type distribution
        categories = self.categorize_questions(questions)
        type_distribution = {k: len(v) for k, v in categories.items() if v}
        
        return {
            'total_questions': len(questions),
            'avg_word_count': sum(word_counts) / len(word_counts),
            'min_word_count': min(word_counts),
            'max_word_count': max(word_counts),
            'avg_char_count': sum(char_counts) / len(char_counts),
            'year_distribution': year_distribution,
            'type_distribution': type_distribution,
            'questions_with_years': len(years),
            'multiple_choice_count': sum(1 for q in questions if q['has_options'])
        }