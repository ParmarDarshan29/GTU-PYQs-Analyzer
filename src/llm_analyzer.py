"""LLM integration for advanced question analysis."""

import requests
import streamlit as st
from typing import List, Dict
import time
from config.config import OPENROUTER_URL, MAX_RETRIES, API_TIMEOUT

class LLMAnalyzer:
    """Handles LLM API integration for question analysis."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = OPENROUTER_URL
        self.max_retries = MAX_RETRIES
        self.timeout = API_TIMEOUT
    
    def call_openrouter_api(self, prompt: str, model: str = "openai/gpt-4o-mini") -> str:
        """
        Make API call to OpenRouter with retry logic.
        
        Args:
            prompt: The prompt to send to the LLM
            model: Model identifier
            
        Returns:
            LLM response text
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.url, 
                    headers=headers, 
                    json=payload, 
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    st.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries}), retrying...")
                    time.sleep(2)
                else:
                    return "Error: Request timeout after multiple attempts"
                    
            except requests.exceptions.HTTPError as e:
                if attempt < self.max_retries - 1:
                    st.warning(f"HTTP error {e.response.status_code} (attempt {attempt + 1}/{self.max_retries}), retrying...")
                    time.sleep(2)
                else:
                    return f"Error: HTTP {e.response.status_code} - {e.response.text}"
                    
            except requests.RequestException as e:
                if attempt < self.max_retries - 1:
                    st.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}), retrying...")
                    time.sleep(2)
                else:
                    return f"Error after {self.max_retries} attempts: {str(e)}"
        
        return "Analysis failed after multiple attempts"
    
    def comprehensive_analysis(self, questions: List[Dict], keywords: List, model: str) -> str:
        """
        Perform comprehensive question analysis using LLM.
        
        Args:
            questions: List of question dictionaries
            keywords: List of important keywords/topics
            model: LLM model to use
            
        Returns:
            Comprehensive analysis text
        """
        # Limit questions to avoid token limits
        sample_size = min(50, len(questions))
        question_sample = questions[:sample_size]
        
        # Prepare questions text
        questions_text = "\n".join([
            f"{i+1}. {q['text'][:200]}..." if len(q['text']) > 200 else f"{i+1}. {q['text']}"
            for i, q in enumerate(question_sample)
        ])
        
        # Prepare keywords text
        keywords_text = ", ".join([kw[0] for kw in keywords[:15]]) if keywords else "No keywords extracted"
        
        prompt = f"""
You are an expert educational data analyst specializing in GTU (Gujarat Technological University) exam patterns. 
Analyze these exam questions and provide comprehensive insights for student preparation.

KEY TOPICS IDENTIFIED: {keywords_text}

QUESTIONS SAMPLE ({len(question_sample)} of {len(questions)} total):
{questions_text}

Please provide a detailed analysis with the following sections:

## 1. FREQUENCY ANALYSIS
- Identify and group semantically similar questions
- List the top 5 most repeated question patterns with estimated frequency
- Highlight questions that appear multiple times with slight variations

## 2. TOPIC DISTRIBUTION & IMPORTANCE
- Break down the main subject areas covered
- Rank topics by frequency and importance (High/Medium/Low priority)
- Identify core concepts that appear consistently

## 3. QUESTION TYPES & PATTERNS
- Categorize questions by type (MCQ, Short Answer, Long Answer, Numerical, etc.)
- Identify common question stems and patterns
- Note any specific GTU formatting or style patterns

## 4. DIFFICULTY ANALYSIS
- Estimate difficulty distribution (Easy/Medium/Hard)
- Identify which topics tend to have harder questions
- Note any progression in difficulty levels

## 5. STRATEGIC STUDY RECOMMENDATIONS
- Priority topics for focused study (based on frequency)
- Question types students should practice most
- Suggested preparation strategy and time allocation
- Important formulas, concepts, or definitions to memorize

## 6. EXAM PREPARATION INSIGHTS
- Predict likely question areas for future exams
- Common mistakes students should avoid
- Time management suggestions for exam day
- Resources or study methods recommended for each topic type

Format your response with clear headings, bullet points, and actionable insights that will help students prepare effectively.
        """
        
        return self.call_openrouter_api(prompt, model)
    
    def generate_study_plan(self, analysis_results: Dict, model: str) -> str:
        """
        Generate personalized study plan based on analysis.
        
        Args:
            analysis_results: Dictionary containing analysis results
            model: LLM model to use
            
        Returns:
            Study plan text
        """
        # Extract key information from results
        total_questions = analysis_results.get('total_questions', 0)
        frequent_topics = analysis_results.get('frequent_topics', [])
        difficulty_distribution = analysis_results.get('difficulty_distribution', {})
        
        prompt = f"""
Based on the GTU exam analysis, create a personalized 30-day study plan.

ANALYSIS SUMMARY:
- Total Questions Analyzed: {total_questions}
- Most Frequent Topics: {', '.join(frequent_topics[:10]) if frequent_topics else 'Various topics'}
- Difficulty Distribution: {difficulty_distribution}

Create a structured 30-day study plan with:

## WEEK 1-2: FOUNDATION BUILDING
- Day-by-day topic coverage
- Essential concepts to master first
- Recommended study hours per topic

## WEEK 3-4: PRACTICE & REINFORCEMENT  
- Question practice schedule
- Mock test recommendations
- Weak area identification and improvement

## WEEK 5: REVISION & EXAM PREP
- Quick revision techniques
- Last-minute preparation tips
- Exam day strategy

For each week, include:
- Daily study targets
- Practice question quotas
- Self-assessment checkpoints
- Resource recommendations

Make it practical and achievable for average students.
        """
        
        return self.call_openrouter_api(prompt, model)
    
    def analyze_question_trends(self, questions: List[Dict], model: str) -> str:
        """
        Analyze trends in question patterns over time.
        
        Args:
            questions: List of question dictionaries with year information
            model: LLM model to use
            
        Returns:
            Trend analysis text
        """
        # Extract questions with years
        questions_with_years = [q for q in questions if q.get('year')]
        
        if not questions_with_years:
            return "No year information available for trend analysis."
        
        # Group questions by year
        year_groups = {}
        for q in questions_with_years:
            year = q['year']
            if year not in year_groups:
                year_groups[year] = []
            year_groups[year].append(q['text'][:100])  # Truncate for API limits
        
        years_data = "\n\n".join([
            f"YEAR {year} ({len(questions)} questions):\n" + "\n".join([f"- {q}" for q in questions[:5]])
            for year, questions in sorted(year_groups.items())
        ])
        
        prompt = f"""
Analyze the evolution of GTU exam question patterns over the years.

QUESTIONS BY YEAR:
{years_data}

Please analyze:

## 1. TEMPORAL TRENDS
- How question styles have evolved over years
- Changes in difficulty levels
- New topics introduced or topics removed

## 2. PATTERN EVOLUTION  
- Changes in question formatting
- Evolution in answer expectations
- Shift in focus areas

## 3. PREDICTIVE INSIGHTS
- What patterns suggest about future exam trends
- Topics gaining or losing importance
- Recommended preparation adjustments

## 4. STRATEGIC IMPLICATIONS
- How students should adapt their preparation
- Which historical patterns are most reliable
- Focus areas for upcoming exams

Provide actionable insights based on historical patterns.
        """
        
        return self.call_openrouter_api(prompt, model)
    
    def validate_api_key(self) -> bool:
        """
        Validate if the provided API key works.
        
        Returns:
            True if API key is valid, False otherwise
        """
        test_prompt = "Hello, this is a test. Please respond with 'API key is working'."
        
        try:
            response = self.call_openrouter_api(test_prompt)
            return "API key is working" in response or len(response) > 10
        except:
            return False