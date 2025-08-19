"""Configuration settings for GTU PYQs Analyzer."""

# API Configuration
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 3
API_TIMEOUT = 60

# Model Options
MODEL_OPTIONS = {
    "GPT-4o Mini (Fast & Cost-effective)": "openai/gpt-4o-mini",
    "GPT-4 Turbo (High Quality)": "openai/gpt-4-turbo",
    "Claude Sonnet (Alternative)": "anthropic/claude-3-sonnet"
}

# Analysis Settings
DEFAULT_MAX_QUESTIONS = 200
MIN_QUESTION_LENGTH = 10
DEFAULT_CLUSTERS = 5
MAX_KEYWORDS = 50
TOP_KEYWORDS_DISPLAY = 20

# Question Patterns
QUESTION_PATTERNS = [
    r"(?:Q\d+[\.\)]\s*)",           # Q1. Q2) etc.
    r"(?:\d+[\.\)]\s*)",            # 1. 2) etc.
    r"(?:Question\s*\d+[\.\)]\s*)", # Question 1. etc.
    r"(?:\(\d+\)\s*)",              # (1) (2) etc.
    r"(?:[a-z][\.\)]\s*)",          # a. b) etc.
]

# Visualization Settings
CHART_COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'viridis': 'viridis',
    'plasma': 'plasma'
}

# UI Settings
PAGE_TITLE = "GTU PYQs Analyzer Pro"
PAGE_ICON = "🎓"
LAYOUT = "wide"

# File Processing
SUPPORTED_FILE_TYPES = ["pdf"]
MAX_FILE_SIZE_MB = 50