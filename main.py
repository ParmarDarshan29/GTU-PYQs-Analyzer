"""
GTU PYQs Analyzer - Main Streamlit Application
Advanced question pattern analysis and study recommendations system.
"""

import streamlit as st
import time
import json
from datetime import datetime
from typing import List, Dict
from pathlib import Path
import re

# Import custom modules
from src.pdf_processor import PDFProcessor
from src.question_processor import QuestionProcessor
from src.nlp_analyzer import NLPAnalyzer
from src.llm_analyzer import LLMAnalyzer
from src.visualizer import Visualizer
from src.utils import (
    download_nltk_data, validate_api_key, create_session_id,
    export_to_json, estimate_processing_time, AnalysisCache,
    format_duration, handle_api_error, create_export_package,
    create_pdf_from_text
)
from config.config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT, MODEL_OPTIONS,
    DEFAULT_MAX_QUESTIONS, SUPPORTED_FILE_TYPES
)


def setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=LAYOUT,
        initial_sidebar_state="expanded"
    )


def load_custom_css():
    """Load custom CSS styling."""
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3em;
        margin-bottom: 0.2em;
        margin-top: 0.2em;
        font-weight: bold;
    }
    .sub-header {
        text-align: center;
        color: #A23B72;
        font-size: 1.2em;
        margin-top: 0.2em;
        margin-bottom: 1em;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1em;
        border-radius: 10px;
        margin: 0.5em 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .analysis-section {
        background-color: #ffffff;
        padding: 1.5em;
        border-radius: 10px;
        margin: 1em 0;
        border-left: 4px solid #2E86AB;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
        color: #262730;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E86AB;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


def create_sidebar():
    """Create and populate the sidebar."""
    st.sidebar.header("Configuration")
    
    # API Configuration
    api_key = st.sidebar.text_input(
        "OpenRouter API Key:",
        type="password",
        help="Get your API key from openrouter.ai"
    )
    
    # Model Selection
    st.sidebar.subheader("AI Model")
    selected_model_name = st.sidebar.selectbox(
        "Choose LLM Model:",
        options=list(MODEL_OPTIONS.keys()),
        help="Different models have varying capabilities and costs"
    )
    model_name = MODEL_OPTIONS[selected_model_name]
    # Hidden analysis options (use sensible defaults; UI controls removed)
    max_questions = DEFAULT_MAX_QUESTIONS
    enable_clustering = True
    enable_complexity = True
    enable_trends = False

    # Hidden advanced options (defaults)
    clustering_method = "K-Means"
    similarity_threshold = 0.7
    cache_results = True
    
    return {
        'api_key': api_key,
        'model_name': model_name,
        'max_questions': max_questions,
        'enable_clustering': enable_clustering,
        'enable_complexity': enable_complexity,
        'enable_trends': enable_trends,
        'clustering_method': clustering_method,
        'similarity_threshold': similarity_threshold,
        'cache_results': cache_results
    }


def validate_inputs(config: Dict, uploaded_files: List) -> bool:
    """Validate user inputs before processing."""
    errors = []
    
    # Validate API key
    if not config['api_key']:
        errors.append("API key is required")
    elif not validate_api_key(config['api_key']):
        errors.append("API key format appears invalid")
    
    # Validate files
    if not uploaded_files:
        errors.append("Please upload at least one PDF file")
    
    # Check file types
    for file in uploaded_files:
        if not file.name.lower().endswith('.pdf'):
            errors.append(f"{file.name} is not a PDF file")
    
    # Display errors
    if errors:
        for error in errors:
            st.error(error)
        return False
    
    return True


def process_files(uploaded_files: List, config: Dict):
    """Process uploaded files and extract questions."""
    pdf_processor = PDFProcessor()
    question_processor = QuestionProcessor()
    
    # Initialize containers for results
    all_questions = []
    file_metadata = []
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_steps = len(uploaded_files) + 3  # Files + 3 processing steps
    current_step = 0
    
    # Process each file
    for i, file in enumerate(uploaded_files):
        status_text.text(f"Processing {file.name}...")
        
        # Validate file
        if not pdf_processor.validate_file(file):
            st.error(f"Skipping invalid file: {file.name}")
            continue
        
        # Extract text and metadata
        try:
            text, metadata = pdf_processor.extract_text_from_pdf(file)
            
            if not text.strip():
                st.warning(f"No text extracted from {file.name}")
                continue
            
            # Segment questions
            questions = question_processor.segment_questions(text)
            
            # Apply question limit per file
            max_per_file = config['max_questions'] // len(uploaded_files)
            questions = questions[:max_per_file]
            
            # Add to collections
            all_questions.extend(questions)
            file_metadata.append(metadata)
            
            st.success(f"Extracted {len(questions)} questions from {file.name}")
            
        except Exception as e:
            error_msg = handle_api_error(e, f"while processing {file.name}")
            st.error(error_msg)
            continue
        
        current_step += 1
        progress_bar.progress(current_step / total_steps)
    
    # Apply global question limit
    all_questions = all_questions[:config['max_questions']]
    
    # Generate file summary
    file_summary = pdf_processor.get_file_summary(file_metadata)
    
    status_text.text("File processing complete!")
    progress_bar.progress(current_step / total_steps)
    
    return all_questions, file_metadata, file_summary


def perform_nlp_analysis(questions: List[Dict], config: Dict):
    """Perform NLP analysis on questions."""
    nlp_analyzer = NLPAnalyzer()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = {}
    
    # Extract keywords
    status_text.text("Extracting key topics...")
    progress_bar.progress(0.2)
    keywords = nlp_analyzer.extract_keywords(questions)
    results['keywords'] = keywords
    
    # Semantic clustering
    if config['enable_clustering'] and len(questions) > 5:
        status_text.text("Finding similar questions...")
        progress_bar.progress(0.5)
        clusters = nlp_analyzer.find_semantic_clusters(questions)
        results['clusters'] = clusters
    else:
        results['clusters'] = {}
    
    # Complexity analysis
    if config['enable_complexity']:
        status_text.text("Analyzing question complexity...")
        progress_bar.progress(0.7)
        complexity_scores = nlp_analyzer.analyze_question_complexity(questions)
        results['complexity_scores'] = complexity_scores
    else:
        results['complexity_scores'] = []
    
    # Frequency analysis
    status_text.text("Finding frequent patterns...")
    progress_bar.progress(0.9)
    frequent_questions = nlp_analyzer.find_frequent_questions(questions)
    results['frequent_questions'] = frequent_questions
    
    progress_bar.progress(1.0)
    status_text.text("NLP analysis complete!")
    
    return results


def perform_llm_analysis(questions: List[Dict], nlp_results: Dict, config: Dict):
    """Perform LLM-based analysis."""
    llm_analyzer = LLMAnalyzer(config['api_key'])
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Test API connection
    status_text.text("Testing API connection...")
    if not llm_analyzer.validate_api_key():
        st.error("API key validation failed. Please check your key.")
        return {}
    
    progress_bar.progress(0.2)
    
    # Comprehensive analysis
    status_text.text("Generating AI insights...")
    progress_bar.progress(0.4)
    
    try:
        comprehensive_analysis = llm_analyzer.comprehensive_analysis(
            questions, 
            nlp_results.get('keywords', []), 
            config['model_name']
        )
        
        progress_bar.progress(0.7)
        
        # Generate study plan
        status_text.text("Creating study recommendations...")
        study_plan = llm_analyzer.generate_study_plan(
            {
                'total_questions': len(questions),
                'frequent_topics': [kw[0] for kw in nlp_results.get('keywords', [])[:10]],
                'difficulty_distribution': {}
            },
            config['model_name']
        )
        
        progress_bar.progress(0.9)
        
        # Trend analysis (if enabled)
        trend_analysis = ""
        if config['enable_trends']:
            status_text.text("Analyzing historical trends...")
            trend_analysis = llm_analyzer.analyze_question_trends(questions, config['model_name'])
        
        progress_bar.progress(1.0)
        status_text.text("AI analysis complete!")
        
        return {
            'comprehensive_analysis': comprehensive_analysis,
            'study_plan': study_plan,
            'trend_analysis': trend_analysis
        }
        
    except Exception as e:
        error_msg = handle_api_error(e, "during LLM analysis")
        st.error(error_msg)
        return {}


def display_results(questions: List[Dict], nlp_results: Dict, llm_results: Dict, 
                   file_metadata: List[Dict], file_summary: Dict):
    """Display comprehensive analysis results, split LLM sections into separate tabs, and provide PDF export."""
    visualizer = Visualizer()

    # Overview metrics
    st.markdown("## 📊 Analysis Overview")
    visualizer.create_overview_metrics(questions, file_summary)

    # Get LLM analysis text
    analysis_text = llm_results.get('comprehensive_analysis', '')
    if not analysis_text:
        st.info("AI analysis not available. Check your API key and connection.")
        return

    # Strip a leading LLM-generated TOC block (heading-only lines) if present.
    # Strip a leading LLM-generated TOC block (heading-only lines) if present.
    # Many LLMs print a short list of section headings before the full sections,
    # which results in duplicated headings. We'll detect a block of consecutive
    # heading-like lines at the start (3+) and remove them. Matching is made
    # flexible to handle numbering, punctuation, ampersands, and minor variants.
    lines = analysis_text.splitlines()

    # Keywords to detect section headings (keeps it robust to small variations)
    heading_keywords = [
        'FREQUENCY', 'TOPIC', 'DISTRIBUTION', 'IMPORTANCE',
        'QUESTION', 'TYPES', 'PATTERNS', 'DIFFICULTY',
        'STRATEGIC', 'STUDY', 'RECOMMENDATIONS', 'EXAM', 'PREPARATION', 'INSIGHTS'
    ]

    def looks_like_heading_only(s: str) -> bool:
        if not s or len(s.strip()) == 0:
            return False
        # remove numbering like '1.' or '1)'
        s_clean = re.sub(r'^\s*\d+[\.)]\s*', '', s)
        # normalize: remove punctuation and multiple spaces
        s_norm = re.sub(r'[^A-Z0-9 ]', '', s_clean.upper()).strip()
        # consider it a heading if it contains at least one of the heading keywords
        # and is relatively short (heuristic)
        if len(s_norm) > 120:
            return False
        return any(k in s_norm for k in heading_keywords)

    # scan from the first non-empty line
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    heading_block_start = idx
    heading_block_count = 0
    j = heading_block_start
    while j < len(lines):
        if not lines[j].strip():
            # allow a single blank line inside the heading block but break if many
            # a blank line probably indicates end of TOC
            break
        if looks_like_heading_only(lines[j]):
            heading_block_count += 1
            j += 1
            continue
        # If we encounter a numbered section like '1. FREQUENCY...' treat this as
        # the real content start and stop scanning (we will remove the preceding TOC)
        if re.match(r'^\s*\d+[\.)]\s*', lines[j]):
            break
        # otherwise stop — we've hit non-heading content
        break

    # If we found a TOC-like block (3+ headings) and there's more content after it,
    # strip the heading-only block from the start of analysis_text.
    if heading_block_count >= 3 and j < len(lines):
        analysis_text = '\n'.join(lines[j:]).strip()

    # Expected sections (in order). We'll search case-insensitively in the LLM output.
    expected_sections = [
        '1. FREQUENCY ANALYSIS',
        '2. TOPIC DISTRIBUTION & IMPORTANCE',
        '3. QUESTION TYPES & PATTERNS',
        '4. DIFFICULTY ANALYSIS',
        '5. STRATEGIC STUDY RECOMMENDATIONS',
        '6. EXAM PREPARATION INSIGHTS'
    ]

    # Build mapping of found section start positions
    upper_text = analysis_text.upper()
    found = []
    for key in expected_sections:
        idx = upper_text.find(key)
        if idx >= 0:
            found.append((idx, key))
        else:
            # try without number prefix (e.g., 'FREQUENCY ANALYSIS')
            key_no_num = key.split('. ', 1)[-1]
            idx2 = upper_text.find(key_no_num)
            if idx2 >= 0:
                found.append((idx2, key))

    sections = {}
    if found:
        found.sort()
        for i, (pos, key) in enumerate(found):
            start = pos
            end = found[i+1][0] if i+1 < len(found) else len(analysis_text)
            part = analysis_text[start:end].strip()
            # remove heading line if present
            lines = part.splitlines()
            if len(lines) > 1:
                content = '\n'.join(lines[1:]).strip()
            else:
                content = ''
            sections[key] = content
    else:
        # Fallback: show whole analysis as a single section
        sections['Full Analysis'] = analysis_text

    # Create tabs for each found/expected section (preserve requested order)
    if 'Full Analysis' in sections:
        tabs = st.tabs(['AI Analysis'])
        with tabs[0]:
            st.markdown(sections['Full Analysis'])
    else:
        tab_labels = []
        tab_keys = []
        for key in expected_sections:
            if key in sections:
                label = key.split(' ', 1)[1] if key[0].isdigit() else key
                tab_labels.append(label)
                tab_keys.append(key)

        tabs = st.tabs(tab_labels)
        for i, key in enumerate(tab_keys):
            with tabs[i]:
                st.markdown(f"### {key}")
                st.markdown(sections.get(key, 'No content available for this section.'))

    # Append a dedicated Study Plan tab to the section tabs
    # (we'll render study plan and trend analysis inside that tab)
    if 'Full Analysis' in sections:
        # If full analysis only, add a second tab for study plan
        more_tabs = st.tabs(['Study Plan'])
        with more_tabs[0]:
            if llm_results.get('study_plan'):
                st.markdown("## Personalized Study Plan")
                st.markdown(llm_results['study_plan'])
            if llm_results.get('trend_analysis'):
                st.markdown("## Historical Trend Analysis")
                st.markdown(llm_results['trend_analysis'])
    else:
        # If we have section tabs, add Study Plan as an additional tab
        if 'tab_labels' in locals():
            tab_labels.append('Personalized Study Plan')
            tab_keys.append('STUDY_PLAN')
            # Re-create tabs with the new label set and render existing ones again
            tabs = st.tabs(tab_labels)
            for i, key in enumerate(tab_keys):
                with tabs[i]:
                    if key == 'STUDY_PLAN':
                        if llm_results.get('study_plan'):
                            st.markdown("## Personalized Study Plan")
                            st.markdown(llm_results['study_plan'])
                        if llm_results.get('trend_analysis'):
                            st.markdown("## Historical Trend Analysis")
                            st.markdown(llm_results['trend_analysis'])
                    else:
                        # Render section content
                        st.markdown(f"### {key}")
                        st.markdown(sections.get(key, 'No content available for this section.'))

    # Download PDF of full analysis
    full_text = f"GTU PYQs Analysis Report\n\n{analysis_text}\n\nSTUDY PLAN:\n{llm_results.get('study_plan','')}\n\nTRENDS:\n{llm_results.get('trend_analysis','')}"
    try:
        pdf_bytes = create_pdf_from_text('GTU PYQs Analysis Report', full_text)
        st.download_button(
            label="Download Analysis as PDF",
            data=pdf_bytes,
            file_name=f"gtu_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime='application/pdf'
        )
    except Exception as e:
        # If PDF generation fails, show a warning but do not offer JSON export per user preference
        st.warning(f"PDF generation failed: {str(e)}. PDF export is unavailable at the moment.")

    # Questions export removed per user request (no JSON downloads). If needed, we can add PDF export here.


def main():
    """Main application function."""
    # Setup page
    setup_page_config()
    load_custom_css()
    
    # Header
    # Show logo if available (centered and large)
    try:
        base_dir = Path(__file__).resolve().parent
        logo_path = base_dir / "download.png"
        if logo_path.exists():
            # use a slightly narrower center column to keep the logo medium-sized
            col1, col2, col3 = st.columns([2,1,2])
            with col2:
                # show a medium fixed width to avoid oversized logo
                st.image(str(logo_path), width=150)
    except Exception:
        # Fail silently if image can't be loaded
        pass

    st.markdown('<h1 class="main-header"> GTU PYQs Analyzer </h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Question Pattern Analysis & AI-Powered Study Recommendations</p>', unsafe_allow_html=True)
    
    # Initialize NLTK data
    download_nltk_data()
    
    # Initialize cache
    cache = AnalysisCache()
    
    # Create sidebar and get configuration
    config = create_sidebar()
    
    # Main content area
    st.markdown("## Upload Question Papers")
    st.info("**Tip:** Upload multiple PDF files from the same subject for comprehensive analysis. Mixed subjects may reduce analysis accuracy.")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Select Previous Year Question PDFs",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
        help="Upload PDF files containing exam questions. All files should be from the same subject for best results."
    )
    
    # Display file information
    if uploaded_files:
        st.markdown("### Uploaded Files")
        total_size = 0
        
        for i, file in enumerate(uploaded_files, 1):
            file_size = len(file.getvalue()) if hasattr(file, 'getvalue') else 0
            total_size += file_size
            
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{i}.** {file.name}")
            with col2:
                st.write(f"{file_size / 1024:.1f} KB")
            with col3:
                st.write("PDF")
        
        # Show estimated processing time
        estimated_pages = len(uploaded_files) * 10  # Rough estimate
        processing_time = estimate_processing_time(len(uploaded_files), estimated_pages)
        
        st.info(f"**Files:** {len(uploaded_files)} | **Total Size:** {total_size/1024:.1f} KB | **Est. Time:** {processing_time}")
    
    # Analysis button
    if uploaded_files and config['api_key']:
        # Check cache if enabled
        cache_key = None
        if config['cache_results']:
            cache_key = cache.get_cache_key(uploaded_files, config)
            cached_result = cache.get(cache_key)
            
            if cached_result:
                st.success("Found cached results! Loading previous analysis...")
                
                # Extract cached data
                cached_data = cached_result['result']
                questions = cached_data['questions']
                nlp_results = cached_data['nlp_results']
                llm_results = cached_data['llm_results']
                file_metadata = cached_data['file_metadata']
                file_summary = cached_data['file_summary']
                
                # Display results
                display_results(questions, nlp_results, llm_results, file_metadata, file_summary)
                
                # Option to run fresh analysis
                if st.button("Run Fresh Analysis", help="Ignore cache and run new analysis"):
                    cache.clear()
                    st.experimental_rerun()
                
                return
        
        # Main analysis button
        if st.button("Start Advanced Analysis", type="primary", help="Begin comprehensive question analysis"):
            
            # Validate inputs
            if not validate_inputs(config, uploaded_files):
                return
            
            start_time = time.time()
            
            try:
                with st.spinner("Processing your files..."):
                    
                    # Step 1: Process files
                    st.markdown("### File Processing")
                    questions, file_metadata, file_summary = process_files(uploaded_files, config)
                    
                    if not questions:
                        st.error("No questions were extracted from the uploaded files. Please check your PDFs and try again.")
                        return
                    
                    # Step 2: NLP Analysis
                    st.markdown("### NLP Analysis")
                    nlp_results = perform_nlp_analysis(questions, config)
                    
                    # Step 3: LLM Analysis
                    st.markdown("### AI Analysis")
                    llm_results = perform_llm_analysis(questions, nlp_results, config)
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    
                    # Cache results if enabled
                    if config['cache_results'] and cache_key:
                        cache_data = {
                            'questions': questions,
                            'nlp_results': nlp_results,
                            'llm_results': llm_results,
                            'file_metadata': file_metadata,
                            'file_summary': file_summary
                        }
                        cache.set(cache_key, cache_data)
                    
                    # Success message
                    st.success(f"Analysis completed in {format_duration(processing_time)}!")
                    
                    # Display results
                    display_results(questions, nlp_results, llm_results, file_metadata, file_summary)
                    
            except Exception as e:
                error_msg = handle_api_error(e, "during analysis")
                st.error(f"Analysis failed: {error_msg}")
                
                # Show debug information in expander
                with st.expander("Debug Information"):
                    st.write("**Error Type:**", type(e).__name__)
                    st.write("**Error Message:**", str(e))
                    st.write("**Configuration:**", config)
                    
                    if 'questions' in locals():
                        st.write("**Questions Extracted:**", len(questions))
                    
                st.info("Try reducing the number of files or questions, or check your API key and internet connection.")
    
    elif uploaded_files and not config['api_key']:
        st.warning("Please enter your OpenRouter API key in the sidebar to start analysis.")
        
        with st.expander("How to get an API key"):
            st.markdown("""
            1. Go to [OpenRouter.ai](https://openrouter.ai)
            2. Sign up for an account
            3. Navigate to the API section
            4. Generate a new API key
            5. Copy and paste it into the sidebar
            
            **Note:** You'll need to add credits to your OpenRouter account to use the API.
            """)
    
    elif not uploaded_files:
        # Show demo/tutorial section
        st.markdown("## How It Works")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 1. Upload
            - Upload PDF files containing GTU question papers
            - Support for multiple files from same subject
            - Automatic text extraction and preprocessing
            """)
        
        with col2:
            st.markdown("""
            ### 2. Analyze
            - AI-powered question pattern recognition
            - Semantic clustering of similar questions
            - Topic importance and frequency analysis
            """)
        
        with col3:
            st.markdown("""
            ### 3. Results
            - Interactive visualizations
            - Personalized study recommendations
            - Comprehensive analysis reports
            """)
        
        # Features showcase
        st.markdown("## Key Features")
        
        feature_col1, feature_col2 = st.columns(2)
        
        with feature_col1:
            st.markdown("""
            **Advanced Analysis:**
            - Semantic question clustering
            - Topic importance ranking
            - Question difficulty assessment
            - Historical trend analysis
            
            **Rich Visualizations:**
            - Interactive charts and graphs
            - Word clouds for key topics
            - Question distribution analysis
            - Complexity scoring visualizations
            """)
        
        with feature_col2:
            st.markdown("""
            **AI-Powered Insights:**
            - GPT-4 powered question analysis
            - Personalized study recommendations
            - Pattern recognition and prediction
            - Strategic preparation guidance
            
            **Export & Sharing:**
            - JSON data export
            - Comprehensive analysis reports
            - Question database download
            - Study plan generation
            """)
        
        # # Sample results preview
        # st.markdown("## Sample Analysis Preview")
        
        # # Create sample data for demonstration
        # sample_topics = [
        #     ("Data Structures", 0.95),
        #     ("Algorithms", 0.87),
        #     ("Database Management", 0.76),
        #     ("Computer Networks", 0.68),
        #     ("Operating Systems", 0.61)
        # ]
        
        # visualizer = Visualizer()
        
        # # Sample word cloud data
        # st.subheader("Most Important Topics (Sample)")
        # sample_df = pd.DataFrame(sample_topics, columns=['Topic', 'Importance'])
        # st.bar_chart(sample_df.set_index('Topic'))
        
        # st.info("**Note:** This is sample data. Upload your files to see real analysis results!")
    
    # Footer
    st.markdown("---")
    
    # About section
    with st.expander("About GTU PYQs Analyzer"):
        st.markdown("""
        **GTU PYQs Analyzer** is an advanced educational tool that uses artificial intelligence 
        and natural language processing to analyze previous year question papers from Gujarat Technological University.
        
        **Key Technologies:**
        - **NLP Processing**: NLTK, Scikit-learn, TF-IDF Analysis
        - **Machine Learning**: K-Means Clustering, Cosine Similarity
        - **AI Integration**: OpenRouter API with multiple LLM options
        - **Visualization**: Plotly, WordCloud, Interactive Charts
        - **PDF Processing**: PyMuPDF for robust text extraction
        
        **Created for students to:**
        - Identify frequently asked question patterns
        - Understand topic importance and priority
        - Get AI-powered study recommendations
        - Optimize exam preparation strategy
        
        **Version:** 2.0 | **Last Updated:** {datetime.now().strftime('%B %Y')}
        """)
    
    # Usage statistics (if you want to track)
    if 'session_count' not in st.session_state:
        st.session_state.session_count = 1
    else:
        st.session_state.session_count += 1


if __name__ == "__main__":
    main()