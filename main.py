"""
GTU PYQs Analyzer - Main Streamlit Application
Advanced question pattern analysis and study recommendations system.
"""

import streamlit as st
import time
import json
from datetime import datetime
from typing import List, Dict

# Import custom modules
from src.pdf_processor import PDFProcessor
from src.question_processor import QuestionProcessor
from src.nlp_analyzer import NLPAnalyzer
from src.llm_analyzer import LLMAnalyzer
from src.visualizer import Visualizer
from src.utils import (
    download_nltk_data, validate_api_key, create_session_id,
    export_to_json, estimate_processing_time, AnalysisCache,
    format_duration, handle_api_error, create_export_package
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
        margin-bottom: 0.5em;
        font-weight: bold;
    }
    .sub-header {
        text-align: center;
        color: #A23B72;
        font-size: 1.2em;
        margin-bottom: 2em;
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
    
    # Analysis Options
    st.sidebar.subheader("Analysis Options")
    
    max_questions = st.sidebar.slider(
        "Max Questions to Analyze",
        min_value=50,
        max_value=1000,
        value=DEFAULT_MAX_QUESTIONS,
        step=50,
        help="Higher values provide more comprehensive analysis but take longer"
    )
    
    enable_clustering = st.sidebar.checkbox(
        "Enable Semantic Clustering",
        value=True,
        help="Groups similar questions together"
    )
    
    enable_complexity = st.sidebar.checkbox(
        "Enable Complexity Analysis",
        value=True,
        help="Analyzes question difficulty and readability"
    )
    
    enable_trends = st.sidebar.checkbox(
        "Enable Trend Analysis",
        value=False,
        help="Analyzes patterns across years (if year data available)"
    )
    
    # Advanced Options
    with st.sidebar.expander("Advanced Options"):
        clustering_method = st.selectbox(
            "Clustering Algorithm",
            ["K-Means", "Hierarchical"],
            help="Algorithm used for grouping similar questions"
        )
        
        similarity_threshold = st.slider(
            "Similarity Threshold",
            0.1, 1.0, 0.7,
            help="How similar questions must be to group together"
        )
        
        cache_results = st.checkbox(
            "Cache Results",
            value=True,
            help="Store results to speed up repeated analysis"
        )
    
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
    """Display comprehensive analysis results."""
    visualizer = Visualizer()
    
    # Overview metrics
    st.markdown("## 📊 Analysis Overview")
    visualizer.create_overview_metrics(questions, file_summary)
    
    # Success message
    st.markdown(f"""
    <div class="success-box">
        <h4>Analysis Completed Successfully!</h4>
        <p>Processed <strong>{len(questions)}</strong> questions from <strong>{file_summary.get('total_files', 0)}</strong> files.</p>
        <p>Found <strong>{len(nlp_results.get('keywords', []))}</strong> key topics and <strong>{len(nlp_results.get('clusters', {}))}</strong> question clusters.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create result tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "AI Analysis", 
        "Visualizations", 
        "Detailed Insights", 
        "Raw Data",
        "Export"
    ])
    
    with tab1:
        st.markdown("### Comprehensive AI Analysis")
        if llm_results.get('comprehensive_analysis'):
            st.markdown(llm_results['comprehensive_analysis'])
        else:
            st.warning("AI analysis not available. Check your API key and connection.")
        
        if llm_results.get('study_plan'):
            st.markdown("### Personalized Study Plan")
            st.markdown(llm_results['study_plan'])
        
        if llm_results.get('trend_analysis'):
            st.markdown("### Historical Trend Analysis")
            st.markdown(llm_results['trend_analysis'])
    
    with tab2:
        st.markdown("### Interactive Visualizations")
        
        # Question length distribution
        visualizer.plot_question_length_distribution(questions)
        
        # Keywords visualization
        if nlp_results.get('keywords'):
            visualizer.create_keywords_wordcloud(nlp_results['keywords'])
        
        # Cluster analysis
        if nlp_results.get('clusters'):
            visualizer.plot_cluster_distribution(nlp_results['clusters'])
        
        # Complexity analysis
        if nlp_results.get('complexity_scores'):
            visualizer.plot_complexity_analysis(nlp_results['complexity_scores'])
        
        # Year distribution
        visualizer.plot_year_distribution(questions)
        
        # File comparison
        if len(file_metadata) > 1:
            visualizer.create_comparison_charts(file_metadata)
    
    with tab3:
        st.markdown("### Detailed Analysis Results")
        
        # Keywords table
        if nlp_results.get('keywords'):
            visualizer.display_keywords_table(nlp_results['keywords'])
        
        # Frequent questions
        if nlp_results.get('frequent_questions'):
            frequent_data = nlp_results['frequent_questions']
            if frequent_data.get('frequent_groups'):
                st.subheader("Most Frequent Question Patterns")
                for i, group in enumerate(frequent_data['frequent_groups'][:10]):
                    with st.expander(f"Pattern {i+1}: Asked {group['frequency']} times"):
                        st.write(f"**Representative Question:** {group['representative']['text'][:200]}...")
                        st.write(f"**Frequency:** {group['frequency']} similar questions")
        
        # File analysis summary
        st.subheader("File Processing Summary")
        for metadata in file_metadata:
            with st.expander(f"{metadata['file_name']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Pages:** {metadata['page_count']}")
                    st.write(f"**Total Words:** {metadata.get('total_words', 0)}")
                with col2:
                    st.write(f"**File Size:** {metadata.get('file_size', 0)} bytes")
                    st.write(f"**Title:** {metadata.get('title', 'Unknown')}")
    
    with tab4:
        st.markdown("### Question Database")
        
        # Create DataFrame for display
        import pandas as pd
        questions_df = pd.DataFrame([
            {
                'ID': q['id'],
                'Question Preview': q['text'][:100] + "..." if len(q['text']) > 100 else q['text'],
                'Word Count': q['word_count'],
                'Year': q.get('year', 'N/A'),
                'Type': 'MCQ' if q.get('has_options') else 'Descriptive',
                'Estimated Marks': q.get('estimated_marks', 'N/A')
            }
            for q in questions
        ])
        
        st.dataframe(questions_df, use_container_width=True, height=400)
        
        # Question statistics
        question_processor = QuestionProcessor()
        stats = question_processor.get_question_statistics(questions)
        
        if stats:
            st.subheader("Question Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Words", f"{stats['avg_word_count']:.1f}")
                st.metric("Min Words", stats['min_word_count'])
            
            with col2:
                st.metric("Max Words", stats['max_word_count'])
                st.metric("MCQ Questions", stats.get('mcq_questions', 0))
            
            with col3:
                st.metric("Questions with Years", stats.get('questions_with_years', 0))
                st.metric("Year Range", f"{len(stats.get('year_distribution', {}))}")
    
    with tab5:
        st.markdown("### Export Analysis Results")
        
        # Create export package
        export_package = create_export_package(questions, {
            'nlp_results': nlp_results,
            'llm_results': llm_results,
            'file_summary': file_summary
        }, file_metadata)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export questions
            questions_json = export_to_json(questions, "gtu_questions")
            st.download_button(
                label="Download Questions (JSON)",
                data=questions_json,
                file_name=f"gtu_questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Download all extracted questions in JSON format"
            )
        
        with col2:
            # Export complete analysis
            complete_analysis = export_to_json(export_package, "gtu_complete_analysis")
            st.download_button(
                label="Download Complete Analysis",
                data=complete_analysis,
                file_name=f"gtu_complete_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Download complete analysis including all results"
            )
        
        # Export summary report
        st.subheader("Summary Report")
        summary_text = f"""
# GTU PYQs Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total Questions Analyzed:** {len(questions)}
- **Files Processed:** {file_summary.get('total_files', 0)}
- **Key Topics Identified:** {len(nlp_results.get('keywords', []))}
- **Question Clusters:** {len(nlp_results.get('clusters', {}))}

## Top Topics
{chr(10).join([f"- {kw[0]} (Score: {kw[1]:.4f})" for kw in nlp_results.get('keywords', [])[:10]])}

## Recommendations
Based on the analysis, focus your preparation on the topics listed above as they appear most frequently in past examinations.
        """
        
        st.text_area("Report Preview", summary_text, height=200)
        
        st.download_button(
            label="Download Summary Report",
            data=summary_text,
            file_name=f"gtu_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            help="Download a formatted summary report"
        )


def main():
    """Main application function."""
    # Setup page
    setup_page_config()
    load_custom_css()
    
    # Header
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