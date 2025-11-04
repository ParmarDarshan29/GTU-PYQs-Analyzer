# GTU PYQs Analyzer Pro 

An advanced AI-powered system for analyzing Gujarat Technological University (GTU) Previous Year Questions (PYQs) to help students optimize their exam preparation strategy.

## Features

### Advanced Question Analysis
- **Semantic Clustering**: Groups similar questions using ML algorithms
- **Topic Extraction**: TF-IDF based keyword and topic importance analysis
- **Question Complexity**: Readability and difficulty assessment
- **Pattern Recognition**: Identifies frequently repeated question patterns

### AI-Powered Insights
- **Multiple LLM Support**: GPT-4, Claude, and other models via OpenRouter
- **Comprehensive Analysis**: AI-generated study recommendations
- **Trend Analysis**: Historical question pattern evolution
- **Personalized Study Plans**: Adaptive preparation strategies

### Rich Visualizations
- **Interactive Charts**: Plotly-based data visualizations
- **Word Clouds**: Visual topic importance representation
- **Complexity Analysis**: Question difficulty distribution
- **Cluster Visualization**: Question similarity networks

### Export & Sharing
- **Multiple Formats**: JSON, TXT export options
- **Comprehensive Reports**: Detailed analysis summaries
- **Question Database**: Structured question data export
- **Caching System**: Fast repeat analysis

## Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenRouter API key ([Get one here](https://openrouter.ai))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/gtu-pyqs-analyzer.git
cd gtu-pyqs-analyzer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run main.py
```

4. **Access the web interface**
   - Open your browser to `http://localhost:8501`
   - Enter your OpenRouter API key
   - Upload PDF files and start analysis

## Deploy to Streamlit Community Cloud (recommended)

This is the easiest way to publish your app publicly or privately and manage secrets securely.

1. Push your repository to GitHub (public or private).
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click "New app", choose your repository and branch, and set the main file to `main.py`.
4. Add your OpenRouter API key in the app's Secrets (Manage app → Secrets) as `OPENROUTER_API_KEY`.
    - Alternatively, for local testing create a `.streamlit/secrets.toml` file (do not commit it):
      ```toml
      # .streamlit/secrets.toml (local only)
      OPENROUTER_API_KEY = "your_openrouter_api_key_here"
      ```
5. Streamlit Cloud will install dependencies from `requirements.txt` and start the app. If you need a custom start command, use:
    ```
    streamlit run main.py
    ```

Notes:
- Keep secrets out of source control. Use the Streamlit Cloud Secrets UI for production keys.
- The app is configured to read `OPENROUTER_API_KEY` from `st.secrets` or the `OPENROUTER_API_KEY` environment variable automatically.
- If PDF generation requires additional system packages in a hosted container, consult the Streamlit Cloud docs or prefer a Docker-based host like Render.

## Project Structure

```
gtu_pyqs_analyzer/
├── main.py                     # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                  # This file
├── config/
│   └── config.py              # Configuration settings
├── src/
│   ├── __init__.py
│   ├── pdf_processor.py       # PDF text extraction
│   ├── question_processor.py  # Question segmentation & cleaning
│   ├── nlp_analyzer.py        # NLP analysis & clustering
│   ├── llm_analyzer.py        # LLM API integration
│   ├── visualizer.py          # Visualization components
│   └── utils.py               # Utility functions
├── assets/
│   └── styles.css             # Custom CSS styles
└── data/
    └── sample_data/           # Sample files for testing
```

##  Configuration

### API Setup
1. Get an API key from [OpenRouter.ai](https://openrouter.ai)
2. Add credits to your account
3. Enter the key in the sidebar when running the app

### Analysis Options
- **Max Questions**: Limit processing for faster analysis
- **Clustering**: Enable/disable semantic question grouping
- **Complexity Analysis**: Question difficulty assessment
- **Trend Analysis**: Historical pattern analysis
- **Caching**: Store results for faster repeat analysis

## Usage Guide

### 1. Prepare Your Files
- Ensure PDFs contain clear, readable text
- All files should be from the same subject
- Remove any password protection

### 2. Upload & Configure
- Upload multiple PDF files
- Select your preferred AI model
- Adjust analysis settings in sidebar

### 3. Review Results
- **AI Analysis**: Comprehensive insights and recommendations
- **Visualizations**: Interactive charts and graphs
- **Detailed Insights**: Topic analysis and question patterns
- **Raw Data**: Question database and statistics
- **Export**: Download results in multiple formats

## Use Cases

### For Students
- **Exam Preparation**: Identify high-priority topics
- **Study Planning**: Get personalized study schedules
- **Pattern Recognition**: Understand question formats
- **Difficulty Assessment**: Know what to expect

### For Educators
- **Curriculum Analysis**: Understand topic emphasis
- **Question Bank Creation**: Identify question patterns
- **Difficulty Balancing**: Assess question complexity
- **Trend Monitoring**: Track syllabus changes

### For Researchers
- **Educational Data Mining**: Extract learning patterns
- **Assessment Analysis**: Study examination trends
- **NLP Applications**: Question processing techniques
- **ML Model Training**: Educational dataset creation

## Technical Details

### Core Technologies
- **Frontend**: Streamlit with custom CSS
- **NLP Processing**: NLTK, Scikit-learn, TF-IDF
- **Machine Learning**: K-Means clustering, Cosine similarity
- **PDF Processing**: PyMuPDF (fitz)
- **Visualization**: Plotly, Matplotlib, WordCloud
- **AI Integration**: OpenRouter API with multiple LLMs

### Key Algorithms
- **Question Segmentation**: Multi-pattern regex matching
- **Text Preprocessing**: Tokenization, stemming, stop-word removal
- **Clustering**: K-Means with TF-IDF vectorization
- **Similarity Analysis**: Cosine similarity matrix calculation
- **Complexity Scoring**: Flesch-Kincaid readability metrics

### Performance Optimization
- **Caching**: Session-based result storage
- **Batching**: Efficient API request handling
- **Streaming**: Progressive result display
- **Error Handling**: Robust failure recovery

## Output Examples

### Analysis Report Sample
```
# GTU PYQs Analysis Report

## Summary
- Total Questions: 234
- Key Topics: 15
- Question Clusters: 8
- Average Complexity: Medium

## Top Topics (by frequency)
1. Data Structures (Score: 0.95)
2. Algorithms (Score: 0.87)
3. Database Management (Score: 0.76)
4. Computer Networks (Score: 0.68)
5. Operating Systems (Score: 0.61)

## Study Recommendations
- Focus 40% time on Data Structures
- Practice algorithm implementation
- Review database query concepts
- Prepare network protocol diagrams
```

## Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/gtu-pyqs-analyzer.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run tests
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **OpenRouter** for providing accessible LLM APIs
- **Streamlit** for the amazing web framework
- **GTU Students** for inspiration and feedback
- **Open Source Community** for the excellent libraries

## Resume Highlights

This project demonstrates:

### Technical Skills
- **Python**: Advanced programming with modern libraries
- **Machine Learning**: Clustering, similarity analysis, NLP
- **Data Science**: Statistical analysis, visualization, data processing
- **API Integration**: RESTful APIs, error handling, retry logic
- **Web Development**: Streamlit, custom CSS, responsive design

### Software Engineering
- **Modular Architecture**: Clean code organization
- **Error Handling**: Robust exception management
- **Testing**: Input validation, edge case handling
- **Performance**: Caching, optimization, scalability
- **Documentation**: Comprehensive project documentation

### Problem Solving
- **Educational Technology**: Real-world application
- **User Experience**: Intuitive interface design
- **Data Processing**: Large-scale text analysis
- **AI Integration**: Multi-model LLM utilization

---

