"""Visualization utilities for data analysis results."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from typing import List, Dict
import matplotlib.pyplot as plt
from config.config import CHART_COLORS


class Visualizer:
    """Handles all visualization components for the application."""
    
    def __init__(self):
        self.colors = CHART_COLORS
    
    def create_overview_metrics(self, questions: List[Dict], file_summary: Dict):
        """
        Create overview metrics display.
        
        Args:
            questions: List of question dictionaries
            file_summary: Summary of processed files
        """
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Total Questions", 
                len(questions),
                help="Total number of questions extracted from all files"
            )
        
        with col2:
            if questions:
                avg_length = np.mean([q['word_count'] for q in questions])
                st.metric(
                    "📝 Avg Question Length", 
                    f"{avg_length:.1f} words",
                    help="Average word count per question"
                )
            else:
                st.metric("📝 Avg Question Length", "N/A")
        
        with col3:
            st.metric(
                "📄 Files Processed", 
                file_summary.get('total_files', 0),
                help="Number of PDF files successfully processed"
            )
        
        with col4:
            st.metric(
                "📖 Total Pages", 
                file_summary.get('total_pages', 0),
                help="Total pages processed across all files"
            )
    
    def plot_question_length_distribution(self, questions: List[Dict]):
        """Create question length distribution histogram."""
        if not questions:
            st.warning("No questions available for length distribution.")
            return
        
        lengths = [q['word_count'] for q in questions]
        
        fig = px.histogram(
            x=lengths,
            nbins=20,
            title="📊 Question Length Distribution",
            labels={'x': 'Word Count', 'y': 'Number of Questions'},
            color_discrete_sequence=[self.colors['primary']]
        )
        
        fig.update_layout(
            showlegend=False,
            height=400,
            xaxis_title="Word Count",
            yaxis_title="Number of Questions"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_keywords_wordcloud(self, keywords: List):
        """Create and display word cloud from keywords."""
        if not keywords:
            st.warning("No keywords available for word cloud.")
            return
        
        st.subheader("📊 Key Topics Word Cloud")
        
        # Create word frequency dictionary
        word_freq = {}
        for word, score in keywords[:30]:  # Top 30 keywords
            # Scale scores for better visualization
            word_freq[word] = max(1, int(score * 1000))
        
        if word_freq:
            try:
                # Generate word cloud
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    colormap='viridis',
                    max_words=50,
                    relative_scaling=0.5
                ).generate_from_frequencies(word_freq)
                
                # Display word cloud
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                
                st.pyplot(fig, use_container_width=True)
                plt.close()
                
            except Exception as e:
                st.error(f"Error generating word cloud: {str(e)}")
                
                # Fallback: Display keywords as table
                self.display_keywords_table(keywords)
    
    def display_keywords_table(self, keywords: List):
        """Display keywords as a formatted table."""
        if not keywords:
            return
        
        st.subheader("🎯 Most Important Topics")
        
        # Create DataFrame
        keywords_df = pd.DataFrame(
            keywords[:20], 
            columns=['Topic', 'Importance Score']
        )
        keywords_df['Importance Score'] = keywords_df['Importance Score'].round(4)
        keywords_df.index = keywords_df.index + 1  # Start index from 1
        
        st.dataframe(keywords_df, use_container_width=True)
    
    def plot_cluster_distribution(self, clusters: Dict):
        """Create cluster distribution visualization."""
        if not clusters:
            st.warning("No clusters available for visualization.")
            return
        
        cluster_sizes = [len(cluster) for cluster in clusters.values()]
        cluster_names = [f"Cluster {i+1}" for i in range(len(clusters))]
        
        fig = px.bar(
            x=cluster_names,
            y=cluster_sizes,
            title="🔍 Question Distribution by Similarity Clusters",
            labels={'x': 'Clusters', 'y': 'Number of Questions'},
            color=cluster_sizes,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show cluster details
        self.display_cluster_details(clusters)
    
    def display_cluster_details(self, clusters: Dict):
        """Display detailed information about each cluster."""
        st.subheader("🔍 Cluster Details")
        
        for i, (cluster_id, cluster_questions) in enumerate(clusters.items()):
            if len(cluster_questions) > 1:  # Only show clusters with multiple questions
                with st.expander(
                    f"Cluster {i+1}: {len(cluster_questions)} similar questions", 
                    expanded=False
                ):
                    for j, item in enumerate(cluster_questions[:5]):  # Show top 5
                        question_text = item['question']['text']
                        if len(question_text) > 150:
                            question_text = question_text[:150] + "..."
                        
                        st.write(f"**{j+1}.** {question_text}")
                        
                        if 'similarity_score' in item:
                            st.caption(f"Similarity Score: {item['similarity_score']:.3f}")
                    
                    if len(cluster_questions) > 5:
                        st.caption(f"... and {len(cluster_questions) - 5} more questions")
    
    def plot_complexity_analysis(self, complexity_scores: List[Dict]):
        """Create complexity analysis visualizations."""
        if not complexity_scores:
            st.warning("No complexity data available.")
            return
        
        st.subheader("📈 Question Complexity Analysis")
        
        # Create DataFrame
        df = pd.DataFrame(complexity_scores)
        
        # Complexity scatter plot
        fig = px.scatter(
            df,
            x='word_count',
            y='flesch_score',
            size='complexity_index',
            color='difficulty_level',
            title="Question Complexity vs Length",
            labels={
                'word_count': 'Word Count', 
                'flesch_score': 'Readability Score (Higher = Easier)',
                'complexity_index': 'Complexity Index'
            },
            hover_data=['fk_grade', 'sentence_count']
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Difficulty distribution
        difficulty_counts = df['difficulty_level'].value_counts()
        
        fig_pie = px.pie(
            values=difficulty_counts.values,
            names=difficulty_counts.index,
            title="Question Difficulty Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    def plot_year_distribution(self, questions: List[Dict]):
        """Plot distribution of questions by year."""
        questions_with_years = [q for q in questions if q.get('year')]
        
        if not questions_with_years:
            st.info("No year information available in the questions.")
            return
        
        # Count questions by year
        year_counts = {}
        for q in questions_with_years:
            year = q['year']
            year_counts[year] = year_counts.get(year, 0) + 1
        
        if not year_counts:
            return
        
        # Create bar chart
        years = sorted(year_counts.keys())
        counts = [year_counts[year] for year in years]
        
        fig = px.bar(
            x=years,
            y=counts,
            title="📅 Question Distribution by Year",
            labels={'x': 'Year', 'y': 'Number of Questions'},
            color=counts,
            color_continuous_scale='blues'
        )
        
        fig.update_layout(
            showlegend=False,
            height=400,
            xaxis_title="Year",
            yaxis_title="Number of Questions"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_comparison_charts(self, file_metadata: List[Dict]):
        """Create comparison charts for multiple files."""
        if len(file_metadata) < 2:
            return
        
        st.subheader("📊 File Comparison")
        
        # Prepare data
        file_data = []
        for meta in file_metadata:
            file_data.append({
                'File': meta['file_name'][:20] + '...' if len(meta['file_name']) > 20 else meta['file_name'],
                'Pages': meta['page_count'],
                'Words': meta.get('total_words', 0),
                'Size (KB)': meta.get('file_size', 0) / 1024
            })
        
        df = pd.DataFrame(file_data)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Pages', 'Words', 'Size (KB)'),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        # Add bar charts
        fig.add_trace(
            go.Bar(x=df['File'], y=df['Pages'], name='Pages', marker_color=self.colors['primary']),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=df['File'], y=df['Words'], name='Words', marker_color=self.colors['secondary']),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=df['File'], y=df['Size (KB)'], name='Size', marker_color=self.colors['success']),
            row=1, col=3
        )
        
        fig.update_layout(height=400, showlegend=False)
        fig.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_progress_metrics(self, current: int, total: int, message: str):
        """Display progress information."""
        progress = current / total if total > 0 else 0
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.progress(progress)
            st.caption(message)
        
        with col2:
            st.metric("Progress", f"{current}/{total}")
    
    def create_summary_dashboard(self, analysis_data: Dict):
        """Create a comprehensive summary dashboard."""
        st.markdown("### 📊 Analysis Summary Dashboard")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = analysis_data.get('metrics', {})
        
        with col1:
            st.metric(
                "Questions Analyzed",
                metrics.get('total_questions', 0),
                delta=metrics.get('unique_questions', 0),
                delta_color="normal",
                help="Total questions with unique count"
            )
        
        with col2:
            st.metric(
                "Topic Clusters",
                metrics.get('clusters', 0),
                help="Number of semantic clusters found"
            )
        
        with col3:
            avg_complexity = metrics.get('avg_complexity', 0)
            st.metric(
                "Avg Complexity",
                f"{avg_complexity:.1f}",
                delta=f"±{metrics.get('complexity_std', 0):.1f}",
                help="Average complexity score with standard deviation"
            )
        
        with col4:
            coverage = metrics.get('topic_coverage', 0)
            st.metric(
                "Topic Coverage",
                f"{coverage:.1f}%",
                help="Percentage of curriculum topics covered"
            )