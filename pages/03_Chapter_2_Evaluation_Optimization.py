import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Chapter 2: Evaluation & Optimization", page_icon="📊", layout="wide")

st.title("📊 Chapter 2: Evaluating and Optimizing RAG")

st.markdown("""
## The Quality Imperative

User feedback consistently shows that LLM responses can be too generic or noticeably AI-generated. 
As humans, we're very sensitive to small discrepancies, and with numerous options available, 
customers will quickly abandon low-quality applications.

**Key Challenge**: How do we measure performance and make improvements in RAG systems?
""")

# RAG Components Visualization
st.markdown("## 🔧 RAG System Components")

col1, col2 = st.columns([3, 1])

with col1:
    # Create RAG pipeline visualization
    fig = go.Figure()
    
    # Indexing Pipeline (Left side)
    indexing_components = [
        {"x": 1, "y": 5, "text": "Text\nExtraction", "color": "#FF6B6B"},
        {"x": 1, "y": 4, "text": "Chunking/\nSplitting", "color": "#4ECDC4"},
        {"x": 1, "y": 3, "text": "Embedding", "color": "#45B7D1"},
        {"x": 1, "y": 2, "text": "Database\nStorage", "color": "#96CEB4"}
    ]
    
    # Query Pipeline (Right side)
    query_components = [
        {"x": 4, "y": 5, "text": "Query\nProcessing", "color": "#FFEAA7"},
        {"x": 4, "y": 4, "text": "Retrieval\nStrategy", "color": "#DDA0DD"},
        {"x": 4, "y": 3, "text": "LLM Model\nChoice", "color": "#FFB6C1"},
        {"x": 4, "y": 2, "text": "Response\nGeneration", "color": "#98FB98"}
    ]
    
    all_components = indexing_components + query_components
    
    for comp in all_components:
        fig.add_trace(go.Scatter(
            x=[comp["x"]], y=[comp["y"]], 
            mode='markers+text',
            marker=dict(size=80, color=comp["color"]),
            text=comp["text"],
            textposition="middle center",
            textfont=dict(size=9, color="white"),
            showlegend=False
        ))
    
    # Add pipeline labels
    fig.add_annotation(x=1, y=5.7, text="Indexing Pipeline", 
                      showarrow=False, font=dict(size=14, color="blue"))
    fig.add_annotation(x=4, y=5.7, text="Query Pipeline", 
                      showarrow=False, font=dict(size=14, color="green"))
    
    # Add arrows between components
    arrows = [
        (1, 5, 1, 4), (1, 4, 1, 3), (1, 3, 1, 2),  # Indexing flow
        (4, 5, 4, 4), (4, 4, 4, 3), (4, 3, 4, 2),  # Query flow
        (1, 2, 4, 4)  # Database to Retrieval
    ]
    
    for arrow in arrows:
        fig.add_annotation(
            x=arrow[2], y=arrow[3],
            ax=arrow[0], ay=arrow[1],
            xref="x", yref="y",
            axref="x", ayref="y",
            arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="gray"
        )
    
    fig.update_layout(
        title="RAG System Components to Optimize",
        xaxis=dict(range=[0, 5], showgrid=False, showticklabels=False),
        yaxis=dict(range=[1, 6], showgrid=False, showticklabels=False),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("""
    ### Optimization Areas:
    
    **Indexing Pipeline:**
    - Text extraction quality
    - Chunking strategies
    - Embedding models
    - Database selection
    
    **Query Pipeline:**
    - Query processing
    - Retrieval algorithms
    - LLM model choice
    - Prompt engineering
    """)

# Evaluation Approaches
st.markdown("## 📏 RAG Evaluation Approaches")

tab1, tab2 = st.tabs(["With Ground Truth", "Without Ground Truth"])

with tab1:
    st.markdown("""
    ### Traditional ML Metrics
    
    When labeled data is available, we can use standard metrics:
    """)
    
    # Example evaluation metrics
    metrics_data = {
        'Metric': ['Exact Match', 'F1 Score', 'BLEU', 'ROUGE-L', 'BERTScore'],
        'Description': [
            'Perfect string match with ground truth',
            'Harmonic mean of precision and recall',
            'N-gram overlap with reference',
            'Longest common subsequence',
            'Semantic similarity using BERT'
        ],
        'Use Case': [
            'Factual QA', 'Token-level accuracy', 
            'Translation tasks', 'Summarization', 'Semantic similarity'
        ],
        'Score Range': ['0-1', '0-1', '0-1', '0-1', '0-1']
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, use_container_width=True)
    
    # Example evaluation
    st.markdown("### 📝 Example Evaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Question:** "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes, France?"
        
        **Ground Truth:** "Saint Bernadette Soubirous"
        
        **Model Response:** "Saint Bernadette"
        """)
    
    with col2:
        # Calculate example metrics
        example_scores = {
            'Metric': ['Exact Match', 'F1 Score', 'Partial Match'],
            'Score': [0.0, 0.67, 1.0]
        }
        
        fig_scores = px.bar(example_scores, x='Metric', y='Score',
                           title='Example Evaluation Scores',
                           color='Score', color_continuous_scale='RdYlGn')
        fig_scores.update_layout(showlegend=False)
        st.plotly_chart(fig_scores, use_container_width=True)

with tab2:
    st.markdown("""
    ### LLM-as-a-Judge Evaluation
    
    When ground truth isn't available, use strong LLMs (GPT-4, Claude 3.5) as evaluators:
    """)
    
    # LLM evaluation criteria
    eval_criteria = {
        'Criteria': ['Relevance', 'Accuracy', 'Completeness', 'Coherence', 'Helpfulness'],
        'Weight': [0.25, 0.25, 0.20, 0.15, 0.15],
        'Example Score': [8.5, 9.0, 7.5, 8.0, 8.5]
    }
    
    df_criteria = pd.DataFrame(eval_criteria)
    
    fig_criteria = px.bar(df_criteria, x='Criteria', y='Example Score',
                         title='LLM-as-a-Judge Evaluation Criteria',
                         color='Weight', color_continuous_scale='viridis')
    st.plotly_chart(fig_criteria, use_container_width=True)
    
    # Evaluation prompt example
    st.markdown("### 🤖 Example Evaluation Prompt")
    st.code("""
    Evaluate the following RAG response on a scale of 1-10:

    Question: {question}
    Context: {retrieved_context}
    Response: {model_response}

    Rate the response on:
    1. Relevance (how well it answers the question)
    2. Accuracy (factual correctness based on context)
    3. Completeness (covers all important aspects)
    4. Coherence (logical flow and clarity)
    5. Helpfulness (practical value to user)

    Provide scores and brief justification for each criterion.
    """, language="text")

# Optimization Strategies
st.markdown("## ⚡ Pipeline Optimization Strategies")

optimization_tabs = st.tabs(["Retrieval", "Generation", "End-to-End"])

with optimization_tabs[0]:
    st.markdown("### 🔍 Retrieval Optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Chunking Strategies:**
        - Fixed-size chunks
        - Semantic chunking
        - Overlapping windows
        - Hierarchical chunking
        
        **Embedding Models:**
        - Domain-specific embeddings
        - Multi-language models
        - Fine-tuned embeddings
        """)
    
    with col2:
        # Chunking strategy comparison
        chunk_data = {
            'Strategy': ['Fixed Size', 'Semantic', 'Overlapping', 'Hierarchical'],
            'Precision': [6, 8, 7, 9],
            'Recall': [7, 6, 8, 8],
            'Speed': [9, 6, 7, 5]
        }
        
        df_chunk = pd.DataFrame(chunk_data)
        
        fig_chunk = go.Figure()
        for metric in ['Precision', 'Recall', 'Speed']:
            fig_chunk.add_trace(go.Scatter(
                x=df_chunk['Strategy'],
                y=df_chunk[metric],
                mode='lines+markers',
                name=metric,
                line=dict(width=3)
            ))
        
        fig_chunk.update_layout(
            title='Chunking Strategy Comparison',
            yaxis_title='Score (1-10)',
            yaxis=dict(range=[0, 10])
        )
        
        st.plotly_chart(fig_chunk, use_container_width=True)

with optimization_tabs[1]:
    st.markdown("### 🎯 Generation Optimization")
    
    # Prompt engineering techniques
    prompt_techniques = {
        'Technique': ['Few-shot', 'Chain-of-Thought', 'Role Playing', 'Template Optimization'],
        'Effectiveness': [7, 8, 6, 9],
        'Complexity': [3, 6, 4, 2],
        'Use Case': ['Examples', 'Reasoning', 'Persona', 'Structure']
    }
    
    df_prompt = pd.DataFrame(prompt_techniques)
    
    fig_prompt = px.scatter(df_prompt, x='Complexity', y='Effectiveness',
                           size=[10]*4, color='Technique',
                           title='Prompt Engineering Techniques',
                           hover_data=['Use Case'])
    fig_prompt.update_layout(
        xaxis_title='Implementation Complexity',
        yaxis_title='Effectiveness Score'
    )
    
    st.plotly_chart(fig_prompt, use_container_width=True)
    
    st.markdown("""
    **Model Selection Factors:**
    - Task complexity
    - Latency requirements
    - Cost constraints
    - Quality thresholds
    """)

with optimization_tabs[2]:
    st.markdown("### 🔄 End-to-End Optimization")
    
    # A/B testing framework
    st.markdown("#### A/B Testing Framework")
    
    ab_results = {
        'Variant': ['Baseline', 'Optimized Chunking', 'Better Embeddings', 'Improved Prompts'],
        'User Satisfaction': [6.5, 7.2, 7.8, 8.1],
        'Response Time (s)': [2.3, 2.1, 2.5, 2.2],
        'Cost per Query': [0.05, 0.06, 0.08, 0.05]
    }
    
    df_ab = pd.DataFrame(ab_results)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_satisfaction = px.bar(df_ab, x='Variant', y='User Satisfaction',
                                 title='User Satisfaction by Variant',
                                 color='User Satisfaction',
                                 color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_satisfaction, use_container_width=True)
    
    with col2:
        fig_cost = px.bar(df_ab, x='Variant', y='Cost per Query',
                         title='Cost per Query by Variant',
                         color='Cost per Query',
                         color_continuous_scale='RdYlBu_r')
        st.plotly_chart(fig_cost, use_container_width=True)

# Interactive Optimization Simulator
st.markdown("## 🎮 Interactive Optimization Simulator")

st.markdown("Adjust the parameters below to see how they affect RAG performance:")

col1, col2, col3 = st.columns(3)

with col1:
    chunk_size = st.slider("Chunk Size", 100, 1000, 500, 50)
    embedding_quality = st.slider("Embedding Quality", 1, 10, 7)

with col2:
    retrieval_k = st.slider("Top-K Retrieval", 1, 20, 5)
    model_size = st.selectbox("Model Size", ["Small", "Medium", "Large"])

with col3:
    prompt_engineering = st.slider("Prompt Engineering", 1, 10, 5)
    context_window = st.slider("Context Window", 1000, 8000, 4000, 500)

# Calculate simulated performance
def calculate_performance(chunk_size, embedding_quality, retrieval_k, model_size, prompt_engineering, context_window):
    # Simplified performance calculation
    base_score = 5.0
    
    # Chunk size effect (optimal around 500)
    chunk_effect = 1 - abs(chunk_size - 500) / 1000
    
    # Other factors
    embedding_effect = embedding_quality / 10
    retrieval_effect = min(retrieval_k / 10, 1.0)
    model_effect = {"Small": 0.7, "Medium": 0.85, "Large": 1.0}[model_size]
    prompt_effect = prompt_engineering / 10
    context_effect = min(context_window / 4000, 1.0)
    
    performance = base_score + chunk_effect + embedding_effect + retrieval_effect + model_effect + prompt_effect + context_effect
    return min(performance, 10.0)

performance_score = calculate_performance(chunk_size, embedding_quality, retrieval_k, model_size, prompt_engineering, context_window)

st.markdown(f"### 📊 Predicted Performance Score: **{performance_score:.2f}/10**")

# Performance breakdown
performance_factors = {
    'Factor': ['Chunking', 'Embeddings', 'Retrieval', 'Model', 'Prompts', 'Context'],
    'Contribution': [
        1 - abs(chunk_size - 500) / 1000,
        embedding_quality / 10,
        min(retrieval_k / 10, 1.0),
        {"Small": 0.7, "Medium": 0.85, "Large": 1.0}[model_size],
        prompt_engineering / 10,
        min(context_window / 4000, 1.0)
    ]
}

fig_breakdown = px.bar(performance_factors, x='Factor', y='Contribution',
                      title='Performance Factor Breakdown',
                      color='Contribution',
                      color_continuous_scale='RdYlGn')
st.plotly_chart(fig_breakdown, use_container_width=True)

# Key Takeaways
st.markdown("---")
st.markdown("## 🎯 Key Takeaways")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### ✅ Best Practices:
    - Use both automated and human evaluation
    - Implement A/B testing for optimizations
    - Monitor performance continuously
    - Balance quality, speed, and cost
    """)

with col2:
    st.markdown("""
    ### 🚨 Common Pitfalls:
    - Over-optimizing for single metrics
    - Ignoring user feedback
    - Not testing in production conditions
    - Premature optimization
    """)

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Chapter 2 Summary")
    st.markdown("""
    ### Key Concepts:
    - RAG evaluation methods
    - Optimization strategies
    - A/B testing frameworks
    - Performance monitoring
    
    ### Evaluation Types:
    ✅ Ground truth metrics  
    ✅ LLM-as-a-judge  
    ✅ User feedback  
    ✅ Production metrics  
    """)
    
    st.markdown("### 💡 Key Takeaway")
    st.success("Continuous evaluation and optimization are essential for maintaining high-quality RAG systems in production.")

