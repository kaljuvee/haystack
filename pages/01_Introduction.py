import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Introduction", page_icon="📖", layout="wide")

st.title("📖 Introduction to RAG in Production")

st.markdown("""
## The AI Revolution

More than two years after OpenAI made large language models (LLMs) available to the general public 
through ChatGPT, the landscape continues to evolve rapidly. Organizations across all sectors are 
reimagining how customers experience their products through AI.

### Key Developments

""")

# Timeline visualization
timeline_data = {
    'Year': [2022, 2023, 2024, 2025],
    'Milestone': [
        'ChatGPT Launch',
        'Enterprise AI Adoption',
        'RAG Standardization',
        'Production Maturity'
    ],
    'Impact': [100, 300, 500, 700]
}

fig = px.line(timeline_data, x='Year', y='Impact', 
              title='AI Adoption Timeline',
              markers=True)
fig.update_traces(line=dict(color='#1f77b4', width=3))
fig.update_layout(
    xaxis_title="Year",
    yaxis_title="Adoption Impact",
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
## Why RAG?

Retrieval-Augmented Generation (RAG) has emerged as the paradigm for making generative AI useful 
for a wide range of applications. RAG addresses key limitations of standalone LLMs:

""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### ❌ LLM Limitations
    - **Knowledge Cutoff**: Training data has a specific cutoff date
    - **Hallucinations**: May generate plausible but incorrect information
    - **Context Limits**: Limited input context window
    - **Static Knowledge**: Cannot access real-time or private data
    """)

with col2:
    st.markdown("""
    ### ✅ RAG Benefits
    - **Current Information**: Access to up-to-date data
    - **Grounded Responses**: Answers based on retrieved documents
    - **Domain Expertise**: Leverage private/specialized knowledge
    - **Cost Effective**: Avoid fine-tuning for domain adaptation
    """)

st.markdown("""
## Haystack Framework

This guide uses **Haystack**, a popular and battle-tested open source Python framework for building 
compound AI systems. Haystack provides:

""")

# Haystack features
features = {
    'Feature': ['Modular Components', 'Model Integrations', 'Custom Logic', 'Observability', 'Documentation'],
    'Description': [
        'Combine existing components into powerful systems',
        'Integrations with major model providers and databases',
        'Add custom functionality through custom components',
        'Monitoring and stability options for production',
        'Comprehensive docs and active community'
    ],
    'Importance': [9, 8, 7, 9, 6]
}

fig2 = px.bar(features, x='Feature', y='Importance', 
              title='Haystack Framework Strengths',
              color='Importance',
              color_continuous_scale='viridis')
fig2.update_layout(showlegend=False)

st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
## What You'll Learn

This interactive guide will take you through:

1. **RAG Fundamentals** - Understanding core concepts and building your first RAG app
2. **Evaluation & Optimization** - Measuring and improving RAG performance
3. **Scalable AI** - Moving from prototype to production-ready systems
4. **Observable AI** - Monitoring, logging, and handling data drift
5. **Governance** - Managing costs, privacy, security, and compliance
6. **Advanced Techniques** - AI agents, multimodal RAG, and knowledge graphs

### Interactive Elements

Each section includes:
- 📊 **Visualizations** of key concepts
- 💻 **Code examples** and best practices
- 🔧 **Interactive demos** where possible
- 📈 **Performance metrics** and evaluation techniques

""")

# Progress indicator
st.markdown("---")
st.markdown("### 🚀 Ready to Start?")
st.markdown("Navigate to **Chapter 1** in the sidebar to begin your journey into RAG with Haystack!")

# Sidebar content
with st.sidebar:
    st.markdown("## 📚 Guide Overview")
    st.info("""
    This guide demonstrates concepts from the O'Reilly book 
    "Retrieval-Augmented Generation in Production with Haystack" 
    by Skanda Vivek.
    """)
    
    st.markdown("### 🎯 Learning Objectives")
    st.markdown("""
    - Understand RAG fundamentals
    - Build production-ready systems
    - Implement monitoring and governance
    - Explore advanced RAG techniques
    """)

