import streamlit as st

# Configure the page
st.set_page_config(
    page_title="Haystack RAG Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page content
st.title("🔍 Haystack RAG in Production")
st.subheader("A Comprehensive Guide to Retrieval-Augmented Generation")

st.markdown("""
Welcome to the Haystack RAG Demo application! This interactive guide demonstrates the key concepts 
from the O'Reilly book "Retrieval-Augmented Generation in Production with Haystack" by Skanda Vivek.

## About This Demo

This application provides an interactive exploration of:

- **RAG Fundamentals**: Understanding the core concepts of Retrieval-Augmented Generation
- **Haystack Framework**: Learning how to build production-ready RAG systems
- **Evaluation & Optimization**: Techniques for improving RAG performance
- **Scalability**: Moving from prototype to production
- **Observability**: Monitoring and maintaining RAG systems
- **Governance**: Managing costs, privacy, and security
- **Advanced Techniques**: Exploring cutting-edge RAG applications

## Navigation

Use the sidebar to navigate through different sections of the guide. Each page corresponds to a chapter 
from the book and includes:

- Key concepts and explanations
- Interactive examples and demonstrations
- Code snippets and best practices
- Practical implementation guidance

## Getting Started

Select a chapter from the sidebar to begin your journey into production-ready RAG systems with Haystack!
""")

# Sidebar navigation info
with st.sidebar:
    st.markdown("## 📚 Book Chapters")
    st.markdown("""
    - **Introduction**: RAG with Haystack
    - **Chapter 1**: Introduction to RAG
    - **Chapter 2**: Evaluating and Optimizing RAG
    - **Chapter 3**: Scalable AI
    - **Chapter 4**: Observable AI
    - **Chapter 5**: Governance of AI
    - **Chapter 6**: Advanced RAG
    """)
    
    st.markdown("---")
    st.markdown("**Author**: Skanda Vivek")
    st.markdown("**Publisher**: O'Reilly Media")

