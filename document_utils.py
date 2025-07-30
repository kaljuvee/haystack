"""
Document processing utilities for the Haystack RAG Demo application.
"""

import streamlit as st
import PyPDF2
import io
import os
import re
import textstat
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text content from uploaded PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def get_sample_documents() -> Dict[str, str]:
    """Get list of available sample documents."""
    sample_docs = {
        "AI Systems Overview": "sample_documents/ai_systems_overview.pdf",
        "Machine Learning Best Practices": "sample_documents/ml_best_practices.pdf", 
        "Data Science Methodology": "sample_documents/data_science_methodology.pdf"
    }
    return sample_docs

def load_sample_document(doc_path: str) -> str:
    """Load text from a sample document."""
    try:
        full_path = os.path.join(os.path.dirname(__file__), doc_path)
        with open(full_path, 'rb') as file:
            return extract_text_from_pdf(file)
    except Exception as e:
        st.error(f"Error loading sample document: {str(e)}")
        return ""

def analyze_document_stats(text: str) -> Dict[str, any]:
    """Analyze basic statistics of the document."""
    if not text.strip():
        return {}
    
    # Basic statistics
    word_count = len(text.split())
    char_count = len(text)
    sentence_count = len(sent_tokenize(text))
    paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
    
    # Readability scores
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
    
    # Average sentence and word length
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    words = text.split()
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    
    return {
        "word_count": word_count,
        "character_count": char_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "avg_word_length": round(avg_word_length, 2),
        "flesch_reading_ease": round(flesch_reading_ease, 2),
        "flesch_kincaid_grade": round(flesch_kincaid_grade, 2)
    }

def extract_key_terms(text: str, top_n: int = 10) -> List[Tuple[str, float]]:
    """Extract key terms using TF-IDF."""
    if not text.strip():
        return []
    
    try:
        # Clean and preprocess text
        sentences = sent_tokenize(text)
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8
        )
        
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get average TF-IDF scores
        mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # Create term-score pairs and sort
        term_scores = list(zip(feature_names, mean_scores))
        term_scores.sort(key=lambda x: x[1], reverse=True)
        
        return term_scores[:top_n]
    except Exception as e:
        st.error(f"Error extracting key terms: {str(e)}")
        return []

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    if not text.strip():
        return []
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        
        # Break if we've reached the end
        if i + chunk_size >= len(words):
            break
    
    return chunks

def simulate_retrieval(query: str, text: str, top_k: int = 3) -> List[Dict[str, any]]:
    """Simulate document retrieval using cosine similarity."""
    if not text.strip() or not query.strip():
        return []
    
    try:
        # Chunk the document
        chunks = chunk_text(text)
        if not chunks:
            return []
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        
        # Combine query and chunks for vectorization
        all_texts = [query] + chunks
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate similarities
        query_vector = tfidf_matrix[0:1]
        chunk_vectors = tfidf_matrix[1:]
        
        similarities = cosine_similarity(query_vector, chunk_vectors)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include relevant results
                results.append({
                    "chunk": chunks[idx],
                    "similarity": round(similarities[idx], 4),
                    "chunk_id": idx + 1
                })
        
        return results
    except Exception as e:
        st.error(f"Error in retrieval simulation: {str(e)}")
        return []

def simulate_rag_response(query: str, retrieved_chunks: List[Dict[str, any]]) -> str:
    """Simulate RAG response generation."""
    if not retrieved_chunks:
        return "No relevant information found in the document."
    
    # Combine retrieved chunks
    context = "\n\n".join([chunk["chunk"] for chunk in retrieved_chunks])
    
    # Simple response simulation
    response = f"""Based on the document content, here's what I found regarding your query: "{query}"

**Relevant Information:**
{context[:1000]}{'...' if len(context) > 1000 else ''}

**Summary:**
The document contains information related to your query with similarity scores ranging from {retrieved_chunks[-1]['similarity']:.3f} to {retrieved_chunks[0]['similarity']:.3f}. The most relevant section discusses concepts that directly address your question.

*Note: This is a simulated RAG response for demonstration purposes.*"""
    
    return response

def calculate_rag_metrics(query: str, retrieved_chunks: List[Dict[str, any]], ground_truth: Optional[str] = None) -> Dict[str, float]:
    """Calculate RAG evaluation metrics."""
    metrics = {}
    
    if not retrieved_chunks:
        return {"retrieval_success": 0.0, "avg_similarity": 0.0, "coverage": 0.0}
    
    # Retrieval metrics
    metrics["retrieval_success"] = 1.0 if retrieved_chunks else 0.0
    metrics["avg_similarity"] = np.mean([chunk["similarity"] for chunk in retrieved_chunks])
    metrics["max_similarity"] = max([chunk["similarity"] for chunk in retrieved_chunks])
    metrics["min_similarity"] = min([chunk["similarity"] for chunk in retrieved_chunks])
    
    # Coverage metric (how much of the query terms are covered)
    query_terms = set(query.lower().split())
    retrieved_text = " ".join([chunk["chunk"].lower() for chunk in retrieved_chunks])
    covered_terms = sum(1 for term in query_terms if term in retrieved_text)
    metrics["coverage"] = covered_terms / len(query_terms) if query_terms else 0.0
    
    # Diversity metric (how diverse are the retrieved chunks)
    if len(retrieved_chunks) > 1:
        chunk_texts = [chunk["chunk"] for chunk in retrieved_chunks]
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            vectors = vectorizer.fit_transform(chunk_texts)
            similarities = cosine_similarity(vectors)
            # Average pairwise similarity (lower = more diverse)
            pairwise_sims = []
            for i in range(len(similarities)):
                for j in range(i+1, len(similarities)):
                    pairwise_sims.append(similarities[i][j])
            metrics["diversity"] = 1.0 - np.mean(pairwise_sims) if pairwise_sims else 1.0
        except:
            metrics["diversity"] = 1.0
    else:
        metrics["diversity"] = 1.0
    
    return metrics

def display_document_upload_section():
    """Display the document upload section with sample documents."""
    st.markdown("### 📄 Document Processing")
    
    # Create tabs for upload and sample documents
    upload_tab, sample_tab = st.tabs(["Upload Document", "Sample Documents"])
    
    with upload_tab:
        uploaded_file = st.file_uploader(
            "Upload a PDF document",
            type=['pdf'],
            help="Upload a PDF document to analyze with RAG techniques"
        )
        
        if uploaded_file is not None:
            text = extract_text_from_pdf(uploaded_file)
            if text:
                st.success(f"✅ Document uploaded successfully! Extracted {len(text.split())} words.")
                return text, uploaded_file.name
    
    with sample_tab:
        sample_docs = get_sample_documents()
        selected_doc = st.selectbox(
            "Choose a sample document",
            options=list(sample_docs.keys()),
            help="Select from pre-loaded sample documents"
        )
        
        if st.button("Load Sample Document"):
            doc_path = sample_docs[selected_doc]
            text = load_sample_document(doc_path)
            if text:
                st.success(f"✅ Sample document loaded! Extracted {len(text.split())} words.")
                return text, selected_doc
    
    return None, None

def display_document_analysis(text: str, doc_name: str):
    """Display document analysis results."""
    if not text:
        return
    
    st.markdown(f"### 📊 Document Analysis: {doc_name}")
    
    # Basic statistics
    stats = analyze_document_stats(text)
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Words", stats["word_count"])
            st.metric("Sentences", stats["sentence_count"])
        
        with col2:
            st.metric("Characters", stats["character_count"])
            st.metric("Paragraphs", stats["paragraph_count"])
        
        with col3:
            st.metric("Avg Sentence Length", f"{stats['avg_sentence_length']} words")
            st.metric("Avg Word Length", f"{stats['avg_word_length']} chars")
        
        with col4:
            st.metric("Reading Ease", stats["flesch_reading_ease"])
            st.metric("Grade Level", stats["flesch_kincaid_grade"])
    
    # Key terms
    st.markdown("#### 🔑 Key Terms")
    key_terms = extract_key_terms(text)
    if key_terms:
        terms_df = pd.DataFrame(key_terms, columns=["Term", "TF-IDF Score"])
        st.dataframe(terms_df, use_container_width=True)
    
    # Text preview
    with st.expander("📖 Document Preview"):
        st.text_area("Document Content", text[:2000] + "..." if len(text) > 2000 else text, height=200)

def display_rag_demo(text: str, doc_name: str):
    """Display interactive RAG demonstration."""
    if not text:
        return
    
    st.markdown(f"### 🤖 RAG Demonstration with {doc_name}")
    
    # Query input
    query = st.text_input(
        "Enter your question about the document:",
        placeholder="e.g., What are the main challenges in AI implementation?",
        help="Ask any question about the uploaded document"
    )
    
    if query and st.button("🔍 Search & Generate Response"):
        with st.spinner("Processing your query..."):
            # Simulate retrieval
            retrieved_chunks = simulate_retrieval(query, text)
            
            if retrieved_chunks:
                # Display retrieval results
                st.markdown("#### 📋 Retrieved Chunks")
                for i, chunk in enumerate(retrieved_chunks):
                    with st.expander(f"Chunk {chunk['chunk_id']} (Similarity: {chunk['similarity']:.3f})"):
                        st.write(chunk["chunk"])
                
                # Generate and display response
                st.markdown("#### 🎯 Generated Response")
                response = simulate_rag_response(query, retrieved_chunks)
                st.markdown(response)
                
                # Display metrics
                st.markdown("#### 📈 RAG Metrics")
                metrics = calculate_rag_metrics(query, retrieved_chunks)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Retrieval Success", f"{metrics['retrieval_success']:.1%}")
                with col2:
                    st.metric("Avg Similarity", f"{metrics['avg_similarity']:.3f}")
                with col3:
                    st.metric("Query Coverage", f"{metrics['coverage']:.1%}")
                with col4:
                    st.metric("Result Diversity", f"{metrics['diversity']:.3f}")
                
            else:
                st.warning("No relevant chunks found for your query. Try rephrasing your question.")

