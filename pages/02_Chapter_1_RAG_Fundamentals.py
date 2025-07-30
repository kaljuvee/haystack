import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Chapter 1: RAG Fundamentals", page_icon="🧠", layout="wide")

st.title("🧠 Chapter 1: Introduction to RAG with Haystack")

st.markdown("""
## The Shift to Compound AI Systems

The AI landscape has evolved from simple model-centric approaches to **compound AI systems** that involve:
- Multiple LLM calls
- Dynamic data connections
- Complex orchestration workflows
- Integration of various AI components

""")

# Interactive RAG Architecture Diagram
st.markdown("## 🏗️ RAG Architecture Overview")

col1, col2 = st.columns([2, 1])

with col1:
    # Create a flow diagram using plotly
    fig = go.Figure()
    
    # Add nodes
    nodes = [
        {"x": 1, "y": 4, "text": "User Query", "color": "#FF6B6B"},
        {"x": 3, "y": 4, "text": "Retrieval System", "color": "#4ECDC4"},
        {"x": 5, "y": 4, "text": "Document Store", "color": "#45B7D1"},
        {"x": 3, "y": 2, "text": "Context Augmentation", "color": "#96CEB4"},
        {"x": 1, "y": 2, "text": "LLM Generation", "color": "#FFEAA7"},
        {"x": 1, "y": 0, "text": "Final Response", "color": "#DDA0DD"}
    ]
    
    for node in nodes:
        fig.add_trace(go.Scatter(
            x=[node["x"]], y=[node["y"]], 
            mode='markers+text',
            marker=dict(size=60, color=node["color"]),
            text=node["text"],
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            showlegend=False
        ))
    
    # Add arrows
    arrows = [
        (1, 4, 3, 4),  # Query to Retrieval
        (3, 4, 5, 4),  # Retrieval to Document Store
        (3, 4, 3, 2),  # Retrieval to Context
        (3, 2, 1, 2),  # Context to LLM
        (1, 2, 1, 0),  # LLM to Response
    ]
    
    for arrow in arrows:
        fig.add_annotation(
            x=arrow[2], y=arrow[3],
            ax=arrow[0], ay=arrow[1],
            xref="x", yref="y",
            axref="x", ayref="y",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="gray"
        )
    
    fig.update_layout(
        title="RAG System Flow",
        xaxis=dict(range=[0, 6], showgrid=False, showticklabels=False),
        yaxis=dict(range=[-1, 5], showgrid=False, showticklabels=False),
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("""
    ### RAG Process Steps:
    
    1. **User Query** - Input question or request
    2. **Retrieval** - Search relevant documents
    3. **Document Store** - Knowledge base access
    4. **Context Augmentation** - Combine query + retrieved docs
    5. **LLM Generation** - Generate response with context
    6. **Final Response** - Contextually grounded answer
    """)

# LLM Fundamentals Section
st.markdown("## 🤖 Large Language Models (LLMs)")

st.markdown("""
LLMs are large-scale neural networks with billions of parameters, trained on natural language processing tasks. 
They model the generative likelihood of word sequences and predict probabilities of future tokens.

### Key LLM Characteristics:
""")

llm_data = {
    'Model': ['GPT-3.5', 'GPT-4', 'Claude-3', 'Llama-2', 'Gemini'],
    'Parameters (B)': [175, 1000, 200, 70, 540],
    'Context Window': [4096, 32768, 200000, 4096, 32768],
    'Training Data (TB)': [45, 100, 60, 2, 80]
}

df = pd.DataFrame(llm_data)

col1, col2 = st.columns(2)

with col1:
    fig1 = px.bar(df, x='Model', y='Parameters (B)', 
                  title='Model Size Comparison',
                  color='Parameters (B)',
                  color_continuous_scale='viridis')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.bar(df, x='Model', y='Context Window', 
                  title='Context Window Sizes',
                  color='Context Window',
                  color_continuous_scale='plasma')
    st.plotly_chart(fig2, use_container_width=True)

# RAG vs Traditional LLM Comparison
st.markdown("## ⚖️ RAG vs Traditional LLM Approaches")

comparison_data = {
    'Aspect': ['Knowledge Currency', 'Accuracy', 'Cost', 'Customization', 'Hallucination Risk'],
    'Traditional LLM': [3, 6, 4, 3, 7],
    'RAG System': [9, 9, 7, 9, 3]
}

df_comp = pd.DataFrame(comparison_data)

fig3 = go.Figure()
fig3.add_trace(go.Scatterpolar(
    r=df_comp['Traditional LLM'],
    theta=df_comp['Aspect'],
    fill='toself',
    name='Traditional LLM',
    line_color='red'
))
fig3.add_trace(go.Scatterpolar(
    r=df_comp['RAG System'],
    theta=df_comp['Aspect'],
    fill='toself',
    name='RAG System',
    line_color='blue'
))

fig3.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 10]
        )),
    showlegend=True,
    title="RAG vs Traditional LLM Comparison (1-10 scale)"
)

st.plotly_chart(fig3, use_container_width=True)

# Interactive RAG Demo Section
st.markdown("## 🔧 Interactive RAG Concept Demo")

st.markdown("""
Let's simulate how RAG works with a simple example:
""")

# User input for demo
user_query = st.text_input("Enter a query:", value="What is machine learning?")

if user_query:
    # Simulated document retrieval
    documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
        "Machine learning algorithms build mathematical models based on training data to make predictions or decisions.",
        "Common types of machine learning include supervised learning, unsupervised learning, and reinforcement learning."
    ]
    
    st.markdown("### 📚 Retrieved Documents:")
    for i, doc in enumerate(documents, 1):
        st.markdown(f"**Document {i}:** {doc}")
    
    st.markdown("### 🤖 RAG Response Generation:")
    st.markdown(f"""
    **Query:** {user_query}
    
    **Context:** {' '.join(documents)}
    
    **Generated Response:** Based on the retrieved documents, {user_query.lower()} refers to a subset of artificial intelligence that enables computers to learn from experience without explicit programming. It involves building mathematical models from training data to make predictions, with common types including supervised, unsupervised, and reinforcement learning approaches.
    """)

# Building Industry Applications
st.markdown("## 🏭 Building Industry LLM Applications")

st.markdown("""
Enterprise use cases require adapting LLMs to:
- Organization-specific data sources
- Custom workflows
- Domain expertise
- Real-time information needs

### Challenges with Direct LLM Usage:
""")

challenges = {
    'Challenge': ['Latency', 'Cost', 'Context Limits', 'Knowledge Cutoff', 'Hallucinations'],
    'Impact': [8, 9, 7, 8, 9],
    'RAG Solution': [6, 7, 9, 2, 3]
}

df_challenges = pd.DataFrame(challenges)

fig4 = px.bar(df_challenges, x='Challenge', y=['Impact', 'RAG Solution'], 
              title='Challenges and RAG Solutions',
              barmode='group',
              color_discrete_map={'Impact': 'red', 'RAG Solution': 'green'})

st.plotly_chart(fig4, use_container_width=True)

# Haystack Framework Benefits
st.markdown("## 🌾 Why Haystack?")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🔧 Framework Benefits:
    - **Modular Architecture**: Combine components flexibly
    - **Production Ready**: Battle-tested in enterprise environments
    - **Extensive Integrations**: Major model providers and databases
    - **Custom Components**: Add domain-specific logic
    - **Observability**: Built-in monitoring and debugging
    """)

with col2:
    st.markdown("""
    ### 📊 Use Cases:
    - Document Q&A systems
    - Knowledge base search
    - Content generation
    - Data analysis workflows
    - Multi-modal applications
    """)

# Next Steps
st.markdown("---")
st.markdown("## 🚀 Next Steps")
st.info("""
In the next chapter, we'll explore how to evaluate and optimize RAG systems for better performance, 
including metrics, testing strategies, and pipeline optimizations.
""")

# Sidebar
with st.sidebar:
    st.markdown("## 📖 Chapter 1 Summary")
    st.markdown("""
    ### Key Concepts:
    - LLM fundamentals
    - RAG architecture
    - Compound AI systems
    - Haystack framework
    
    ### Learning Outcomes:
    ✅ Understand RAG vs traditional LLMs  
    ✅ Know RAG system components  
    ✅ Recognize enterprise challenges  
    ✅ Appreciate Haystack benefits  
    """)
    
    st.markdown("### 💡 Key Takeaway")
    st.success("RAG bridges the gap between general LLMs and domain-specific applications by dynamically incorporating relevant context.")

