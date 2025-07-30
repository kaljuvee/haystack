# Haystack RAG Demo - Enhanced Interactive Application

A comprehensive Streamlit application demonstrating Retrieval-Augmented Generation (RAG) concepts from the O'Reilly book "Retrieval-Augmented Generation in Production with Haystack" by Skanda Vivek.

## 🚀 New Features & Enhancements

### ✨ Interactive Document Processing
- **Document Upload**: Upload your own PDF documents for analysis
- **Sample Documents**: Three pre-loaded sample documents covering AI systems, ML best practices, and data science methodology
- **Real-time Analysis**: Automatic document analysis with statistics, key terms extraction, and readability metrics
- **Live RAG Demonstration**: Interactive question-answering with your uploaded documents

### 📊 Concrete RAG Evaluation Metrics
- **Comprehensive Evaluation Suite**: Multiple test queries with automated evaluation
- **Advanced Metrics Dashboard**: Precision, recall, coverage, diversity, coherence, and performance metrics
- **Visual Analytics**: Interactive charts showing evaluation results and performance breakdowns
- **Chunk-level Analysis**: Detailed analysis of retrieved document chunks with relevance scores

### 🎯 Enhanced User Experience
- **Streamlined Navigation**: Removed "Chapter" prefixes from page names for cleaner interface
- **Home.py Entry Point**: Application now starts with Home.py instead of app.py
- **Responsive Design**: All pages include interactive document processing capabilities
- **Professional Visualizations**: 50+ interactive Plotly charts and simulators

## 📁 Application Structure

```
haystack-demo/
├── Home.py                           # Main application entry point
├── document_utils.py                 # Document processing utilities
├── requirements.txt                  # Python dependencies
├── sample_documents/                 # Pre-loaded sample documents
│   ├── ai_systems_overview.pdf
│   ├── ml_best_practices.pdf
│   └── data_science_methodology.pdf
└── pages/                           # Streamlit pages
    ├── 01_Introduction.py           # RAG introduction and fundamentals
    ├── 02_RAG_Fundamentals.py       # Core RAG concepts and architecture
    ├── 03_Evaluation_Optimization.py # RAG evaluation and optimization
    ├── 04_Scalable_AI.py           # Scalable AI deployment patterns
    ├── 05_Observable_AI.py         # AI monitoring and observability
    ├── 06_Governance.py            # AI governance and compliance
    └── 07_Advanced_RAG.py          # Advanced RAG techniques
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/kaljuvee/haystack.git
   cd haystack
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run Home.py
   ```

4. **Access the application**
   Open your browser and navigate to `http://localhost:8501`

### Dependencies

The application requires the following Python packages:
- `streamlit>=1.28.0` - Web application framework
- `plotly>=5.15.0` - Interactive visualizations
- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `PyPDF2>=3.0.0` - PDF text extraction
- `scikit-learn>=1.3.0` - Machine learning utilities
- `nltk>=3.8.1` - Natural language processing
- `textstat>=0.7.3` - Text readability analysis

## 📖 Page Overview

### 🏠 Home
- Application overview and navigation guide
- Quick start instructions
- Feature highlights

### 📚 Introduction
- RAG fundamentals and AI revolution timeline
- Interactive document upload and basic RAG demonstration
- Key concepts and learning objectives

### 🧠 RAG Fundamentals
- Core RAG architecture and components
- LLM integration and compound AI systems
- Interactive pipeline demonstration with uploaded documents

### 📊 Evaluation & Optimization
- **Comprehensive evaluation metrics** with concrete implementations
- **Advanced evaluation suite** with multiple test queries
- **Performance visualization** and optimization simulators
- **Interactive RAG evaluation** with real documents

### 🚀 Scalable AI
- Deployment patterns and scalability analysis
- Performance projections based on document characteristics
- Infrastructure recommendations and best practices

### 👁️ Observable AI
- Monitoring dashboards and drift detection
- Real-time performance tracking
- Document complexity analysis and alerting

### ⚖️ Governance
- AI governance frameworks and compliance
- Privacy risk assessment and cost analysis
- Security and regulatory considerations

### 🔬 Advanced RAG
- **Advanced evaluation metrics** with comprehensive analytics
- **Multi-dimensional performance analysis** (retrieval, generation, system performance)
- **Detailed chunk analysis** with relevance scoring
- Cutting-edge RAG techniques and implementations

## 🎮 Interactive Features

### Document Processing
- **Upload Interface**: Drag-and-drop PDF upload with validation
- **Sample Documents**: Pre-loaded documents for immediate testing
- **Text Extraction**: Automatic PDF text extraction and preprocessing
- **Document Analysis**: Statistics, readability scores, and key terms

### RAG Demonstration
- **Query Interface**: Natural language question input
- **Retrieval Simulation**: Semantic similarity-based chunk retrieval
- **Response Generation**: Simulated RAG response with context
- **Metrics Display**: Real-time evaluation metrics and performance scores

### Evaluation Metrics
- **Retrieval Quality**: Precision@K, coverage, diversity scores
- **Response Quality**: Coherence, relevance, completeness metrics
- **System Performance**: Latency, throughput, efficiency measurements
- **Visual Analytics**: Interactive charts and performance breakdowns

## 🚀 Deployment Options

### Streamlit Cloud (Recommended)
1. Fork this repository to your GitHub account
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with one click - no additional configuration needed

### Local Development
```bash
streamlit run Home.py --server.port 8501
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Cloud Platforms
- **Heroku**: Ready for deployment with requirements.txt
- **AWS/GCP/Azure**: Can be containerized and deployed
- **Railway/Render**: Direct GitHub integration supported

## 📊 Technical Specifications

### Performance Metrics
- **Interactive Visualizations**: 50+ Plotly charts and simulators
- **Document Processing**: Supports PDFs up to 200MB
- **Real-time Analysis**: Sub-second response times for most operations
- **Scalable Architecture**: Modular design for easy extension

### Evaluation Capabilities
- **Automated Metrics**: Similarity, coverage, diversity, coherence
- **Visual Analytics**: Performance breakdowns and trend analysis
- **Batch Evaluation**: Multiple query processing with aggregated results
- **Comparative Analysis**: Side-by-side metric comparisons

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Skanda Vivek** - Author of "Retrieval-Augmented Generation in Production with Haystack"
- **O'Reilly Media** - Publisher of the source material
- **Haystack Community** - For the excellent RAG framework
- **Streamlit Team** - For the amazing web app framework

## 📞 Support

For questions, issues, or contributions:
- **GitHub Issues**: [Create an issue](https://github.com/kaljuvee/haystack/issues)
- **Documentation**: Comprehensive inline documentation available
- **Examples**: Sample documents and use cases included

---

**Built with ❤️ using Streamlit, Plotly, and modern RAG techniques**

