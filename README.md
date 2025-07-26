# 📚 Arxiv Agent - Advanced AI Research Assistant

An intelligent AI research assistant that combines Google Gemini's knowledge with advanced paper retrieval to provide comprehensive, contextually-rich responses about academic research.

## 🌟 Key Features

### 🧠 Hybrid Knowledge System
- **Knowledge-Based Responses**: Uses Gemini's extensive knowledge when no papers are loaded
- **Paper-Enhanced Responses**: Combines general knowledge with specific research findings
- **Intelligent Fallbacks**: Automatically selects the best response strategy

### 🔍 Advanced Semantic Search
- **Multi-Stage Retrieval**: Semantic search + title-based search + keyword matching
- **Query Expansion**: Automatically expands queries with synonyms and related terms
- **Smart Reranking**: Multiple relevance signals with diversity filtering

### 💬 Enhanced User Experience
- **Always Helpful**: Never returns "no papers found" - always provides valuable responses
- **Conversation Context**: Maintains context across multiple questions
- **Multiple Interfaces**: Beautiful Streamlit web UI and powerful CLI

### 📊 Advanced Features
- **Dual-Index Architecture**: Separate semantic and title-based FAISS indices
- **Quality Scoring**: Transparent relevance metrics and scoring
- **Citation Integration**: Proper attribution for paper-specific claims

## 🛠️ Technology Stack

- **LLM**: Google Gemini 1.5 Flash with hybrid knowledge integration
- **Embeddings**: Gemini Embedding Model (`embedding-001`) with dual-index architecture
- **Vector Search**: FAISS with semantic and title-based indices
- **Advanced RAG**: Multi-stage retrieval with intelligent reranking
- **Web UI**: Enhanced Streamlit interface with real-time status
- **Paper API**: Arxiv API with advanced search capabilities

## 📋 Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- Internet connection for Arxiv access

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd arxiv-agent
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your Google API key:
   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

## 🚀 Quick Start

### Web Interface (Recommended)

Start the enhanced Streamlit web application:

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`

**Features Available:**
- 🔍 Search and load papers from Arxiv
- 💬 Chat with hybrid knowledge system
- 📊 View advanced search metrics and scoring
- 📄 Browse paper details and access PDFs
- 🧠 Get comprehensive responses even without papers loaded

### Command Line Interface

Start interactive mode:

```bash
python cli.py --interactive
```

**Available Commands:**
```bash
# Search for papers
python cli.py --search "attention mechanisms transformers"

# Load papers and ask questions
python cli.py --load "large language models" --question "How do LLMs work?"

# Ask questions using hybrid knowledge
python cli.py --question "What are the benefits of self-attention?"
```

## 🎯 Advanced Capabilities

### Hybrid Knowledge System
The agent intelligently combines three response strategies:

1. **Knowledge-Only**: Uses Gemini's comprehensive knowledge when no papers are available
2. **Hybrid (Knowledge Emphasis)**: Combines general knowledge with light paper integration
3. **Hybrid (Paper Emphasis)**: Strong integration of paper findings with knowledge foundation

### Multi-Stage Semantic Search
- **Stage 1**: Semantic search on text chunks using embeddings
- **Stage 2**: Title-based search for paper-level relevance
- **Stage 3**: Keyword matching for exact term coverage
- **Stage 4**: Advanced reranking with multiple signals

### Intelligent Scoring System
- **Semantic Score (50%)**: Cosine similarity from embeddings
- **Keyword Score (30%)**: Exact keyword matching
- **Position Score (20%)**: Earlier content weighted higher
- **Recency Bonus**: Recent papers get additional weighting
- **Diversity Filtering**: Prevents redundant results

## 📖 Usage Guide

### 1. Web Interface

**Without Papers Loaded:**
- Ask general questions about AI, ML, transformers, etc.
- Get comprehensive responses using Gemini's knowledge
- Perfect for learning fundamental concepts

**With Papers Loaded:**
- Search and load specific papers from Arxiv
- Get hybrid responses combining knowledge + paper findings
- Ideal for research-specific insights and recent developments

### 2. Command Line Interface

#### Interactive Mode Commands:
- `search <query>` - Search for papers on Arxiv
- `load <query>` - Load papers into knowledge base
- `papers` - Show currently loaded papers with metrics
- `clear` - Clear the knowledge base
- `status` - Show detailed agent status
- `help` - Show help message
- `quit` - Exit the application

#### Example Workflows:
```bash
# General knowledge query (no papers needed)
python cli.py --question "How does attention mechanism work in transformers?"

# Research-specific query with papers
python cli.py --load "retrieval augmented generation" --question "What are recent RAG improvements?"

# Interactive exploration
python cli.py --interactive
> search transformer attention mechanisms
> load transformer attention mechanisms  
> What are the computational complexity issues?
> How do recent papers address these issues?
```

## 🔍 Example Usage

### Example 1: Researching a Topic

```bash
# Start interactive mode
python cli.py --interactive

# Search for papers on transformers
search transformer models

# Load papers into knowledge base
load transformer models

# Ask questions
What are the main advantages of transformer architecture?
How do transformers compare to RNNs?
What are the latest improvements in transformer models?
```

### Example 2: Quick Analysis

```bash
# Load papers and get immediate analysis
python cli.py --load "machine learning" --question "What are the current trends in ML?"
```

### Example 3: Web Interface Workflow

1. Open the web interface
2. Search for "reinforcement learning"
3. Load papers into knowledge base
4. Ask: "What are the main challenges in RL?"
5. Follow up: "How do these papers address exploration vs exploitation?"

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Arxiv API     │    │   Gemini API    │    │   FAISS Index   │
│                 │    │                 │    │                 │
│ • Paper Search  │───▶│ • LLM (Gemini)  │    │ • Vector Search │
│ • Metadata      │    │ • Embeddings    │    │ • Similarity    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Arxiv Agent Core                            │
│                                                                 │
│ • Paper Retrieval                                              │
│ • Content Extraction                                           │
│ • Text Chunking                                                │
│ • RAG Pipeline                                                 │
│ • Response Generation                                          │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐
│   Web UI        │    │   CLI           │
│   (Streamlit)   │    │   (Terminal)    │
└─────────────────┘    └─────────────────┘
```

## 🔧 Configuration

### Environment Variables

- `GOOGLE_API_KEY`: Your Google Gemini API key (required)
- `ARXIV_EMAIL`: Your email for Arxiv API (optional)

### Customization

You can modify the following parameters in `arxiv_agent.py`:

- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `top_k`: Number of search results (default: 5)
- `max_results`: Maximum papers to search (default: 10)

## 📊 Performance Tips

1. **Search Queries**: Use specific, descriptive search terms
2. **Paper Loading**: Load 5-10 papers for optimal performance
3. **Question Specificity**: Ask specific questions for better responses
4. **Knowledge Base**: Clear the knowledge base when switching topics
5. **API Limits**: Be mindful of Gemini API rate limits

## 🧪 Testing and Validation

### Test Suites Available

```bash
# Test basic RAG improvements
python test_improvements.py

# Test advanced semantic search
python test_semantic_search.py

# Test hybrid knowledge system
python test_hybrid_response.py

# Test installation and setup
python test_installation.py
```

### Performance Metrics

**Response Quality:**
- ✅ Knowledge-only responses: 4000-6000+ characters
- ✅ Hybrid responses: 5000+ characters with citations
- ✅ Technical depth: Research-appropriate detail
- ✅ Response time: 3-6 seconds average

**Search Quality:**
- ✅ Multi-signal scoring: 0.6-0.8 relevance scores
- ✅ Diversity filtering: Multiple papers represented
- ✅ Context awareness: Conversation continuity
- ✅ Citation accuracy: Proper source attribution

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**:
   ```
   ❌ Failed to initialize Arxiv Agent: Google API key is required
   ```
   **Solution**: Set the `GOOGLE_API_KEY` environment variable

2. **No Papers Found**:
   - **Old Behavior**: Returns error message
   - **New Behavior**: Uses knowledge-based response automatically

3. **Embedding Errors**:
   ```
   ❌ Error getting embeddings
   ```
   **Solution**: Check API key, internet connection, and rate limits

4. **Import Errors**:
   ```
   ModuleNotFoundError: No module named 'google.generativeai'
   ```
   **Solution**: Install requirements: `pip install -r requirements.txt`

### Debug Mode

Enable detailed logging:

```python
# In arxiv_agent.py
logging.basicConfig(level=logging.DEBUG)
```

### Performance Tuning

```python
# Adjust search parameters
top_k = 8                    # Number of results
search_threshold = 0.3       # Minimum similarity
max_chunks_per_paper = 3     # Diversity control
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini API for LLM and embedding capabilities
- Arxiv for providing access to research papers
- FAISS for efficient vector search
- Streamlit for the web interface framework

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section
2. Review the logs for error messages
3. Open an issue on GitHub
4. Check the documentation

---

**Happy researching! 🎉** 