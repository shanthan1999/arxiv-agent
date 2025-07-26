import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os
from typing import List, Optional

from arxiv_agent import ArxivAgent, Paper

# Page configuration
st.set_page_config(
    page_title="Arxiv Agent - AI Paper Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .paper-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .similarity-score {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

def initialize_agent():
    """Initialize the Arxiv agent"""
    if 'agent' not in st.session_state:
        try:
            st.session_state.agent = ArxivAgent()
            st.success("✅ Arxiv Agent initialized successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize Arxiv Agent: {str(e)}")
            st.info("Please make sure you have set the GOOGLE_API_KEY environment variable.")
            return None
    return st.session_state.agent

def display_paper(paper: Paper, show_abstract: bool = True):
    """Display a paper in a formatted card"""
    with st.container():
        st.markdown(f"""
        <div class="paper-card">
            <h4>{paper.title}</h4>
            <p><strong>Authors:</strong> {', '.join(paper.authors)}</p>
            <p><strong>Arxiv ID:</strong> {paper.arxiv_id}</p>
            <p><strong>Published:</strong> {paper.published_date}</p>
            <p><strong>Categories:</strong> {', '.join(paper.categories)}</p>
            {f'<p><strong>Abstract:</strong> {paper.abstract}</p>' if show_abstract else ''}
            <p><a href="{paper.pdf_url}" target="_blank">📄 View PDF</a></p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application"""
    st.markdown('<h1 class="main-header">📚 Arxiv Agent - AI Paper Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize agent
    agent = initialize_agent()
    if not agent:
        return
    
    # Sidebar
    st.sidebar.title("🔧 Controls")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_papers' not in st.session_state:
        st.session_state.current_papers = []
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    
    # Search section
    st.sidebar.header("🔍 Search Papers")
    search_query = st.sidebar.text_input(
        "Search for papers:",
        value=st.session_state.search_query,
        placeholder="e.g., 'machine learning', 'transformer models', 'computer vision'"
    )
    
    max_results = st.sidebar.slider("Max results:", 1, 20, 10)
    
    if st.sidebar.button("🔍 Search Papers", type="primary"):
        if search_query.strip():
            with st.spinner("Searching for papers..."):
                papers = agent.search_papers(search_query, max_results)
                if papers:
                    st.session_state.current_papers = papers
                    st.session_state.search_query = search_query
                    st.success(f"Found {len(papers)} papers!")
                else:
                    st.error("No papers found. Try a different search query.")
        else:
            st.error("Please enter a search query.")
    
    # Load papers into knowledge base
    if st.sidebar.button("📚 Load Papers into Knowledge Base"):
        if st.session_state.current_papers:
            with st.spinner("Building knowledge base..."):
                agent.build_knowledge_base(st.session_state.current_papers)
                st.success(f"✅ Loaded {len(st.session_state.current_papers)} papers into knowledge base!")
        else:
            st.error("No papers to load. Please search for papers first.")
    
    # Clear knowledge base
    if st.sidebar.button("🗑️ Clear Knowledge Base"):
        agent.papers = []
        agent.chunks = []
        agent.chunk_to_paper = []
        agent.chunk_keywords = []
        agent.paper_keywords = []
        agent.semantic_index = agent.semantic_index.__class__(agent.dimension)
        agent.title_index = agent.title_index.__class__(agent.dimension)
        agent.chunk_embeddings = None
        agent.title_embeddings = None
        st.session_state.chat_history = []
        st.success("✅ Knowledge base cleared!")
    
    # Knowledge base status
    st.sidebar.header("📊 Knowledge Base Status")
    if agent.papers:
        st.sidebar.success(f"✅ {len(agent.papers)} papers loaded")
        st.sidebar.info(f"📝 {len(agent.chunks)} text chunks indexed")
        st.sidebar.info(f"🔍 {agent.semantic_index.ntotal} semantic vectors")
        st.sidebar.info(f"📋 {agent.title_index.ntotal} title vectors")
        if hasattr(agent, 'chunk_keywords') and agent.chunk_keywords:
            avg_keywords = sum(len(kw) for kw in agent.chunk_keywords) / len(agent.chunk_keywords)
            st.sidebar.info(f"🔤 {avg_keywords:.1f} avg keywords/chunk")
    else:
        st.sidebar.warning("⚠️ No papers loaded")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Papers", "ℹ️ About"])
    
    with tab1:
        st.header("💬 Chat with Arxiv Papers")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        user_query = st.text_area(
            "Ask a question about the papers:",
            placeholder="e.g., 'What are the main findings?', 'How does this compare to other approaches?'",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("💬 Send", type="primary"):
                if user_query.strip():
                    # Add user message to history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_query,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Get response with conversation context
                    with st.spinner("Analyzing papers and generating response..."):
                        response = agent.chat(user_query, conversation_history=st.session_state.chat_history)
                    
                    # Add assistant response to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    st.rerun()
                else:
                    st.error("Please enter a question.")
        
        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
    
    with tab2:
        st.header("📄 Current Papers")
        
        if st.session_state.current_papers:
            st.write(f"Showing {len(st.session_state.current_papers)} papers:")
            
            # Create a DataFrame for better display
            papers_data = []
            for paper in st.session_state.current_papers:
                papers_data.append({
                    "Title": paper.title,
                    "Authors": ", ".join(paper.authors),
                    "Arxiv ID": paper.arxiv_id,
                    "Published": paper.published_date,
                    "Categories": ", ".join(paper.categories),
                    "PDF": paper.pdf_url
                })
            
            df = pd.DataFrame(papers_data)
            st.dataframe(df, use_container_width=True)
            
            # Detailed view
            st.subheader("📋 Detailed View")
            for i, paper in enumerate(st.session_state.current_papers):
                with st.expander(f"{i+1}. {paper.title}"):
                    display_paper(paper, show_abstract=True)
        else:
            st.info("No papers loaded. Use the search function in the sidebar to find papers.")
    
    with tab3:
        st.header("ℹ️ About Arxiv Agent")
        
        st.markdown("""
        ## 🤖 What is Arxiv Agent?
        
        Arxiv Agent is an AI-powered assistant that helps you interact with Arxiv papers using advanced natural language processing and retrieval-augmented generation (RAG).
        
        ## 🚀 Features
        
        - **🔍 Smart Paper Search**: Search for papers on Arxiv using natural language queries
        - **🧠 Hybrid AI Responses**: Combines Gemini's knowledge with paper-specific findings
        - **📊 Advanced Semantic Search**: Multi-stage retrieval with intelligent reranking
        - **💬 Interactive Chat**: Works with or without papers loaded
        - **📄 Paper Management**: View and manage your paper collection
        - **🎯 Knowledge Fallback**: Uses Gemini's expertise when no papers are available
        
        ## 🛠️ How it Works
        
        1. **Search**: Find relevant papers using the search function
        2. **Load**: Add papers to the knowledge base for analysis
        3. **Chat**: Ask questions about the papers and get detailed answers
        4. **Explore**: Browse paper details and access PDFs
        
        ## 🔧 Technical Details
        
        - **LLM**: Google Gemini 1.5 Flash with hybrid knowledge integration
        - **Embeddings**: Gemini Embedding Model with dual-index architecture
        - **Vector Search**: FAISS with semantic and title-based indices
        - **Hybrid RAG**: Combines retrieval with Gemini's inherent knowledge
        - **Advanced Reranking**: Multi-signal scoring with diversity filtering
        
        ## 📝 Usage Tips
        
        - **With Papers**: Search and load papers for research-specific insights
        - **Without Papers**: Ask general questions using Gemini's knowledge
        - **Hybrid Mode**: Get the best of both - general knowledge + specific findings
        - **Follow-up Questions**: Use conversation context for deeper exploration
        - **Topic Switching**: Clear knowledge base when changing research areas
        
        ## 🔑 Setup
        
        Make sure you have set the `GOOGLE_API_KEY` environment variable with your Gemini API key.
        """)

if __name__ == "__main__":
    main() 