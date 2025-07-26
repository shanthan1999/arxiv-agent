#!/usr/bin/env python3
"""
Test script to verify RAG and generation improvements
"""

import os
import sys
from dotenv import load_dotenv
from arxiv_agent import ArxivAgent

def test_improved_rag():
    """Test the improved RAG functionality"""
    load_dotenv()
    
    # Check if API key is available
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not found in environment variables")
        print("Please set your Gemini API key in the .env file")
        return False
    
    try:
        print("🚀 Initializing Arxiv Agent...")
        agent = ArxivAgent()
        
        print("🔍 Searching for papers on 'large language models'...")
        papers = agent.search_papers("large language models", max_results=5)
        
        if not papers:
            print("❌ No papers found")
            return False
        
        print(f"✅ Found {len(papers)} papers")
        for i, paper in enumerate(papers, 1):
            print(f"  {i}. {paper.title}")
        
        print("\n📚 Building knowledge base...")
        agent.build_knowledge_base(papers)
        
        print(f"✅ Knowledge base built with {len(agent.chunks)} chunks")
        
        # Test queries
        test_queries = [
            "What are the main capabilities of large language models?",
            "How do these papers address the limitations of LLMs?",
            "What are the key technical innovations mentioned?",
            "What future research directions are suggested?"
        ]
        
        print("\n💬 Testing improved chat responses...")
        conversation_history = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Query {i}: {query} ---")
            
            # Add user message to history
            conversation_history.append({
                "role": "user",
                "content": query
            })
            
            response = agent.chat(query, conversation_history=conversation_history)
            
            # Add assistant response to history
            conversation_history.append({
                "role": "assistant", 
                "content": response
            })
            
            print(f"Response length: {len(response)} characters")
            print(f"Response preview: {response[:200]}...")
            
            # Check response quality indicators
            quality_indicators = [
                "paper" in response.lower(),
                "arxiv" in response.lower() or "arXiv" in response,
                len(response) > 300,  # Substantial response
                any(paper.title.split()[0].lower() in response.lower() for paper in papers[:3])
            ]
            
            quality_score = sum(quality_indicators)
            print(f"Quality indicators met: {quality_score}/4")
            
            if quality_score >= 3:
                print("✅ Good quality response")
            else:
                print("⚠️ Response could be improved")
        
        print("\n🎉 Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def test_search_quality():
    """Test search and retrieval quality"""
    load_dotenv()
    
    try:
        agent = ArxivAgent()
        
        # Search for papers
        papers = agent.search_papers("transformer attention mechanisms", max_results=3)
        if not papers:
            print("❌ No papers found for search quality test")
            return False
        
        agent.build_knowledge_base(papers)
        
        # Test search quality
        query = "How does attention mechanism work in transformers?"
        search_results = agent.search_knowledge_base(query, top_k=3)
        
        print(f"\n🔍 Search Results for: '{query}'")
        print(f"Found {len(search_results)} relevant results")
        
        for i, result in enumerate(search_results, 1):
            print(f"\nResult {i}:")
            print(f"  Paper: {result.paper.title}")
            print(f"  Similarity: {result.similarity_score:.3f}")
            print(f"  Chunks: {len(result.relevant_chunks)}")
            print(f"  Preview: {result.relevant_chunks[0][:150]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in search quality test: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing RAG and Generation Improvements")
    print("=" * 50)
    
    # Test basic functionality
    if test_improved_rag():
        print("\n" + "=" * 50)
        print("✅ Basic RAG test passed")
    else:
        print("\n" + "=" * 50)
        print("❌ Basic RAG test failed")
        sys.exit(1)
    
    # Test search quality
    if test_search_quality():
        print("\n" + "=" * 50)
        print("✅ Search quality test passed")
    else:
        print("\n" + "=" * 50)
        print("❌ Search quality test failed")
    
    print("\n🎉 All tests completed!")
    print("\nKey improvements implemented:")
    print("• Enhanced text chunking with sentence boundaries")
    print("• Improved paper content extraction with structured formatting")
    print("• Better search result aggregation and filtering")
    print("• Enhanced response generation with detailed prompts")
    print("• Conversation context integration")
    print("• Better embedding handling with rate limiting")
    print("• Quality scoring and relevance filtering")