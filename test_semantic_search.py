#!/usr/bin/env python3
"""
Test script to demonstrate improved semantic search and reranking capabilities
"""

import os
import sys
from dotenv import load_dotenv
from arxiv_agent import ArxivAgent

def test_semantic_search_improvements():
    """Test the enhanced semantic search functionality"""
    load_dotenv()
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not found in environment variables")
        return False
    
    try:
        print("🚀 Initializing Arxiv Agent with enhanced semantic search...")
        agent = ArxivAgent()
        
        # Test with a diverse set of papers
        print("🔍 Searching for papers on 'attention mechanisms in transformers'...")
        papers = agent.search_papers("attention mechanisms in transformers", max_results=8)
        
        if not papers:
            print("❌ No papers found")
            return False
        
        print(f"✅ Found {len(papers)} papers:")
        for i, paper in enumerate(papers, 1):
            print(f"  {i}. {paper.title} ({paper.published_date})")
        
        print("\n📚 Building enhanced knowledge base with multi-stage indexing...")
        agent.build_knowledge_base(papers)
        
        print(f"✅ Knowledge base built:")
        print(f"  • {len(agent.chunks)} text chunks")
        print(f"  • {len(agent.chunk_keywords)} keyword sets")
        print(f"  • Semantic index: {agent.semantic_index.ntotal} vectors")
        print(f"  • Title index: {agent.title_index.ntotal} vectors")
        
        # Test various query types to demonstrate improvements
        test_queries = [
            {
                "query": "How does self-attention work in transformer models?",
                "description": "Direct semantic match"
            },
            {
                "query": "What are the computational complexity issues with attention?",
                "description": "Keyword + semantic combination"
            },
            {
                "query": "Multi-head attention mechanism benefits",
                "description": "Technical term expansion"
            },
            {
                "query": "Attention visualization and interpretability",
                "description": "Related concept search"
            }
        ]
        
        print("\n🔍 Testing enhanced semantic search with different query types...")
        
        for i, test_case in enumerate(test_queries, 1):
            query = test_case["query"]
            description = test_case["description"]
            
            print(f"\n{'='*70}")
            print(f"Test {i}: {description}")
            print(f"Query: '{query}'")
            print('='*70)
            
            # Get search results with enhanced scoring
            search_results = agent.search_knowledge_base(query, top_k=4)
            
            if not search_results:
                print("❌ No results found")
                continue
            
            print(f"📊 Found {len(search_results)} relevant papers:")
            
            for j, result in enumerate(search_results, 1):
                print(f"\nResult {j}:")
                print(f"  📄 Paper: {result.paper.title}")
                print(f"  📅 Published: {result.paper.published_date}")
                print(f"  🎯 Final Score: {result.final_score:.3f}")
                print(f"  🔤 Keyword Matches: {result.keyword_matches}")
                print(f"  📈 Semantic Score: {result.similarity_score:.3f}")
                print(f"  🕒 Recency Score: {result.recency_score:.3f}")
                print(f"  🌈 Diversity: {result.semantic_diversity:.3f}")
                print(f"  📝 Chunks Retrieved: {len(result.relevant_chunks)}")
                
                # Show first chunk preview
                if result.relevant_chunks:
                    preview = result.relevant_chunks[0][:200] + "..." if len(result.relevant_chunks[0]) > 200 else result.relevant_chunks[0]
                    print(f"  💡 Content Preview: {preview}")
            
            # Test response generation with enhanced results
            print(f"\n💬 Generating response with enhanced context...")
            response = agent.generate_response(query, search_results)
            
            print(f"📝 Response Quality Metrics:")
            print(f"  • Length: {len(response)} characters")
            print(f"  • Paper citations: {'✅' if any(result.paper.title.split()[0].lower() in response.lower() for result in search_results) else '❌'}")
            print(f"  • Technical depth: {'✅' if len(response) > 800 else '❌'}")
            print(f"  • Specific findings: {'✅' if 'finding' in response.lower() or 'result' in response.lower() else '❌'}")
            
            print(f"\n📄 Response Preview:")
            print(response[:300] + "..." if len(response) > 300 else response)
            
            input("\nPress Enter to continue to next test...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def test_query_expansion():
    """Test query expansion capabilities"""
    print("\n🔄 Testing Query Expansion...")
    
    try:
        agent = ArxivAgent()
        
        test_queries = [
            "LLM performance",
            "transformer attention",
            "RAG systems",
            "neural classification"
        ]
        
        for query in test_queries:
            expanded = agent._expand_query(query)
            print(f"Original: '{query}'")
            print(f"Expanded: {expanded}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error in query expansion test: {e}")
        return False

def test_keyword_extraction():
    """Test keyword extraction functionality"""
    print("\n🔤 Testing Keyword Extraction...")
    
    try:
        agent = ArxivAgent()
        
        test_texts = [
            "Transformer models use self-attention mechanisms for sequence processing",
            "Large language models demonstrate emergent capabilities in few-shot learning",
            "Retrieval-augmented generation improves factual accuracy in language models"
        ]
        
        for text in test_texts:
            keywords = agent._extract_keywords(text)
            print(f"Text: '{text}'")
            print(f"Keywords: {sorted(keywords)}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error in keyword extraction test: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Enhanced Semantic Search and Reranking")
    print("=" * 60)
    
    # Test main semantic search improvements
    if test_semantic_search_improvements():
        print("\n✅ Semantic search improvements test passed")
    else:
        print("\n❌ Semantic search improvements test failed")
        sys.exit(1)
    
    # Test query expansion
    if test_query_expansion():
        print("✅ Query expansion test passed")
    else:
        print("❌ Query expansion test failed")
    
    # Test keyword extraction
    if test_keyword_extraction():
        print("✅ Keyword extraction test passed")
    else:
        print("❌ Keyword extraction test failed")
    
    print("\n🎉 All semantic search tests completed!")
    print("\nKey improvements demonstrated:")
    print("• Multi-stage retrieval with semantic and title-based search")
    print("• Advanced reranking with multiple relevance signals")
    print("• Query expansion for better coverage")
    print("• Keyword extraction and hybrid search")
    print("• Diversity filtering to avoid redundant results")
    print("• Recency scoring for temporal relevance")
    print("• Enhanced scoring with weighted combinations")
    print("• Position-aware chunk scoring")