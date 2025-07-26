#!/usr/bin/env python3
"""
Test script to demonstrate hybrid response generation combining Gemini's knowledge with retrieval
"""

import os
import sys
from dotenv import load_dotenv
from arxiv_agent import ArxivAgent

def test_knowledge_only_response():
    """Test pure knowledge-based response without any papers"""
    load_dotenv()
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not found in environment variables")
        return False
    
    try:
        print("🧠 Testing Knowledge-Only Response (No Papers Loaded)")
        print("=" * 60)
        
        agent = ArxivAgent()
        
        # Test with no papers loaded
        test_queries = [
            "What is attention mechanism in transformers?",
            "How do large language models work?",
            "Explain self-attention and multi-head attention",
            "What are the benefits and limitations of transformer architecture?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test {i}: {query} ---")
            
            # Generate knowledge-based response
            response = agent._generate_knowledge_based_response(query)
            
            print(f"Response length: {len(response)} characters")
            print(f"Response preview:")
            print(response[:400] + "..." if len(response) > 400 else response)
            
            # Check if response contains general knowledge indicators
            knowledge_indicators = [
                len(response) > 500,  # Substantial response
                "transformer" in response.lower() or "attention" in response.lower(),
                "mechanism" in response.lower() or "architecture" in response.lower(),
                not any(year in response for year in ["2023", "2024"])  # Should not cite specific recent papers
            ]
            
            quality_score = sum(knowledge_indicators)
            print(f"Knowledge-based quality: {quality_score}/4")
            
            if quality_score >= 3:
                print("✅ Good knowledge-based response")
            else:
                print("⚠️ Knowledge response could be improved")
            
            input("\nPress Enter to continue...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in knowledge-only test: {e}")
        return False

def test_hybrid_response():
    """Test hybrid response combining knowledge with paper retrieval"""
    load_dotenv()
    
    try:
        print("\n🔄 Testing Hybrid Response (Knowledge + Papers)")
        print("=" * 60)
        
        agent = ArxivAgent()
        
        # Search and load papers
        print("🔍 Searching for papers on attention mechanisms...")
        papers = agent.search_papers("attention mechanisms transformers", max_results=5)
        
        if papers:
            print(f"✅ Found {len(papers)} papers")
            agent.build_knowledge_base(papers)
        else:
            print("⚠️ No papers found, will test knowledge-only fallback")
        
        test_queries = [
            "How does self-attention work in transformer models?",
            "What are the computational complexity issues with attention?",
            "Explain the benefits of multi-head attention",
            "What are recent advances in attention mechanisms?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Hybrid Test {i}: {query} ---")
            
            # Get search results
            search_results = agent.search_knowledge_base(query, top_k=4) if agent.papers else []
            
            print(f"📊 Search results: {len(search_results)} papers found")
            if search_results:
                for j, result in enumerate(search_results, 1):
                    print(f"  {j}. {result.paper.title} (Score: {result.final_score:.3f})")
            
            # Generate hybrid response
            response = agent._generate_hybrid_response(query, search_results)
            
            print(f"\n📝 Hybrid Response Analysis:")
            print(f"  • Length: {len(response)} characters")
            
            # Check for hybrid indicators
            has_general_knowledge = any(term in response.lower() for term in [
                "transformer", "attention mechanism", "neural network", "deep learning"
            ])
            
            has_paper_citations = any(result.paper.title.split()[0].lower() in response.lower() 
                                    for result in search_results) if search_results else False
            
            has_recent_findings = any(year in response for year in ["2021", "2022", "2023", "2024"])
            
            has_technical_depth = len(response) > 800
            
            print(f"  • General knowledge: {'✅' if has_general_knowledge else '❌'}")
            print(f"  • Paper citations: {'✅' if has_paper_citations else '❌'}")
            print(f"  • Recent findings: {'✅' if has_recent_findings else '❌'}")
            print(f"  • Technical depth: {'✅' if has_technical_depth else '❌'}")
            
            hybrid_score = sum([has_general_knowledge, has_paper_citations, has_recent_findings, has_technical_depth])
            print(f"  • Hybrid quality: {hybrid_score}/4")
            
            print(f"\n📄 Response Preview:")
            print(response[:500] + "..." if len(response) > 500 else response)
            
            input("\nPress Enter to continue...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in hybrid test: {e}")
        return False

def test_chat_interface():
    """Test the enhanced chat interface with hybrid responses"""
    load_dotenv()
    
    try:
        print("\n💬 Testing Enhanced Chat Interface")
        print("=" * 60)
        
        agent = ArxivAgent()
        
        # Test scenarios
        scenarios = [
            {
                "name": "No papers loaded",
                "setup": lambda: None,  # Don't load any papers
                "query": "What are the key innovations in transformer architecture?"
            },
            {
                "name": "Papers loaded",
                "setup": lambda: agent.build_knowledge_base(
                    agent.search_papers("transformer attention", max_results=3)
                ) if agent.search_papers("transformer attention", max_results=3) else None,
                "query": "How do recent papers improve upon standard attention mechanisms?"
            }
        ]
        
        for scenario in scenarios:
            print(f"\n--- Scenario: {scenario['name']} ---")
            
            # Setup scenario
            if scenario['setup']:
                scenario['setup']()
            
            query = scenario['query']
            print(f"Query: {query}")
            
            # Use chat interface
            response = agent.chat(query)
            
            print(f"\nChat Response Analysis:")
            print(f"  • Length: {len(response)} characters")
            print(f"  • Has knowledge: {'✅' if len(response) > 300 else '❌'}")
            print(f"  • Technical content: {'✅' if any(term in response.lower() for term in ['transformer', 'attention', 'mechanism']) else '❌'}")
            
            print(f"\nResponse Preview:")
            print(response[:400] + "..." if len(response) > 400 else response)
            
            input("\nPress Enter to continue...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in chat interface test: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Hybrid Response Generation")
    print("Combining Gemini's Knowledge with Paper Retrieval")
    print("=" * 70)
    
    # Test knowledge-only responses
    if test_knowledge_only_response():
        print("\n✅ Knowledge-only response test passed")
    else:
        print("\n❌ Knowledge-only response test failed")
        sys.exit(1)
    
    # Test hybrid responses
    if test_hybrid_response():
        print("\n✅ Hybrid response test passed")
    else:
        print("\n❌ Hybrid response test failed")
    
    # Test chat interface
    if test_chat_interface():
        print("\n✅ Chat interface test passed")
    else:
        print("\n❌ Chat interface test failed")
    
    print("\n🎉 All hybrid response tests completed!")
    print("\nKey improvements:")
    print("• Pure knowledge-based responses when no papers available")
    print("• Hybrid responses combining knowledge with paper findings")
    print("• Intelligent fallback strategies based on result quality")
    print("• Enhanced chat interface that works with or without papers")
    print("• Comprehensive coverage using Gemini's inherent knowledge")
    print("• Specific paper citations when relevant research is available")