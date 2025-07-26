#!/usr/bin/env python3
"""
Basic example of using the Arxiv Agent
"""

import os
from dotenv import load_dotenv
from arxiv_agent import ArxivAgent

def main():
    """Basic usage example"""
    load_dotenv()
    
    # Initialize agent
    print("🚀 Initializing Arxiv Agent...")
    agent = ArxivAgent()
    
    # Example 1: Ask a general question (uses knowledge-based response)
    print("\n💬 Example 1: General Knowledge Question")
    print("=" * 50)
    query = "What are transformers in machine learning?"
    print(f"Question: {query}")
    
    response = agent.chat(query)
    print(f"\nResponse:\n{response}")
    
    # Example 2: Search and load papers, then ask specific questions
    print("\n\n💬 Example 2: Paper-Specific Questions")
    print("=" * 50)
    
    # Search for papers
    print("🔍 Searching for papers on 'attention mechanisms'...")
    papers = agent.search_papers("attention mechanisms", max_results=3)
    
    if papers:
        print(f"✅ Found {len(papers)} papers")
        
        # Build knowledge base
        agent.build_knowledge_base(papers)
        
        # Ask specific question
        query = "How do attention mechanisms improve model performance?"
        print(f"\nQuestion: {query}")
        
        response = agent.chat(query)
        print(f"\nResponse:\n{response}")
    else:
        print("❌ No papers found")

if __name__ == "__main__":
    main()