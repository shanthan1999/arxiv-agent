#!/usr/bin/env python3
"""
Example demonstrating the improved RAG and generation capabilities
"""

import os
from dotenv import load_dotenv
from arxiv_agent import ArxivAgent

def main():
    """Demonstrate improved RAG capabilities"""
    load_dotenv()
    
    # Initialize agent
    print("🚀 Initializing Arxiv Agent with improvements...")
    agent = ArxivAgent()
    
    # Search for papers on a specific topic
    topic = "retrieval augmented generation"
    print(f"\n🔍 Searching for papers on '{topic}'...")
    papers = agent.search_papers(topic, max_results=6)
    
    if not papers:
        print("❌ No papers found")
        return
    
    print(f"✅ Found {len(papers)} papers:")
    for i, paper in enumerate(papers, 1):
        print(f"  {i}. {paper.title} ({paper.published_date})")
    
    # Build knowledge base
    print(f"\n📚 Building enhanced knowledge base...")
    agent.build_knowledge_base(papers)
    print(f"✅ Knowledge base built with {len(agent.chunks)} text chunks")
    
    # Demonstrate conversation with context
    print(f"\n💬 Starting conversation with enhanced context awareness...")
    
    conversation_history = []
    
    queries = [
        "What is retrieval augmented generation and how does it work?",
        "What are the main benefits compared to traditional language models?",
        "What challenges and limitations are mentioned in these papers?",
        "How can these challenges be addressed based on the research?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {query}")
        print('='*60)
        
        # Add user message to conversation history
        conversation_history.append({
            "role": "user",
            "content": query
        })
        
        # Get response with conversation context
        response = agent.chat(query, conversation_history=conversation_history)
        
        # Add assistant response to conversation history
        conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        print(response)
        
        # Show some metrics
        print(f"\n📊 Response metrics:")
        print(f"  • Length: {len(response)} characters")
        print(f"  • Contains paper references: {'✅' if 'paper' in response.lower() else '❌'}")
        print(f"  • Contains technical details: {'✅' if len(response) > 500 else '❌'}")
        
        # Brief pause for readability
        input("\nPress Enter to continue to next question...")
    
    print(f"\n🎉 Conversation completed!")
    print(f"\nKey improvements demonstrated:")
    print(f"• Enhanced text chunking preserves sentence boundaries")
    print(f"• Better paper content extraction with structured formatting")
    print(f"• Improved search aggregates multiple chunks per paper")
    print(f"• Enhanced prompts generate more detailed responses")
    print(f"• Conversation context helps with follow-up questions")
    print(f"• Quality filtering ensures relevant results")

if __name__ == "__main__":
    main()