#!/usr/bin/env python3
"""
Test script to verify installation and basic functionality
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    try:
        import google.generativeai as genai
        print("✅ google.generativeai imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import google.generativeai: {e}")
        return False
    
    try:
        import arxiv
        print("✅ arxiv imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import arxiv: {e}")
        return False
    
    try:
        import streamlit as st
        print("✅ streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import streamlit: {e}")
        return False
    
    try:
        import faiss
        print("✅ faiss imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import faiss: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import numpy: {e}")
        return False
    
    return True

def test_environment():
    """Test environment configuration"""
    print("\n🔧 Testing environment configuration...")
    
    # Check for .env file
    if os.path.exists('.env'):
        print("✅ .env file found")
    else:
        print("⚠️ .env file not found - you'll need to create one")
    
    # Check for API key
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        print("✅ GOOGLE_API_KEY found in environment")
        print(f"   Key length: {len(api_key)} characters")
    else:
        print("❌ GOOGLE_API_KEY not found in environment")
        print("   Please set your Gemini API key in the .env file")
        return False
    
    return True

def test_basic_functionality():
    """Test basic agent functionality"""
    print("\n🤖 Testing basic agent functionality...")
    
    try:
        from arxiv_agent import ArxivAgent
        print("✅ ArxivAgent imported successfully")
        
        # Try to initialize agent
        agent = ArxivAgent()
        print("✅ ArxivAgent initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize ArxivAgent: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Arxiv Agent Installation Test")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please install requirements:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Test environment
    if not test_environment():
        print("\n❌ Environment tests failed. Please check your configuration.")
        sys.exit(1)
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality tests failed.")
        sys.exit(1)
    
    print("\n🎉 All tests passed! Your installation is ready.")
    print("\nNext steps:")
    print("1. Run the web interface: streamlit run app.py")
    print("2. Or use the CLI: python cli.py --interactive")
    print("3. Or run the examples: python example_improved.py")

if __name__ == "__main__":
    main()