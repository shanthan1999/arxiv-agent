import os
import json
import time
import logging
import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, Counter
import math

import google.generativeai as genai
import arxiv
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Paper:
    """Data class to store paper information"""
    title: str
    authors: List[str]
    abstract: str
    arxiv_id: str
    published_date: str
    categories: List[str]
    pdf_url: str
    content: Optional[str] = None

@dataclass
class ChunkResult:
    """Data class for individual chunk search results"""
    chunk_text: str
    chunk_index: int
    paper_index: int
    semantic_score: float
    keyword_score: float = 0.0
    position_score: float = 0.0
    combined_score: float = 0.0

@dataclass
class SearchResult:
    """Enhanced data class for search results with multiple scoring signals"""
    paper: Paper
    similarity_score: float
    relevant_chunks: List[str]
    chunk_results: List[ChunkResult] = field(default_factory=list)
    keyword_matches: int = 0
    semantic_diversity: float = 0.0
    recency_score: float = 0.0
    final_score: float = 0.0

class ArxivAgent:
    """Arxiv Agent with RAG capabilities using Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Arxiv Agent"""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        
        # Initialize Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize FAISS indices for multi-stage retrieval
        self.dimension = 768  # Gemini embedding dimension
        self.semantic_index = faiss.IndexFlatIP(self.dimension)  # Main semantic search
        self.title_index = faiss.IndexFlatIP(self.dimension)     # Title-focused search
        
        # Data storage
        self.papers: List[Paper] = []
        self.chunks: List[str] = []
        self.chunk_to_paper: List[int] = []  # Maps chunk index to paper index
        self.chunk_embeddings: Optional[np.ndarray] = None
        self.title_embeddings: Optional[np.ndarray] = None
        
        # Keyword search support
        self.chunk_keywords: List[Set[str]] = []  # Keywords per chunk
        self.paper_keywords: List[Set[str]] = []  # Keywords per paper
        
        logger.info("Arxiv Agent initialized successfully")
    
    def search_papers(self, query: str, max_results: int = 10) -> List[Paper]:
        """Search for papers on Arxiv"""
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for result in search.results():
                paper = Paper(
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    arxiv_id=result.entry_id.split('/')[-1],
                    published_date=result.published.strftime("%Y-%m-%d"),
                    categories=result.categories,
                    pdf_url=result.pdf_url
                )
                papers.append(paper)
                logger.info(f"Found paper: {paper.title}")
            
            return papers
            
        except Exception as e:
            logger.error(f"Error searching papers: {e}")
            return []
    
    def extract_paper_content(self, paper: Paper) -> str:
        """Extract structured content from paper with enhanced formatting"""
        try:
            # Create structured content with clear sections
            content_sections = []
            
            # Title and metadata section
            content_sections.append(f"PAPER TITLE: {paper.title}")
            content_sections.append(f"AUTHORS: {', '.join(paper.authors)}")
            content_sections.append(f"ARXIV ID: {paper.arxiv_id}")
            content_sections.append(f"PUBLISHED: {paper.published_date}")
            content_sections.append(f"CATEGORIES: {', '.join(paper.categories)}")
            
            # Abstract section with enhanced formatting
            abstract_clean = paper.abstract.replace('\n', ' ').strip()
            content_sections.append(f"ABSTRACT: {abstract_clean}")
            
            # Add key research areas based on categories
            research_areas = self._extract_research_areas(paper.categories)
            if research_areas:
                content_sections.append(f"RESEARCH AREAS: {', '.join(research_areas)}")
            
            # Join all sections
            content = "\n\n".join(content_sections)
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting content from {paper.arxiv_id}: {e}")
            return f"TITLE: {paper.title}\nABSTRACT: {paper.abstract}"
    
    def _extract_research_areas(self, categories: List[str]) -> List[str]:
        """Extract human-readable research areas from arxiv categories"""
        category_mapping = {
            'cs.AI': 'Artificial Intelligence',
            'cs.LG': 'Machine Learning',
            'cs.CL': 'Natural Language Processing',
            'cs.CV': 'Computer Vision',
            'cs.NE': 'Neural Networks',
            'cs.RO': 'Robotics',
            'stat.ML': 'Statistical Machine Learning',
            'math.OC': 'Optimization',
            'cs.IR': 'Information Retrieval',
            'cs.HC': 'Human-Computer Interaction'
        }
        
        areas = []
        for cat in categories:
            if cat in category_mapping:
                areas.append(category_mapping[cat])
            else:
                # Extract main category
                main_cat = cat.split('.')[0] if '.' in cat else cat
                areas.append(main_cat.upper())
        
        return list(set(areas))  # Remove duplicates
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text for hybrid search"""
        # Simple keyword extraction - could be enhanced with NLP libraries
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our',
            'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way',
            'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'this', 'that', 'with',
            'have', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when',
            'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them',
            'well', 'were', 'will', 'paper', 'model', 'method', 'approach', 'using', 'based', 'results',
            'show', 'propose', 'present', 'study', 'work', 'research', 'analysis', 'data', 'performance'
        }
        
        keywords = {word for word in words if word not in stop_words and len(word) > 3}
        return keywords
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        # Basic query expansion - could be enhanced with word embeddings or thesaurus
        expansion_map = {
            'llm': ['large language model', 'language model', 'transformer'],
            'transformer': ['attention mechanism', 'self-attention', 'bert', 'gpt'],
            'rag': ['retrieval augmented generation', 'retrieval-augmented', 'knowledge retrieval'],
            'embedding': ['vector representation', 'dense representation', 'semantic representation'],
            'attention': ['self-attention', 'multi-head attention', 'attention mechanism'],
            'neural': ['deep learning', 'artificial neural network', 'deep neural network'],
            'classification': ['categorization', 'prediction', 'supervised learning'],
            'generation': ['text generation', 'language generation', 'natural language generation'],
            'training': ['learning', 'optimization', 'fine-tuning'],
            'performance': ['accuracy', 'effectiveness', 'results', 'evaluation']
        }
        
        expanded_queries = [query]
        query_lower = query.lower()
        
        for term, expansions in expansion_map.items():
            if term in query_lower:
                for expansion in expansions:
                    expanded_query = query_lower.replace(term, expansion)
                    if expanded_query != query_lower:
                        expanded_queries.append(expanded_query)
        
        return expanded_queries[:3]  # Limit to avoid too many variations
    
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        """Split text into overlapping chunks with improved strategy"""
        chunks = []
        
        # Split by sentences first to maintain semantic coherence
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            # Add sentence to current chunk if it fits
            if len(current_chunk + sentence) <= chunk_size:
                current_chunk += sentence + ". "
            else:
                # Save current chunk and start new one with overlap
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from previous chunk
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + sentence + ". "
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Fallback to character-based chunking if sentence-based fails
        if not chunks:
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk = text[start:end]
                chunks.append(chunk)
                start = end - overlap
                if start >= len(text):
                    break
        
        return chunks
    
    def get_embeddings(self, texts: List[str], task_type: str = "retrieval_document") -> np.ndarray:
        """Enhanced embedding generation with better error handling and batching"""
        try:
            embeddings = []
            batch_size = 10  # Process in smaller batches to avoid rate limits
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = []
                
                for text in batch:
                    # Clean and truncate text if too long
                    clean_text = text.strip()
                    if len(clean_text) > 20000:  # Gemini embedding limit
                        clean_text = clean_text[:20000] + "..."
                    
                    if not clean_text:
                        # Handle empty text
                        clean_text = "No content available"
                    
                    try:
                        result = genai.embed_content(
                            model="models/embedding-001",
                            content=clean_text,
                            task_type=task_type
                        )
                        embedding = result['embedding']
                        batch_embeddings.append(embedding)
                        
                        # Small delay to respect rate limits
                        time.sleep(0.1)
                        
                    except Exception as embed_error:
                        logger.warning(f"Failed to embed text chunk: {str(embed_error)[:100]}...")
                        # Use zero vector as fallback
                        batch_embeddings.append([0.0] * self.dimension)
                
                embeddings.extend(batch_embeddings)
                
                # Longer delay between batches
                if i + batch_size < len(texts):
                    time.sleep(0.5)
            
            return np.array(embeddings, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return np.array([])
    
    def build_knowledge_base(self, papers: List[Paper]) -> None:
        """Build enhanced knowledge base with multiple indices and keyword extraction"""
        try:
            logger.info(f"Building enhanced knowledge base from {len(papers)} papers")
            
            # Reset indices and data
            self.semantic_index = faiss.IndexFlatIP(self.dimension)
            self.title_index = faiss.IndexFlatIP(self.dimension)
            self.chunks = []
            self.chunk_to_paper = []
            self.chunk_keywords = []
            self.paper_keywords = []
            
            # Process papers and extract content
            titles = []
            for i, paper in enumerate(papers):
                # Extract content
                content = self.extract_paper_content(paper)
                paper.content = content
                
                # Extract paper-level keywords
                paper_text = f"{paper.title} {paper.abstract}"
                paper_keywords = self._extract_keywords(paper_text)
                self.paper_keywords.append(paper_keywords)
                
                # Store title for title-based search
                titles.append(paper.title)
                
                # Chunk the content
                chunks = self.chunk_text(content)
                
                # Store chunks and extract keywords
                for chunk in chunks:
                    self.chunks.append(chunk)
                    self.chunk_to_paper.append(i)
                    
                    # Extract chunk-level keywords
                    chunk_keywords = self._extract_keywords(chunk)
                    self.chunk_keywords.append(chunk_keywords)
            
            # Generate embeddings for chunks (semantic search)
            if self.chunks:
                logger.info("Generating semantic embeddings for chunks...")
                chunk_embeddings = self.get_embeddings(self.chunks, task_type="retrieval_document")
                if len(chunk_embeddings) > 0:
                    # Normalize embeddings for cosine similarity
                    faiss.normalize_L2(chunk_embeddings)
                    self.semantic_index.add(chunk_embeddings)
                    self.chunk_embeddings = chunk_embeddings
                    logger.info(f"Added {len(chunk_embeddings)} chunk embeddings to semantic index")
            
            # Generate embeddings for titles (title-focused search)
            if titles:
                logger.info("Generating title embeddings...")
                title_embeddings = self.get_embeddings(titles, task_type="retrieval_document")
                if len(title_embeddings) > 0:
                    faiss.normalize_L2(title_embeddings)
                    self.title_index.add(title_embeddings)
                    self.title_embeddings = title_embeddings
                    logger.info(f"Added {len(title_embeddings)} title embeddings to title index")
            
            self.papers = papers
            logger.info(f"Knowledge base built successfully with {len(self.chunks)} chunks from {len(papers)} papers")
            
        except Exception as e:
            logger.error(f"Error building knowledge base: {e}")
    
    def _calculate_keyword_score(self, query: str, chunk_keywords: Set[str], paper_keywords: Set[str]) -> float:
        """Calculate keyword-based relevance score"""
        query_keywords = self._extract_keywords(query)
        if not query_keywords:
            return 0.0
        
        # Calculate overlap with chunk and paper keywords
        chunk_overlap = len(query_keywords.intersection(chunk_keywords))
        paper_overlap = len(query_keywords.intersection(paper_keywords))
        
        # Normalize by query length
        chunk_score = chunk_overlap / len(query_keywords)
        paper_score = paper_overlap / len(query_keywords)
        
        # Combine scores (chunk keywords weighted higher)
        return (chunk_score * 0.7) + (paper_score * 0.3)
    
    def _calculate_recency_score(self, paper: Paper) -> float:
        """Calculate recency score based on publication date"""
        try:
            pub_date = datetime.strptime(paper.published_date, "%Y-%m-%d")
            current_date = datetime.now()
            days_old = (current_date - pub_date).days
            
            # Exponential decay with half-life of 2 years (730 days)
            recency_score = math.exp(-days_old / 730)
            return recency_score
        except:
            return 0.5  # Default score for unparseable dates
    
    def _multi_stage_retrieval(self, query: str, top_k: int = 20) -> List[ChunkResult]:
        """Multi-stage retrieval combining semantic and keyword search"""
        chunk_results = []
        
        # Stage 1: Semantic search on chunks
        if len(self.chunks) > 0:
            query_embedding = self.get_embeddings([query], task_type="retrieval_query")
            if len(query_embedding) > 0:
                faiss.normalize_L2(query_embedding)
                
                # Search semantic index
                search_k = min(top_k * 2, len(self.chunks))
                scores, indices = self.semantic_index.search(query_embedding, search_k)
                
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.chunks) and score > 0.2:  # Lower threshold for initial retrieval
                        paper_idx = self.chunk_to_paper[idx]
                        chunk_text = self.chunks[idx]
                        
                        # Calculate keyword score
                        keyword_score = self._calculate_keyword_score(
                            query, 
                            self.chunk_keywords[idx], 
                            self.paper_keywords[paper_idx]
                        )
                        
                        # Calculate position score (earlier chunks in paper are more important)
                        paper_chunks = [i for i, p in enumerate(self.chunk_to_paper) if p == paper_idx]
                        position_in_paper = paper_chunks.index(idx) if idx in paper_chunks else 0
                        position_score = 1.0 / (1.0 + position_in_paper * 0.1)
                        
                        chunk_result = ChunkResult(
                            chunk_text=chunk_text,
                            chunk_index=idx,
                            paper_index=paper_idx,
                            semantic_score=float(score),
                            keyword_score=keyword_score,
                            position_score=position_score
                        )
                        chunk_results.append(chunk_result)
        
        # Stage 2: Title-based search for additional relevance
        if len(self.papers) > 0 and self.title_embeddings is not None:
            query_embedding = self.get_embeddings([query], task_type="retrieval_query")
            if len(query_embedding) > 0:
                faiss.normalize_L2(query_embedding)
                
                title_scores, title_indices = self.title_index.search(query_embedding, min(5, len(self.papers)))
                
                # Boost chunks from papers with relevant titles
                title_boost = {}
                for score, idx in zip(title_scores[0], title_indices[0]):
                    if idx < len(self.papers) and score > 0.4:
                        title_boost[idx] = float(score) * 0.2  # 20% boost
                
                # Apply title boost to chunk results
                for chunk_result in chunk_results:
                    if chunk_result.paper_index in title_boost:
                        chunk_result.semantic_score += title_boost[chunk_result.paper_index]
        
        return chunk_results
    
    def _rerank_results(self, chunk_results: List[ChunkResult], query: str) -> List[ChunkResult]:
        """Advanced reranking using multiple signals"""
        # Calculate combined scores
        for chunk_result in chunk_results:
            # Weighted combination of different signals
            combined_score = (
                chunk_result.semantic_score * 0.5 +      # Semantic similarity (50%)
                chunk_result.keyword_score * 0.3 +       # Keyword matching (30%)
                chunk_result.position_score * 0.2        # Position in paper (20%)
            )
            chunk_result.combined_score = combined_score
        
        # Sort by combined score
        chunk_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Apply diversity filtering to avoid too many chunks from same paper
        diverse_results = []
        paper_count = defaultdict(int)
        max_chunks_per_paper = 3
        
        for chunk_result in chunk_results:
            paper_idx = chunk_result.paper_index
            if paper_count[paper_idx] < max_chunks_per_paper:
                diverse_results.append(chunk_result)
                paper_count[paper_idx] += 1
        
        return diverse_results
    
    def search_knowledge_base(self, query: str, top_k: int = 8) -> List[SearchResult]:
        """Advanced semantic search with multi-stage retrieval and reranking"""
        try:
            if not self.papers or not self.chunks:
                return []
            
            # Expand query for better coverage
            expanded_queries = self._expand_query(query)
            all_chunk_results = []
            
            # Search with original and expanded queries
            for q in expanded_queries:
                chunk_results = self._multi_stage_retrieval(q, top_k=15)
                all_chunk_results.extend(chunk_results)
            
            # Remove duplicates and rerank
            unique_chunks = {}
            for chunk_result in all_chunk_results:
                chunk_idx = chunk_result.chunk_index
                if chunk_idx not in unique_chunks or chunk_result.combined_score > unique_chunks[chunk_idx].combined_score:
                    unique_chunks[chunk_idx] = chunk_result
            
            # Rerank all unique results
            reranked_chunks = self._rerank_results(list(unique_chunks.values()), query)
            
            # Group by paper and create SearchResult objects
            paper_results = defaultdict(lambda: {
                'paper': None,
                'chunk_results': [],
                'chunks': [],
                'scores': []
            })
            
            for chunk_result in reranked_chunks:
                paper_idx = chunk_result.paper_index
                paper = self.papers[paper_idx]
                paper_key = paper.arxiv_id
                
                if paper_results[paper_key]['paper'] is None:
                    paper_results[paper_key]['paper'] = paper
                
                paper_results[paper_key]['chunk_results'].append(chunk_result)
                paper_results[paper_key]['chunks'].append(chunk_result.chunk_text)
                paper_results[paper_key]['scores'].append(chunk_result.combined_score)
            
            # Create final SearchResult objects with enhanced scoring
            final_results = []
            for paper_key, paper_data in paper_results.items():
                if not paper_data['chunk_results']:
                    continue
                
                paper = paper_data['paper']
                chunk_results = paper_data['chunk_results']
                
                # Calculate various scores
                max_score = max(paper_data['scores'])
                avg_score = sum(paper_data['scores']) / len(paper_data['scores'])
                
                # Calculate keyword matches
                query_keywords = self._extract_keywords(query)
                paper_idx = next(i for i, p in enumerate(self.papers) if p.arxiv_id == paper_key)
                keyword_matches = len(query_keywords.intersection(self.paper_keywords[paper_idx]))
                
                # Calculate semantic diversity (how diverse are the retrieved chunks)
                if len(chunk_results) > 1:
                    scores = [cr.semantic_score for cr in chunk_results]
                    semantic_diversity = np.std(scores) if len(scores) > 1 else 0.0
                else:
                    semantic_diversity = 0.0
                
                # Calculate recency score
                recency_score = self._calculate_recency_score(paper)
                
                # Final weighted score
                final_score = (
                    max_score * 0.4 +           # Best chunk score (40%)
                    avg_score * 0.3 +           # Average chunk score (30%)
                    (keyword_matches / max(len(query_keywords), 1)) * 0.15 +  # Keyword coverage (15%)
                    recency_score * 0.1 +       # Recency (10%)
                    semantic_diversity * 0.05   # Diversity bonus (5%)
                )
                
                search_result = SearchResult(
                    paper=paper,
                    similarity_score=max_score,  # Keep for backward compatibility
                    relevant_chunks=paper_data['chunks'][:3],
                    chunk_results=chunk_results,
                    keyword_matches=keyword_matches,
                    semantic_diversity=semantic_diversity,
                    recency_score=recency_score,
                    final_score=final_score
                )
                final_results.append(search_result)
            
            # Sort by final score and return top results
            final_results.sort(key=lambda x: x.final_score, reverse=True)
            return final_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in advanced search: {e}")
            return []
    
    def _generate_knowledge_based_response(self, query: str) -> str:
        """Generate response using Gemini's inherent knowledge without retrieval"""
        try:
            prompt = f"""You are an expert AI research assistant with deep knowledge of machine learning, natural language processing, computer vision, and related fields. Answer the following question using your comprehensive understanding of the field.

USER QUESTION: {query}

INSTRUCTIONS:
1. Provide a comprehensive answer based on your knowledge of the field
2. Include technical details, key concepts, and methodologies
3. Explain the current state of research and recent developments
4. Mention important papers, researchers, or milestones when relevant
5. Structure your response clearly with sections if appropriate
6. Use technical language appropriate for researchers while remaining accessible
7. If there are different perspectives or approaches, discuss them
8. Include practical implications and applications where relevant

ANSWER:"""
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.4,  # Slightly higher for more comprehensive coverage
                    top_p=0.9,
                    max_output_tokens=2048
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating knowledge-based response: {e}")
            return f"I encountered an error while generating a response: {str(e)}"
    
    def _generate_hybrid_response(self, query: str, search_results: List[SearchResult]) -> str:
        """Generate hybrid response combining Gemini's knowledge with retrieved papers"""
        try:
            # Prepare context from retrieved papers
            paper_context = ""
            if search_results:
                context_parts = []
                for i, result in enumerate(search_results, 1):
                    paper_context_part = f"""
PAPER {i}: {result.paper.title} ({result.paper.published_date})
Authors: {', '.join(result.paper.authors)}
Key Findings:
{chr(10).join(f"• {chunk[:300]}..." if len(chunk) > 300 else f"• {chunk}" for chunk in result.relevant_chunks[:2])}
                    """.strip()
                    context_parts.append(paper_context_part)
                
                paper_context = "\n\n".join(context_parts)
            
            # Create hybrid prompt that combines knowledge with retrieval
            prompt = f"""You are an expert AI research assistant with comprehensive knowledge of machine learning, NLP, computer vision, and related fields. Answer the user's question by combining your extensive knowledge with specific findings from recent research papers.

USER QUESTION: {query}

APPROACH:
1. First, provide a comprehensive answer based on your knowledge of the field
2. Then, enhance and validate your answer with specific findings from the research papers below
3. Highlight where the papers confirm, extend, or challenge conventional understanding
4. Cite specific papers when referencing their findings

{f'''RECENT RESEARCH PAPERS:
{paper_context}''' if paper_context else 'No specific papers were found, so rely on your comprehensive knowledge.'}

INSTRUCTIONS:
1. Start with a solid foundation based on your knowledge
2. Integrate specific findings from the papers where relevant
3. Use technical language appropriate for researchers
4. Structure your response with clear sections
5. Cite papers when referencing their specific contributions
6. Discuss both established knowledge and recent developments
7. Include practical implications and applications
8. If papers contradict established knowledge, discuss the implications

RESPONSE FORMAT:
- Begin with core concepts and established understanding
- Integrate recent research findings and cite sources
- Discuss implications and future directions
- Maintain technical depth while being accessible

ANSWER:"""
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.35,  # Balanced for knowledge + specificity
                    top_p=0.85,
                    max_output_tokens=2500
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating hybrid response: {e}")
            return f"I encountered an error while generating a response: {str(e)}"
    
    def generate_response(self, query: str, search_results: List[SearchResult]) -> str:
        """Enhanced response generation combining Gemini's knowledge with retrieval"""
        try:
            # Determine response strategy based on available results
            if not search_results or len(search_results) == 0:
                # No retrieval results - use pure knowledge-based response
                logger.info("No search results found, using knowledge-based response")
                return self._generate_knowledge_based_response(query)
            
            elif len(search_results) < 2 or all(result.final_score < 0.4 for result in search_results):
                # Low-quality or few results - use knowledge with light paper integration
                logger.info("Low-quality search results, using hybrid approach with knowledge emphasis")
                return self._generate_hybrid_response(query, search_results)
            
            else:
                # Good retrieval results - use hybrid approach with strong paper integration
                logger.info("Good search results found, using hybrid approach with paper emphasis")
                return self._generate_hybrid_response(query, search_results)
            
        except Exception as e:
            logger.error(f"Error in response generation: {e}")
            return f"I encountered an error while generating a response: {str(e)}"
    
    def chat(self, query: str, search_query: Optional[str] = None, conversation_history: Optional[List[Dict]] = None) -> str:
        """Enhanced chat interface with hybrid knowledge and retrieval"""
        try:
            # If no papers loaded, try to search for relevant papers
            if not self.papers and search_query:
                logger.info(f"Searching for papers with query: {search_query}")
                papers = self.search_papers(search_query, max_results=8)
                if papers:
                    self.build_knowledge_base(papers)
                    logger.info(f"Loaded {len(papers)} papers into knowledge base")
                else:
                    logger.info("No papers found, will use knowledge-based response")
            
            # Enhance query with conversation context if available
            enhanced_query = self._enhance_query_with_context(query, conversation_history)
            
            # Search knowledge base if papers are available
            search_results = []
            if self.papers:
                search_results = self.search_knowledge_base(enhanced_query, top_k=6)
                logger.info(f"Found {len(search_results)} relevant papers for query")
            else:
                logger.info("No papers in knowledge base, will use pure knowledge-based response")
            
            # Generate response using hybrid approach
            response = self.generate_response(query, search_results)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"I encountered an error: {str(e)}"
    
    def _enhance_query_with_context(self, query: str, conversation_history: Optional[List[Dict]]) -> str:
        """Enhance query with conversation context for better retrieval"""
        if not conversation_history or len(conversation_history) < 2:
            return query
        
        # Get last few exchanges for context
        recent_context = []
        for message in conversation_history[-4:]:  # Last 2 exchanges
            if message.get("role") == "user":
                recent_context.append(f"Previous question: {message.get('content', '')}")
            elif message.get("role") == "assistant":
                # Extract key topics from assistant response
                content = message.get('content', '')
                if len(content) > 200:
                    content = content[:200] + "..."
                recent_context.append(f"Previous context: {content}")
        
        if recent_context:
            context_str = " | ".join(recent_context)
            enhanced_query = f"{query} [Context: {context_str}]"
            return enhanced_query
        
        return query
    
    def get_paper_info(self, arxiv_id: str) -> Optional[Paper]:
        """Get detailed information about a specific paper"""
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(search.results())
            
            paper = Paper(
                title=result.title,
                authors=[author.name for author in result.authors],
                abstract=result.summary,
                arxiv_id=result.entry_id.split('/')[-1],
                published_date=result.published.strftime("%Y-%m-%d"),
                categories=result.categories,
                pdf_url=result.pdf_url
            )
            
            return paper
            
        except Exception as e:
            logger.error(f"Error getting paper info for {arxiv_id}: {e}")
            return None 