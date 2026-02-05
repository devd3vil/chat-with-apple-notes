"""
RAG chain: orchestrates retrieval, prompt building, and LLM generation.

Handles the end-to-end RAG pipeline with citation extraction.
"""

import re
from src.models import Citation, QAResult
from src.retriever import Retriever
from src.llm import LLM


class RAGChain:
    """RAG chain that retrieves and generates answers with citations."""
    
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        top_k: int = 5,
        min_score: float = 0.8,
    ):
        """
        Initialize RAG chain.
        
        Args:
            retriever: Retriever for fetching relevant chunks.
            llm: Language model for generating answers.
            top_k: Number of chunks to retrieve per query.
        """
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score
    
    def ask(self, query: str) -> QAResult:
        """
        Answer a question using RAG.
        
        Process:
        1. Retrieve top-k relevant chunks
        2. Build a prompt with chunks and query
        3. Generate answer with LLM
        4. Extract and validate citations from answer
        5. Compute confidence score
        
        Args:
            query: User query.
            
        Returns:
            QAResult with answer, citations, and confidence.
            
        Raises:
            ValueError: If query is empty or processing fails.
        """
        if not query or not query.strip():
            raise ValueError("Cannot answer empty query")
        
        try:
            # Step 1: Retrieve relevant chunks
            if hasattr(self.retriever, "retrieve_for_generation"):
                citations = self.retriever.retrieve_for_generation(query, top_k=self.top_k)
            else:
                citations = self.retriever.retrieve(query, top_k=self.top_k)
            citations = [c for c in citations if c.score >= self.min_score]

            if not citations:
                return QAResult(
                    query=query,
                    answer="I could not find any relevant information to answer this question.",
                    citations=[],
                    confidence=0.0,
                )
            
            # Step 2: Build prompt
            prompt = self._build_prompt(query, citations)
            
            # Step 3: Generate answer
            answer = self.llm.generate(prompt, max_tokens=512)
            
            # Step 4: Extract citations from answer
            extracted_citations = self._extract_citations(answer, citations)
            if citations and not extracted_citations:
                answer = self._fallback_answer(citations)
                extracted_citations = citations[: min(3, len(citations))]
            
            # Step 5: Calculate confidence
            confidence = self._calculate_confidence(answer, citations)
            
            return QAResult(
                query=query,
                answer=answer,
                citations=extracted_citations,
                confidence=confidence,
            )
            
        except Exception as e:
            raise ValueError(f"RAG chain failed: {e}")
    
    def _build_prompt(self, query: str, citations: list[Citation]) -> str:
        """
        Build prompt for LLM with retrieved snippets.
        
        Args:
            query: User query.
            citations: Retrieved chunks with scores.
            
        Returns:
            Formatted prompt.
        """
        snippets_text = "\n".join(
            f"[{i+1}] {citation.text}"
            for i, citation in enumerate(citations)
        )
        
        prompt = f"""You are a Q&A bot for a user's Apple Notes. Use only the provided snippets.
Only answer if the snippets are relevant to the question. If not, say you could not find relevant information in the notes.
Cite your sources using [1], [2], etc. Cite only snippets you used.
Be concise, clear, and directly address the question.

Snippets:
{snippets_text}

Question: {query}

Answer:"""
        
        return prompt
    
    def _extract_citations(
        self,
        answer: str,
        available_citations: list[Citation],
    ) -> list[Citation]:
        """
        Extract and validate citations from LLM answer.
        
        Looks for [1], [2], etc. and returns the corresponding citations.
        
        Args:
            answer: Generated answer text.
            available_citations: Available citations from retrieval.
            
        Returns:
            List of cited snippets, preserving order and removing duplicates.
        """
        # Find all citation indices in answer: [1], [2], etc.
        citation_pattern = r'\[(\d+)\]'
        found_indices = re.findall(citation_pattern, answer)
        
        if not found_indices:
            # No citations found in answer
            return []
        
        # Convert to 0-indexed and filter valid indices
        cited_citations: list[Citation] = []
        seen_indices: set[int] = set()
        
        for idx_str in found_indices:
            try:
                idx = int(idx_str) - 1  # Convert to 0-indexed
                
                # Skip if index out of range or already added
                if idx < 0 or idx >= len(available_citations) or idx in seen_indices:
                    continue
                
                seen_indices.add(idx)
                cited_citations.append(available_citations[idx])
                
            except (ValueError, IndexError):
                continue
        
        return cited_citations
    
    def _calculate_confidence(
        self,
        answer: str,
        retrieved_citations: list[Citation],
    ) -> float:
        """
        Calculate confidence score for the answer.
        
        Factors:
        - Average relevance score of retrieved chunks
        - Whether citations were used
        - Answer length (longer answers with specific content more confident)
        
        Args:
            answer: Generated answer.
            retrieved_citations: Retrieved chunks.
            
        Returns:
            Confidence score [0, 1].
        """
        if not retrieved_citations:
            return 0.0
        
        # Average score of retrieved chunks
        avg_retrieval_score = sum(c.score for c in retrieved_citations) / len(retrieved_citations)
        
        # Check if answer contains citations
        citation_pattern = r'\[(\d+)\]'
        has_citations = bool(re.search(citation_pattern, answer))
        
        # Combine factors
        # Base confidence from retrieval quality
        confidence = avg_retrieval_score
        
        # Penalty if no citations used (-0.1)
        if not has_citations:
            confidence -= 0.1
        
        # Bonus for longer, more detailed answers (+0.1 for answers > 100 chars)
        if len(answer) > 100:
            confidence += 0.05
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))

    def _fallback_answer(self, citations: list[Citation]) -> str:
        """
        Build a fallback answer when the model fails to cite.
        
        Args:
            citations: Retrieved citations to surface.
        
        Returns:
            A minimal extractive answer with citations.
        """
        lines = []
        for idx, citation in enumerate(citations[:3], start=1):
            lines.append(f"[{idx}] {citation.text}")
        snippets = "\n".join(lines)
        return f"Most relevant snippets:\n{snippets}"


class Chunker:
    """Splits notes into overlapping chunks."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Size of each chunk in characters.
            chunk_overlap: Overlap between consecutive chunks in characters.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, note_id: str) -> list["Chunk"]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk.
            note_id: ID of the note for chunk metadata.
            
        Returns:
            List of Chunk objects (without embeddings).
        """
        from src.models import Chunk
        
        chunks: list[Chunk] = []
        
        if not text or len(text) == 0:
            return chunks
        
        chunk_idx = 0
        start_char = 0
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = self.chunk_size
        
        while start_char < len(text):
            # Calculate end position
            end_char = min(start_char + self.chunk_size, len(text))
            
            # Extract chunk
            chunk_text = text[start_char:end_char]
            
            # Create chunk object
            chunk = Chunk(
                id=f"{note_id}_{chunk_idx}",
                note_id=note_id,
                text=chunk_text,
                chunk_idx=chunk_idx,
                start_char=start_char,
                end_char=end_char,
                embedding=None,  # Will be filled later
            )
            
            chunks.append(chunk)
            
            # Stop if we've reached the end of the text
            if end_char >= len(text):
                break
            
            # Move to next chunk with overlap
            start_char += step
            
            chunk_idx += 1
        
        return chunks
