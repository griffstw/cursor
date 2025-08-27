"""
Text Summarization Module

This module provides functionality to summarize text using various approaches
including local transformer models and external APIs.
"""

import os
import logging
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import nltk

# Optional imports - gracefully handle if not available
try:
    from transformers import pipeline, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    pipeline = None
    AutoTokenizer = None
    torch = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


@dataclass
class SummaryConfig:
    """Configuration for text summarization."""
    method: str = "extractive"  # "extractive", "abstractive", "openai"
    max_length: int = 150
    min_length: int = 50
    chunk_size: int = 1000
    chunk_overlap: int = 100
    num_key_points: int = 5
    openai_api_key: Optional[str] = None


class TextSummarizer:
    """A class to summarize text using various methods."""
    
    def __init__(self, config: Optional[SummaryConfig] = None):
        """
        Initialize the TextSummarizer.
        
        Args:
            config: Configuration for summarization
        """
        self.config = config or SummaryConfig()
        self.summarizer = None
        self.tokenizer = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the summarization models."""
        try:
            if self.config.method == "abstractive":
                if not TRANSFORMERS_AVAILABLE:
                    logger.warning("Transformers library not available. Falling back to extractive summarization.")
                    self.config.method = "extractive"
                    return
                
                logger.info("Loading summarization model...")
                # Use a lightweight but effective model
                model_name = "facebook/bart-large-cnn"
                
                # Check if CUDA is available
                device = 0 if torch.cuda.is_available() else -1
                
                self.summarizer = pipeline(
                    "summarization",
                    model=model_name,
                    device=device,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            logger.info("Falling back to extractive summarization")
            self.config.method = "extractive"
    
    def summarize_text(self, text: str) -> Dict[str, Any]:
        """
        Summarize the given text.
        
        Args:
            text: Text to summarize
            
        Returns:
            dict: Summary results including main summary and key points
        """
        if not text or not text.strip():
            return {
                "summary": "No content to summarize.",
                "key_points": [],
                "word_count": 0,
                "original_length": 0
            }
        
        # Clean and preprocess text
        cleaned_text = self._preprocess_text(text)
        
        try:
            if self.config.method == "abstractive" and self.summarizer:
                summary = self._abstractive_summarization(cleaned_text)
            elif self.config.method == "openai":
                summary = self._openai_summarization(cleaned_text)
            else:
                summary = self._extractive_summarization(cleaned_text)
            
            # Extract key points
            key_points = self._extract_key_points(cleaned_text)
            
            return {
                "summary": summary,
                "key_points": key_points,
                "word_count": len(summary.split()),
                "original_length": len(text.split()),
                "compression_ratio": len(summary.split()) / len(text.split()) if text.split() else 0
            }
        
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            return {
                "summary": f"Error generating summary: {str(e)}",
                "key_points": [],
                "word_count": 0,
                "original_length": len(text.split())
            }
    
    def _preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text for summarization.
        
        Args:
            text: Raw text
            
        Returns:
            str: Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers if they exist
        text = re.sub(r'--- Page \d+ ---', '', text)
        
        # Remove very short lines that are likely headers or page artifacts
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 10]
        
        return ' '.join(cleaned_lines).strip()
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks for processing.
        
        Args:
            text: Text to chunk
            
        Returns:
            List[str]: List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.config.chunk_size - self.config.chunk_overlap):
            chunk = ' '.join(words[i:i + self.config.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def _abstractive_summarization(self, text: str) -> str:
        """
        Generate abstractive summary using transformer model.
        
        Args:
            text: Text to summarize
            
        Returns:
            str: Generated summary
        """
        if not self.summarizer:
            return self._extractive_summarization(text)
        
        chunks = self._chunk_text(text)
        chunk_summaries = []
        
        for chunk in chunks:
            try:
                # Ensure chunk is not too long for the model
                if len(chunk.split()) > 500:
                    chunk = ' '.join(chunk.split()[:500])
                
                summary = self.summarizer(
                    chunk,
                    max_length=self.config.max_length,
                    min_length=self.config.min_length,
                    do_sample=False
                )
                
                if summary and len(summary) > 0:
                    chunk_summaries.append(summary[0]['summary_text'])
            
            except Exception as e:
                logger.warning(f"Error summarizing chunk: {str(e)}")
                continue
        
        if not chunk_summaries:
            return self._extractive_summarization(text)
        
        # If we have multiple chunk summaries, combine them
        if len(chunk_summaries) > 1:
            combined_summary = ' '.join(chunk_summaries)
            # Summarize the combined summaries if they're still long
            if len(combined_summary.split()) > self.config.max_length:
                try:
                    final_summary = self.summarizer(
                        combined_summary,
                        max_length=self.config.max_length,
                        min_length=self.config.min_length,
                        do_sample=False
                    )
                    return final_summary[0]['summary_text']
                except:
                    return combined_summary[:self.config.max_length * 5]  # Fallback truncation
            return combined_summary
        
        return chunk_summaries[0]
    
    def _extractive_summarization(self, text: str) -> str:
        """
        Generate extractive summary by selecting important sentences.
        
        Args:
            text: Text to summarize
            
        Returns:
            str: Extractive summary
        """
        try:
            # Split into sentences
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) <= 3:
                return text
            
            # Simple scoring based on sentence length and position
            scored_sentences = []
            
            for i, sentence in enumerate(sentences):
                score = 0
                words = sentence.split()
                
                # Length score (prefer medium-length sentences)
                if 10 <= len(words) <= 30:
                    score += 2
                elif 5 <= len(words) <= 50:
                    score += 1
                
                # Position score (first and last sentences often important)
                if i < 3 or i >= len(sentences) - 3:
                    score += 1
                
                # Keyword score (simple frequency-based)
                word_freq = {}
                for word in words:
                    word_lower = word.lower()
                    if len(word_lower) > 3:
                        word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
                
                score += sum(freq for freq in word_freq.values() if freq > 1)
                
                scored_sentences.append((score, sentence))
            
            # Sort by score and select top sentences
            scored_sentences.sort(reverse=True)
            
            # Select sentences for summary (aim for target length)
            summary_sentences = []
            total_words = 0
            target_words = min(self.config.max_length, len(text.split()) // 3)
            
            for score, sentence in scored_sentences:
                if total_words + len(sentence.split()) <= target_words:
                    summary_sentences.append(sentence)
                    total_words += len(sentence.split())
                
                if len(summary_sentences) >= 5:  # Max 5 sentences
                    break
            
            # Sort selected sentences by original order
            original_order = []
            for sentence in summary_sentences:
                try:
                    original_order.append((sentences.index(sentence), sentence))
                except ValueError:
                    continue
            
            original_order.sort()
            
            return ' '.join([sentence for _, sentence in original_order])
        
        except Exception as e:
            logger.error(f"Error in extractive summarization: {str(e)}")
            # Fallback: return first few sentences
            sentences = text.split('. ')
            return '. '.join(sentences[:3]) + '.'
    
    def _extract_key_points(self, text: str) -> List[str]:
        """
        Extract key points from the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List[str]: List of key points
        """
        try:
            sentences = nltk.sent_tokenize(text)
            
            # Score sentences for being key points
            scored_sentences = []
            
            for sentence in sentences:
                score = 0
                words = sentence.lower().split()
                
                # Look for key point indicators
                key_indicators = [
                    'important', 'key', 'main', 'primary', 'significant', 'crucial',
                    'essential', 'fundamental', 'critical', 'major', 'central',
                    'conclusion', 'result', 'finding', 'discovered', 'shows',
                    'indicates', 'demonstrates', 'proves', 'suggests'
                ]
                
                for indicator in key_indicators:
                    if indicator in words:
                        score += 2
                
                # Prefer sentences with numbers or statistics
                if any(word.isdigit() or '%' in word for word in words):
                    score += 1
                
                # Prefer medium-length sentences
                if 8 <= len(words) <= 25:
                    score += 1
                
                scored_sentences.append((score, sentence))
            
            # Sort and select top key points
            scored_sentences.sort(reverse=True)
            
            key_points = []
            for score, sentence in scored_sentences[:self.config.num_key_points]:
                if score > 0:  # Only include sentences with some relevance
                    key_points.append(sentence.strip())
            
            return key_points
        
        except Exception as e:
            logger.error(f"Error extracting key points: {str(e)}")
            return []
    
    def _openai_summarization(self, text: str) -> str:
        """
        Generate summary using OpenAI API (if available).
        
        Args:
            text: Text to summarize
            
        Returns:
            str: Generated summary
        """
        try:
            import openai
            
            if not self.config.openai_api_key:
                logger.warning("OpenAI API key not provided, falling back to local summarization")
                return self._abstractive_summarization(text)
            
            openai.api_key = self.config.openai_api_key
            
            # Truncate text if too long
            max_chars = 3000
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes documents. Provide a clear, concise summary that captures the main ideas and key points."
                    },
                    {
                        "role": "user",
                        "content": f"Please summarize the following text:\n\n{text}"
                    }
                ],
                max_tokens=self.config.max_length * 2,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Error with OpenAI summarization: {str(e)}")
            return self._abstractive_summarization(text)


def main():
    """Test the TextSummarizer functionality."""
    # Sample text for testing
    sample_text = """
    Artificial Intelligence (AI) has become one of the most transformative technologies of the 21st century.
    It encompasses a broad range of techniques and applications that enable machines to perform tasks that
    typically require human intelligence. Machine learning, a subset of AI, focuses on algorithms that can
    learn and improve from experience without being explicitly programmed. Deep learning, which uses neural
    networks with multiple layers, has been particularly successful in areas such as image recognition,
    natural language processing, and speech recognition. The applications of AI are vast and growing,
    including autonomous vehicles, medical diagnosis, financial trading, and personal assistants.
    However, the development of AI also raises important ethical considerations, including concerns about
    job displacement, privacy, bias in algorithms, and the potential for misuse. As AI continues to advance,
    it is crucial for society to address these challenges while harnessing the technology's potential
    for positive impact.
    """
    
    config = SummaryConfig(method="abstractive", max_length=100)
    summarizer = TextSummarizer(config)
    
    result = summarizer.summarize_text(sample_text)
    
    print("=== DOCUMENT SUMMARY ===")
    print(f"Summary: {result['summary']}")
    print(f"\nKey Points:")
    for i, point in enumerate(result['key_points'], 1):
        print(f"{i}. {point}")
    print(f"\nStatistics:")
    print(f"Original length: {result['original_length']} words")
    print(f"Summary length: {result['word_count']} words")
    print(f"Compression ratio: {result['compression_ratio']:.2%}")


if __name__ == "__main__":
    main()