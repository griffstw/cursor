#!/usr/bin/env python3
"""
Document Summarization Agent

A command-line tool for reading and summarizing various document formats
including PDFs and text files.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

import click
from colorama import init, Fore, Style

from document_reader import DocumentReader
from summarizer import TextSummarizer, SummaryConfig

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentAgent:
    """Main class for the document summarization agent."""
    
    def __init__(self, config: Optional[SummaryConfig] = None):
        """
        Initialize the DocumentAgent.
        
        Args:
            config: Configuration for summarization
        """
        self.reader = DocumentReader()
        self.summarizer = TextSummarizer(config)
        self.config = config or SummaryConfig()
    
    def process_document(self, file_path: str, save_output: bool = False, output_dir: str = "summaries") -> dict:
        """
        Process a document: read, analyze, and summarize.
        
        Args:
            file_path: Path to the document
            save_output: Whether to save the summary to a file
            output_dir: Directory to save output files
            
        Returns:
            dict: Processing results
        """
        try:
            print(f"{Fore.BLUE}📄 Processing document: {file_path}{Style.RESET_ALL}")
            
            # Check if file exists and is supported
            if not Path(file_path).exists():
                error_msg = f"File not found: {file_path}"
                print(f"{Fore.RED}❌ {error_msg}{Style.RESET_ALL}")
                return {"error": error_msg}
            
            if not self.reader.is_supported_file(file_path):
                error_msg = f"Unsupported file type: {Path(file_path).suffix}"
                print(f"{Fore.RED}❌ {error_msg}{Style.RESET_ALL}")
                return {"error": error_msg}
            
            # Get file information
            file_info = self.reader.get_file_info(file_path)
            print(f"{Fore.CYAN}📊 File size: {file_info.get('size', 0):,} bytes{Style.RESET_ALL}")
            
            # Read document content
            print(f"{Fore.YELLOW}📖 Reading document...{Style.RESET_ALL}")
            content = self.reader.read_document(file_path)
            
            if not content:
                error_msg = "Failed to extract content from document"
                print(f"{Fore.RED}❌ {error_msg}{Style.RESET_ALL}")
                return {"error": error_msg}
            
            print(f"{Fore.GREEN}✅ Successfully extracted {len(content.split()):,} words{Style.RESET_ALL}")
            
            # Summarize content
            print(f"{Fore.YELLOW}🤖 Generating summary...{Style.RESET_ALL}")
            summary_result = self.summarizer.summarize_text(content)
            
            # Prepare results
            results = {
                "file_path": str(file_path),
                "file_info": file_info,
                "processing_time": datetime.now().isoformat(),
                "summary": summary_result["summary"],
                "key_points": summary_result["key_points"],
                "statistics": {
                    "original_words": summary_result["original_length"],
                    "summary_words": summary_result["word_count"],
                    "compression_ratio": summary_result["compression_ratio"]
                },
                "config": {
                    "method": self.config.method,
                    "max_length": self.config.max_length
                }
            }
            
            # Display results
            self._display_results(results)
            
            # Save output if requested
            if save_output:
                self._save_results(results, output_dir)
            
            return results
        
        except Exception as e:
            error_msg = f"Error processing document: {str(e)}"
            logger.error(error_msg)
            print(f"{Fore.RED}❌ {error_msg}{Style.RESET_ALL}")
            return {"error": error_msg}
    
    def _display_results(self, results: dict):
        """Display summarization results in a formatted way."""
        print(f"\n{Fore.GREEN}{'='*60}")
        print(f"📄 DOCUMENT SUMMARY REPORT")
        print(f"{'='*60}{Style.RESET_ALL}")
        
        print(f"{Fore.CYAN}📁 File: {results['file_path']}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}⏰ Processed: {results['processing_time']}{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}📈 STATISTICS:{Style.RESET_ALL}")
        stats = results['statistics']
        print(f"  • Original length: {stats['original_words']:,} words")
        print(f"  • Summary length: {stats['summary_words']:,} words")
        print(f"  • Compression ratio: {stats['compression_ratio']:.1%}")
        
        print(f"\n{Fore.GREEN}📝 SUMMARY:{Style.RESET_ALL}")
        summary_lines = results['summary'].split('. ')
        for line in summary_lines:
            if line.strip():
                print(f"  {line.strip()}.")
        
        if results['key_points']:
            print(f"\n{Fore.MAGENTA}🔑 KEY POINTS:{Style.RESET_ALL}")
            for i, point in enumerate(results['key_points'], 1):
                print(f"  {i}. {point}")
        
        print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
    
    def _save_results(self, results: dict, output_dir: str):
        """Save results to files."""
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Generate output filename
            input_file = Path(results['file_path'])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{input_file.stem}_summary_{timestamp}"
            
            # Save JSON report
            json_file = output_path / f"{output_filename}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Save readable text summary
            txt_file = output_path / f"{output_filename}.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(f"Document Summary Report\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"File: {results['file_path']}\n")
                f.write(f"Processed: {results['processing_time']}\n")
                f.write(f"Method: {results['config']['method']}\n\n")
                
                f.write(f"Statistics:\n")
                stats = results['statistics']
                f.write(f"  Original length: {stats['original_words']:,} words\n")
                f.write(f"  Summary length: {stats['summary_words']:,} words\n")
                f.write(f"  Compression ratio: {stats['compression_ratio']:.1%}\n\n")
                
                f.write(f"Summary:\n")
                f.write(f"{results['summary']}\n\n")
                
                if results['key_points']:
                    f.write(f"Key Points:\n")
                    for i, point in enumerate(results['key_points'], 1):
                        f.write(f"  {i}. {point}\n")
            
            print(f"{Fore.GREEN}💾 Results saved to:{Style.RESET_ALL}")
            print(f"  📄 {txt_file}")
            print(f"  📊 {json_file}")
        
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            print(f"{Fore.RED}❌ Failed to save results: {str(e)}{Style.RESET_ALL}")


@click.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--method', '-m', type=click.Choice(['extractive', 'abstractive', 'openai']), 
              default='abstractive', help='Summarization method to use')
@click.option('--max-length', '-l', type=int, default=150, 
              help='Maximum length of summary in words')
@click.option('--min-length', type=int, default=50, 
              help='Minimum length of summary in words')
@click.option('--key-points', '-k', type=int, default=5, 
              help='Number of key points to extract')
@click.option('--save', '-s', is_flag=True, 
              help='Save summary to file')
@click.option('--output-dir', '-o', default='summaries', 
              help='Directory to save output files')
@click.option('--openai-key', envvar='OPENAI_API_KEY', 
              help='OpenAI API key (can also use OPENAI_API_KEY env var)')
@click.option('--verbose', '-v', is_flag=True, 
              help='Enable verbose logging')
def main(file_path, method, max_length, min_length, key_points, save, output_dir, openai_key, verbose):
    """
    Document Summarization Agent
    
    Process and summarize documents including PDFs and text files.
    
    Example usage:
    
        python main.py document.pdf
        
        python main.py report.txt --method extractive --save
        
        python main.py paper.pdf --method openai --openai-key YOUR_KEY
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print(f"{Fore.BLUE}🤖 Document Summarization Agent{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'='*40}{Style.RESET_ALL}")
    
    # Create configuration
    config = SummaryConfig(
        method=method,
        max_length=max_length,
        min_length=min_length,
        num_key_points=key_points,
        openai_api_key=openai_key
    )
    
    # Initialize agent
    agent = DocumentAgent(config)
    
    # Process document
    results = agent.process_document(file_path, save_output=save, output_dir=output_dir)
    
    if "error" in results:
        sys.exit(1)
    
    print(f"\n{Fore.GREEN}✅ Processing completed successfully!{Style.RESET_ALL}")


def create_sample_documents():
    """Create sample documents for testing."""
    print("Creating sample documents for testing...")
    
    # Create a sample text file
    sample_text = """
    The Impact of Artificial Intelligence on Modern Society
    
    Artificial Intelligence (AI) has emerged as one of the most significant technological advances of the 21st century, fundamentally transforming how we work, communicate, and solve complex problems. This revolutionary technology encompasses machine learning, deep learning, natural language processing, and computer vision, enabling machines to perform tasks that traditionally required human intelligence.
    
    In the healthcare sector, AI has shown remarkable potential in medical diagnosis, drug discovery, and personalized treatment plans. Machine learning algorithms can analyze vast amounts of medical data to identify patterns and anomalies that might be missed by human doctors, leading to earlier disease detection and improved patient outcomes.
    
    The business world has also been transformed by AI implementation. Companies use AI for customer service chatbots, predictive analytics, supply chain optimization, and automated decision-making processes. These applications have led to increased efficiency, reduced costs, and enhanced customer experiences across various industries.
    
    However, the rapid advancement of AI also raises important ethical and societal concerns. Issues such as job displacement, algorithmic bias, privacy protection, and the concentration of AI power in the hands of a few large corporations require careful consideration and regulation.
    
    Looking toward the future, AI is expected to continue evolving, with developments in areas such as autonomous vehicles, smart cities, and general artificial intelligence. The key to harnessing AI's potential while mitigating its risks lies in responsible development, inclusive governance, and ongoing dialogue between technologists, policymakers, and society at large.
    
    In conclusion, while AI presents unprecedented opportunities for solving global challenges and improving human life, its development and deployment must be guided by ethical principles and a commitment to benefiting all of humanity.
    """
    
    with open("sample_document.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    print("✅ Created sample_document.txt")
    print("\nYou can now test the agent with:")
    print("  python main.py sample_document.txt")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, show help and create sample documents
        print("No arguments provided. Creating sample documents and showing help...\n")
        create_sample_documents()
        print("\n" + "="*50)
        main.main(['--help'], standalone_mode=False)
    else:
        main()