# Document Summarization Agent 🤖

An intelligent document processing agent that can read various document formats (PDFs, text files) and generate comprehensive summaries with key point extraction.

## Features

- **Multi-format Support**: Process PDFs, text files, markdown, and other common document formats
- **Multiple Summarization Methods**: 
  - Extractive summarization (fast, no external dependencies)
  - Abstractive summarization (using transformer models)
  - OpenAI API integration (optional)
- **Key Point Extraction**: Automatically identifies and extracts important points from documents
- **Smart Text Processing**: Handles encoding detection, text cleaning, and chunking for large documents
- **Beautiful CLI Interface**: Colorful, user-friendly command-line interface with progress indicators
- **Export Options**: Save summaries in both human-readable and JSON formats
- **Comprehensive Statistics**: Word counts, compression ratios, and processing metrics

## Installation

1. **Clone or download this repository**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (automatically handled on first run):
   ```bash
   python -c "import nltk; nltk.download('punkt')"
   ```

## Quick Start

### Basic Usage

```bash
# Summarize a text file
python main.py document.txt

# Summarize a PDF with extractive method
python main.py report.pdf --method extractive

# Generate summary and save to file
python main.py paper.pdf --save --output-dir my_summaries

# Use OpenAI for advanced summarization
python main.py document.pdf --method openai --openai-key YOUR_API_KEY
```

### First Time Setup

If you run the script without arguments, it will create a sample document for testing:

```bash
python main.py
```

This creates `sample_document.txt` that you can use to test the agent.

## Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--method` | `-m` | Summarization method (`extractive`, `abstractive`, `openai`) | `abstractive` |
| `--max-length` | `-l` | Maximum summary length in words | `150` |
| `--min-length` | | Minimum summary length in words | `50` |
| `--key-points` | `-k` | Number of key points to extract | `5` |
| `--save` | `-s` | Save summary to file | `False` |
| `--output-dir` | `-o` | Directory for output files | `summaries` |
| `--openai-key` | | OpenAI API key (or set `OPENAI_API_KEY` env var) | `None` |
| `--verbose` | `-v` | Enable verbose logging | `False` |

## Supported File Formats

- **PDF files** (`.pdf`) - Extracts text from all pages
- **Text files** (`.txt`) - With automatic encoding detection
- **Markdown files** (`.md`)
- **Code files** (`.py`, `.js`, `.html`, `.css`, `.json`)

## Summarization Methods

### 1. Extractive Summarization (Default for fallback)
- **Speed**: ⚡ Very fast
- **Dependencies**: None (uses built-in algorithms)
- **How it works**: Selects the most important sentences from the original text
- **Best for**: Quick summaries, when no internet/GPU is available

### 2. Abstractive Summarization (Recommended)
- **Speed**: 🐌 Moderate (requires model download on first use)
- **Dependencies**: Transformers, PyTorch
- **How it works**: Uses BART model to generate new sentences that capture the meaning
- **Best for**: High-quality, coherent summaries

### 3. OpenAI Integration
- **Speed**: 🌐 Fast (depends on API)
- **Dependencies**: OpenAI API key
- **How it works**: Uses GPT-3.5-turbo for advanced summarization
- **Best for**: Highest quality summaries, when API access is available

## Examples

### Basic Document Processing

```bash
# Process a research paper
python main.py research_paper.pdf --method abstractive --save

# Quick summary of a report
python main.py quarterly_report.txt --method extractive --max-length 100

# Detailed analysis with more key points
python main.py meeting_notes.txt --key-points 10 --max-length 200
```

### Advanced Usage

```bash
# Use OpenAI with environment variable
export OPENAI_API_KEY="your-api-key-here"
python main.py document.pdf --method openai

# Save to custom directory with verbose output
python main.py large_document.pdf --save --output-dir ./reports --verbose

# Process multiple documents (using shell loop)
for file in *.pdf; do
    python main.py "$file" --save --output-dir batch_summaries
done
```

## Output Formats

### Console Output
The agent provides a beautiful, colored console output with:
- File information and processing status
- Summary statistics (word counts, compression ratio)
- Main summary text
- Bullet-pointed key points
- Processing time and configuration used

### Saved Files (when using `--save`)
- **`.txt` file**: Human-readable summary report
- **`.json` file**: Structured data with all results and metadata

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key for GPT-based summarization

### Customization
You can modify the behavior by editing the configuration in the source code:

```python
config = SummaryConfig(
    method="abstractive",      # Summarization method
    max_length=150,           # Maximum summary length
    min_length=50,            # Minimum summary length
    chunk_size=1000,          # Text chunk size for processing
    chunk_overlap=100,        # Overlap between chunks
    num_key_points=5          # Number of key points to extract
)
```

## Performance Tips

1. **For large documents**: Use `extractive` method for faster processing
2. **For better quality**: Use `abstractive` method (downloads ~1.6GB model on first use)
3. **For batch processing**: Consider using `extractive` to avoid GPU memory issues
4. **For best results**: Use `openai` method with a valid API key

## Troubleshooting

### Common Issues

**"No module named 'torch'"**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**"Error reading PDF"**
- Some PDFs may be scanned images. Consider using OCR preprocessing
- Try converting the PDF to text first using other tools

**"CUDA out of memory"**
- Use `extractive` method for large documents
- Process documents in smaller chunks
- Restart the process between large documents

**"OpenAI API error"**
- Check your API key is valid
- Ensure you have sufficient credits
- Check internet connection

## Architecture

```
📁 Document Summarization Agent
├── 📄 main.py              # CLI interface and main application
├── 📄 document_reader.py   # Document reading and text extraction
├── 📄 summarizer.py        # Text summarization logic
├── 📄 requirements.txt     # Python dependencies
└── 📁 summaries/          # Default output directory (created when needed)
```

## Contributing

Feel free to contribute by:
- Adding support for more document formats
- Improving summarization algorithms
- Enhancing the user interface
- Adding new features like document comparison

## License

This project is open source. Feel free to use, modify, and distribute according to your needs.

---

**Happy Summarizing! 📚✨**