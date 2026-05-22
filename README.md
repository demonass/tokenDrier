# TokenDrier

A High-Performance, Token-Level Context Compression Library for LLMs.

TokenDrier compresses long contexts (prompts, retrieved documents, system instructions) before feeding them into LLMs. It achieves significant compression ratios while preserving semantic integrity.

## Features

- **Structural Parsing**: Automatically identifies System, Context, Question, and Code blocks
- **Smart Budget Allocation**: Prioritizes critical content (System/Question) over background context
- **OpenHuman-style Deduplication**: SimHash-based text fingerprinting for efficient duplicate detection
- **Duplicate Merging**: Merges repeated content as `text *N` format
- **Keyword Protection**: Protects important entities from compression
- **Lightweight**: No heavy ML model dependencies, only `tiktoken` and `pydantic`

## Installation

```bash
pip install tiktoken pydantic
```

## Quick Start

```python
from tokenDrier import TokenDrier

# Initialize
drier = TokenDrier()

# Your RAG prompt
text = """System: You are a helpful assistant.

Context: The Transformer architecture was introduced in 2017...

Question: What is attention mechanism?"""

# Compress to 50% of original tokens
compressed = drier.dry(text, target_ratio=0.5)
```

## Step-by-Step Guide

### 1. Basic Usage

```python
from tokenDrier import TokenDrier

# Initialize with default settings
drier = TokenDrier()

# Sample RAG prompt
prompt = """System: You are a helpful AI assistant.

Context: Machine learning is a subset of AI that uses algorithms to learn from data. Deep learning is a subset of machine learning using neural networks with multiple layers. These techniques are used in image recognition, NLP, and recommendation systems.

Question: What is deep learning?"""

# Compress with default ratio (50%)
compressed = drier.dry(prompt)

print(compressed)
```

### 2. Custom Compression Ratio

```python
# Compress to 30% of original tokens
compressed = drier.dry(prompt, target_ratio=0.3)

# Compress to 70% (less aggressive)
compressed = drier.dry(prompt, target_ratio=0.7)
```

### 3. Enable/Disable Deduplication

```python
# Enable deduplication (default)
drier = TokenDrier(enable_dedup=True)

# Disable deduplication
drier = TokenDrier(enable_dedup=False)
```

### 4. Duplicate Merging (OpenHuman Style)

```python
# Enable duplicate merging (default)
drier = TokenDrier(merge_duplicates=True)

# Text with duplicates
text = """Context: Hello world. Hello world. Hello world."""

# Result: "Hello world. *3"
compressed = drier.dry(text)
```

### 5. Protect Keywords

```python
# Protect important entities
drier = TokenDrier(
    protected_keywords=["OpenAI", "GPT", "Transformer"]
)

# Keywords will be preserved even under heavy compression
compressed = drier.dry(text, target_ratio=0.3)
```

### 6. Advanced Configuration

```python
drier = TokenDrier(
    target_tokenizer="cl100k_base",  # Tokenizer for target LLM
    default_compression_ratio=0.5,    # Default compression ratio
    keep_system_prompt=True,          # Preserve system prompt intact
    enable_dedup=True,                # Enable deduplication
    dedup_threshold=0.85,             # Similarity threshold (0.5-1.0)
    merge_duplicates=True,            # Merge duplicates as "*N"
    min_duplicate_count=2,            # Minimum duplicates to merge
    protected_keywords=[],            # Keywords to protect
    min_context_length=100            # Min length to trigger compression
)
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_tokenizer` | str | "cl100k_base" | Tokenizer for target LLM |
| `default_compression_ratio` | float | 0.5 | Default compression ratio (0.1-0.9) |
| `keep_system_prompt` | bool | True | Preserve system prompt intact |
| `protected_keywords` | list | [] | Keywords to protect from compression |
| `min_context_length` | int | 100 | Minimum text length to trigger compression |
| `enable_dedup` | bool | True | Enable SimHash deduplication |
| `dedup_threshold` | float | 0.85 | Similarity threshold for deduplication |
| `min_sentence_length` | int | 15 | Minimum sentence length for deduplication |
| `merge_duplicates` | bool | True | Merge duplicates as `text *N` |
| `min_duplicate_count` | int | 2 | Minimum duplicates to trigger merge |

## Compression Workflow

1. **Parse**: Split text into structural nodes (System, Context, Question, Code)
2. **Deduplicate**: Remove/merge duplicate sentences using SimHash
3. **Allocate Budgets**: Assign token budgets based on node priority
4. **Compress**: Apply sentence-level scoring and pruning
5. **Reconstruct**: Join compressed blocks into final output

## Example Output

**Before:**
```
System: You are a helpful assistant.

Context: The Transformer was introduced in 2017. The Transformer was introduced in 2017. It uses self-attention.

Question: What is Transformer?
```

**After (with deduplication and merging):**
```
You are a helpful assistant.

What is Transformer?

The Transformer was introduced in 2017. *2 It uses self-attention.
```
## Dependencies

- `tiktoken >= 0.6.0` - Token counting
- `pydantic >= 2.6.0` - Configuration validation

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
