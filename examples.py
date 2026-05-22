#!/usr/bin/env python3
"""
TokenDrier Library Usage Example
================================

This script demonstrates how to use the TokenDrier library for context compression.
TokenDrier is a high-performance, token-level context compression library for LLMs.
"""

import tiktoken
from tokenDrier import TokenDrier

def print_separator(title):
    """Print a separator with title"""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")

def example_basic_usage():
    """Basic usage example"""
    print_separator("Example 1: Basic Usage")
    
    # Sample RAG prompt
    sample_text = """System: You are a helpful AI assistant specialized in machine learning.

Context: The Transformer architecture was introduced in the 2017 paper "Attention is All You Need" by Vaswani et al. It revolutionized natural language processing by introducing self-attention mechanisms. The key components include multi-head attention, positional encoding, encoder-decoder architecture, and layer normalization. Transformers are the foundation for models like BERT, GPT, T5, and many others. They have achieved state-of-the-art results on tasks such as machine translation, text classification, and question answering.

Question: What are the key components of the Transformer architecture?"""
    
    # Initialize TokenDrier with default settings
    drier = TokenDrier()
    
    # Compress with default ratio (50%)
    compressed = drier.dry(sample_text)
    
    # Calculate statistics
    encoder = tiktoken.get_encoding("cl100k_base")
    original_tokens = len(encoder.encode(sample_text))
    compressed_tokens = len(encoder.encode(compressed))
    
    print(f"Original text:\n{sample_text}")
    print(f"\nCompressed text:\n{compressed}")
    print(f"\nStatistics:")
    print(f"  Original tokens: {original_tokens}")
    print(f"  Compressed tokens: {compressed_tokens}")
    print(f"  Compression ratio: {compressed_tokens/original_tokens:.2f}")
    print(f"  Reduction: {100 - compressed_tokens/original_tokens*100:.1f}%")

def example_custom_ratio():
    """Example with custom compression ratio"""
    print_separator("Example 2: Custom Compression Ratio")
    
    sample_text = """System: You are a helpful assistant.

Context: Machine learning is a subset of artificial intelligence that uses algorithms to learn from data. Deep learning is a subset of machine learning that uses neural networks with multiple layers. These techniques are used in various applications including image recognition, natural language processing, and recommendation systems.

Question: What is the difference between machine learning and deep learning?"""
    
    # Test different compression ratios
    for ratio in [0.3, 0.5, 0.7]:
        drier = TokenDrier()
        compressed = drier.dry(sample_text, target_ratio=ratio)
        
        encoder = tiktoken.get_encoding("cl100k_base")
        original_tokens = len(encoder.encode(sample_text))
        compressed_tokens = len(encoder.encode(compressed))
        
        print(f"\nTarget ratio: {ratio*100}%")
        print(f"Compressed text: {compressed}")
        print(f"Tokens: {compressed_tokens}/{original_tokens} ({100-compressed_tokens/original_tokens*100:.1f}% reduction)")

def example_deduplication():
    """Example demonstrating deduplication features"""
    print_separator("Example 3: Deduplication Features")
    
    # Text with duplicate content
    sample_text = """System: You are a helpful assistant.

Context: Attention mechanisms allow models to focus on relevant parts of input. Attention mechanisms are crucial for transformer models. Attention mechanisms enable better context understanding. Transformers use self-attention to process input sequences. Transformers have encoder and decoder layers. Transformers are widely used in NLP.

Question: What is attention in transformers?"""
    
    # With deduplication (default)
    drier_with_dedup = TokenDrier(enable_dedup=True)
    compressed_with_dedup = drier_with_dedup.dry(sample_text, target_ratio=0.8)
    
    # Without deduplication
    drier_no_dedup = TokenDrier(enable_dedup=False)
    compressed_no_dedup = drier_no_dedup.dry(sample_text, target_ratio=0.8)
    
    encoder = tiktoken.get_encoding("cl100k_base")
    
    print("With deduplication:")
    print(compressed_with_dedup)
    print(f"Tokens: {len(encoder.encode(compressed_with_dedup))}")
    
    print("\nWithout deduplication:")
    print(compressed_no_dedup)
    print(f"Tokens: {len(encoder.encode(compressed_no_dedup))}")

def example_merge_duplicates():
    """Example demonstrating duplicate merging"""
    print_separator("Example 4: Duplicate Merging (OpenHuman Style)")
    
    # Text with exact duplicates
    sample_text = """System: You are a helpful assistant.

Context: The Transformer model was introduced in 2017. The Transformer model was introduced in 2017. The Transformer model was introduced in 2017. It uses self-attention mechanisms. It uses self-attention mechanisms. BERT and GPT are based on transformers.

Question: When was the Transformer model introduced?"""
    
    # With merge
    drier_with_merge = TokenDrier(merge_duplicates=True)
    compressed_with_merge = drier_with_merge.dry(sample_text, target_ratio=0.9)
    
    # Without merge
    drier_no_merge = TokenDrier(merge_duplicates=False)
    compressed_no_merge = drier_no_merge.dry(sample_text, target_ratio=0.9)
    
    print("With duplicate merging:")
    print(compressed_with_merge)
    
    print("\nWithout duplicate merging:")
    print(compressed_no_merge)

def example_protected_keywords():
    """Example demonstrating protected keywords"""
    print_separator("Example 5: Protected Keywords")
    
    sample_text = """System: You are a helpful assistant.

Context: The company OpenAI was founded in 2015. OpenAI developed ChatGPT. The company Google was founded in 1998. Google developed many products. The company Microsoft was founded in 1975. Microsoft acquired GitHub.

Question: Which company developed ChatGPT?"""
    
    # With protected keywords
    drier_protected = TokenDrier(
        protected_keywords=["OpenAI", "Google", "Microsoft"],
        target_ratio=0.4
    )
    compressed_protected = drier_protected.dry(sample_text)
    
    # Without protected keywords
    drier_normal = TokenDrier(target_ratio=0.4)
    compressed_normal = drier_normal.dry(sample_text)
    
    print("With protected keywords (OpenAI, Google, Microsoft):")
    print(compressed_protected)
    
    print("\nWithout protected keywords:")
    print(compressed_normal)

def example_advanced_config():
    """Example demonstrating advanced configuration"""
    print_separator("Example 6: Advanced Configuration")
    
    sample_text = """System: You are an expert in computer science.

Context: Python is a high-level programming language. Python is widely used in data science. Python has a large standard library. Machine learning frameworks like TensorFlow and PyTorch are commonly used with Python. Deep learning requires significant computational resources. GPUs are often used for training deep learning models.

Question: What is Python commonly used for?"""
    
    # Custom configuration
    drier = TokenDrier(
        target_tokenizer="cl100k_base",
        default_compression_ratio=0.5,
        keep_system_prompt=True,
        enable_dedup=True,
        dedup_threshold=0.9,
        merge_duplicates=True,
        min_duplicate_count=2,
        protected_keywords=["Python", "TensorFlow", "PyTorch"],
        min_context_length=50
    )
    
    compressed = drier.dry(sample_text)
    
    print("Original text:\n", sample_text)
    print("\nCompressed with advanced config:\n", compressed)
    
    encoder = tiktoken.get_encoding("cl100k_base")
    print(f"\nOriginal tokens: {len(encoder.encode(sample_text))}")
    print(f"Compressed tokens: {len(encoder.encode(compressed))}")

if __name__ == "__main__":
    print("TokenDrier Library Usage Examples")
    print("="*80)
    
    example_basic_usage()
    example_custom_ratio()
    example_deduplication()
    example_merge_duplicates()
    example_protected_keywords()
    example_advanced_config()
    
    print("\n" + "="*80)
    print(" All examples completed!")
    print("="*80)
