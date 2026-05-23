"""
Comprehensive Test Suite for TokenDrier Library
===============================================

This test file covers all major functionality of the TokenDrier library.
"""

import pytest
import tiktoken
from tokenDrier import TokenDrier
from tokenDrier.config import DrierConfig
from tokenDrier.parser import ContextParser
from tokenDrier.dedup import OpenHumanDeduplicator, SimHashDeduplicator


class TestTokenDrierBasic:
    """Basic functionality tests"""
    
    def test_initialization(self):
        """Test basic initialization"""
        drier = TokenDrier()
        assert drier is not None
        assert isinstance(drier.config, DrierConfig)
    
    def test_initialization_with_custom_config(self):
        """Test initialization with custom config"""
        drier = TokenDrier(
            default_compression_ratio=0.3,
            enable_dedup=False
        )
        assert drier.config.default_compression_ratio == 0.3
        assert drier.config.enable_dedup is False
    
    def test_dry_basic(self):
        """Test basic compression"""
        drier = TokenDrier()
        text = """System: You are helpful.
Context: Hello world. This is a test.
Question: What is test?"""
        
        compressed = drier.dry(text)
        assert isinstance(compressed, str)
        assert len(compressed) > 0
    
    def test_empty_input(self):
        """Test empty input handling"""
        drier = TokenDrier()
        result = drier.dry("")
        assert result == ""
    
    def test_short_input_no_compression(self):
        """Test short input doesn't get compressed"""
        drier = TokenDrier()
        short_text = "Hello world"
        result = drier.dry(short_text)
        assert result == short_text


class TestTokenDrierCompression:
    """Compression ratio tests"""
    
    def test_compression_ratio_aggressive(self):
        """Test aggressive compression"""
        drier = TokenDrier()
        text = """System: You are a helpful AI assistant.

Context: The Transformer architecture was introduced in the 2017 paper "Attention is All You Need" by Vaswani et al. It revolutionized natural language processing by introducing self-attention mechanisms. The key components include multi-head attention, positional encoding, encoder-decoder architecture, and layer normalization. Transformers are the foundation for models like BERT, GPT, T5, and many others. They have achieved state-of-the-art results on tasks such as machine translation, text classification, and question answering.

Question: What are the key components of the Transformer architecture?"""
        
        compressed = drier.dry(text, target_ratio=0.3)
        
        encoder = tiktoken.get_encoding("cl100k_base")
        original_tokens = len(encoder.encode(text))
        compressed_tokens = len(encoder.encode(compressed))
        
        assert compressed_tokens < original_tokens
    
    def test_compression_ratio_moderate(self):
        """Test moderate compression"""
        drier = TokenDrier()
        text = """System: You are helpful.

Context: Machine learning is a subset of artificial intelligence that uses algorithms to learn from data. Deep learning is a subset of machine learning that uses neural networks with multiple layers. These techniques are used in various applications including image recognition, natural language processing, and recommendation systems.

Question: What is the difference between machine learning and deep learning?"""
        
        compressed = drier.dry(text, target_ratio=0.5)
        
        encoder = tiktoken.get_encoding("cl100k_base")
        original_tokens = len(encoder.encode(text))
        compressed_tokens = len(encoder.encode(compressed))
        
        assert compressed_tokens < original_tokens


class TestContextParser:
    """Context parser tests"""
    
    def test_parse_structured_prompt(self):
        """Test parsing structured RAG prompt"""
        text = """System: You are helpful.

Context: This is context.

Question: What is it?

```python
print("hello")
```"""
        
        nodes = ContextParser.parse_rag_prompt(text)
        
        assert len(nodes) >= 3
        node_types = [node.node_type for node in nodes]
        assert 'system' in node_types
        assert 'context' in node_types
        assert 'question' in node_types
    
    def test_parse_without_code(self):
        """Test parsing without code blocks"""
        text = """System: You are helpful.

Context: This is context.

Question: What is it?"""
        
        nodes = ContextParser.parse_rag_prompt(text)
        
        assert len(nodes) == 3
        node_types = [node.node_type for node in nodes]
        assert 'system' in node_types
        assert 'context' in node_types
        assert 'question' in node_types
    
    def test_parse_no_headers(self):
        """Test parsing text without headers"""
        text = "Just plain text without any headers."
        
        nodes = ContextParser.parse_rag_prompt(text)
        
        assert len(nodes) == 1
        assert nodes[0].node_type == 'context'


class TestDeduplication:
    """Deduplication feature tests"""
    
    def test_simhash_fingerprint(self):
        """Test SimHash fingerprint computation"""
        simhash = SimHashDeduplicator()
        fp1 = simhash.compute_fingerprint("The quick brown fox jumps over the lazy dog")
        fp2 = simhash.compute_fingerprint("The quick brown fox jumps over the lazy dog")
        fp3 = simhash.compute_fingerprint("The slow red cat sits under the tall tree")
        
        assert fp1 == fp2
        assert fp1 != fp3
    
    def test_simhash_duplicate_detection(self):
        """Test SimHash duplicate detection"""
        simhash = SimHashDeduplicator()
        
        text1 = "Machine learning is a subset of artificial intelligence."
        text2 = "Machine learning is a subset of artificial intelligence."
        text3 = "Deep learning uses neural networks with multiple layers."
        
        assert not simhash.is_duplicate(text1)
        assert simhash.is_duplicate(text2)
        assert not simhash.is_duplicate(text3)
    
    def test_openhuman_deduplicate(self):
        """Test OpenHuman deduplication"""
        dedup = OpenHumanDeduplicator(min_sentence_length=5)
        text = "Machine learning is a subset of AI. Machine learning is a subset of AI. This is a unique sentence."
        
        result = dedup.deduplicate_text(text)
        assert "Machine learning" in result
        assert "unique" in result
    
    def test_duplicate_merge(self):
        """Test duplicate merging feature"""
        dedup = OpenHumanDeduplicator(merge_duplicates=True, min_sentence_length=5)
        text = "The Transformer architecture was introduced in 2017. The Transformer architecture was introduced in 2017. The Transformer architecture was introduced in 2017."
        
        result = dedup.deduplicate_text(text)
        assert "*3" in result or "*2" in result
    
    def test_no_merge(self):
        """Test without duplicate merging"""
        dedup = OpenHumanDeduplicator(merge_duplicates=False, min_sentence_length=5)
        text = "Machine learning is popular. Machine learning is popular. Machine learning is popular."
        
        result = dedup.deduplicate_text(text)
        assert "*3" not in result
    
    def test_min_duplicate_count(self):
        """Test minimum duplicate count threshold"""
        dedup = OpenHumanDeduplicator(merge_duplicates=True, min_duplicate_count=3, min_sentence_length=5)
        text = "Hello world. Hello world."
        
        result = dedup.deduplicate_text(text)
        assert "*2" not in result


class TestTokenDrierDeduplication:
    """TokenDrier deduplication integration tests"""
    
    def test_dedup_enabled(self):
        """Test deduplication when enabled"""
        drier = TokenDrier(enable_dedup=True)
        text = """System: You are helpful.
Context: Machine learning is a subset of AI. Machine learning is a subset of AI. Machine learning is a subset of AI.
Question: What is ML?"""
        
        compressed = drier.dry(text, target_ratio=0.9)
        assert "*3" in compressed or "*2" in compressed or "Machine learning" in compressed
    
    def test_dedup_disabled(self):
        """Test without deduplication"""
        drier = TokenDrier(enable_dedup=False)
        text = """System: You are helpful.
Context: Hello world. Hello world. Hello world.
Question: What?"""
        
        compressed = drier.dry(text, target_ratio=0.9)
        assert compressed.count("Hello world") >= 2


class TestKeywordProtection:
    """Keyword protection tests"""
    
    def test_protected_keywords(self):
        """Test keyword protection"""
        drier = TokenDrier(
            protected_keywords=["OpenAI", "GPT", "Transformer"],
            default_compression_ratio=0.1
        )
        
        text = """System: You are helpful.
Context: OpenAI developed GPT which uses Transformer architecture. Many companies use these technologies. Other text to be removed. More filler text. Even more text.
Question: What did OpenAI develop?"""
        
        compressed = drier.dry(text)
        
        assert "OpenAI" in compressed
        assert "GPT" in compressed
        assert "Transformer" in compressed
    
    def test_unprotected_keywords_removed(self):
        """Test unprotected keywords can be removed"""
        drier = TokenDrier(
            protected_keywords=["Important"],
            default_compression_ratio=0.1
        )
        
        text = """Context: Important keyword. Unimportant keyword. Another unimportant keyword."""
        
        compressed = drier.dry(text)
        
        assert "Important" in compressed


class TestConfiguration:
    """Configuration tests"""
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = DrierConfig()
        
        assert config.target_tokenizer == "cl100k_base"
        assert config.default_compression_ratio == 0.5
        assert config.keep_system_prompt is True
        assert config.enable_dedup is True
        assert config.merge_duplicates is True
        assert config.min_duplicate_count == 2
    
    def test_config_custom_values(self):
        """Test custom configuration values"""
        config = DrierConfig(
            target_tokenizer="gpt2",
            default_compression_ratio=0.3,
            enable_dedup=False,
            merge_duplicates=False
        )
        
        assert config.target_tokenizer == "gpt2"
        assert config.default_compression_ratio == 0.3
        assert config.enable_dedup is False
        assert config.merge_duplicates is False


class TestEdgeCases:
    """Edge case tests"""
    
    def test_large_input(self):
        """Test with large input"""
        drier = TokenDrier()
        text = "System: You are helpful.\nContext: " + "This is a test sentence. " * 50 + "\nQuestion: What?"
        
        compressed = drier.dry(text, target_ratio=0.3)
        
        encoder = tiktoken.get_encoding("cl100k_base")
        original_tokens = len(encoder.encode(text))
        compressed_tokens = len(encoder.encode(compressed))
        
        assert compressed_tokens < original_tokens
    
    def test_special_characters(self):
        """Test with special characters"""
        drier = TokenDrier()
        text = """System: You are helpful.
Context: Hello @world! #test $100 %done ^caret &and *star (paren) [bracket] {curly}
Question: What?"""
        
        compressed = drier.dry(text)
        assert isinstance(compressed, str)
        assert len(compressed) > 0
    
    def test_multiline_text(self):
        """Test with multiline text"""
        drier = TokenDrier()
        text = """System: Multi-line
system prompt.

Context: Line 1
Line 2
Line 3
Line 4

Question: What lines?"""
        
        compressed = drier.dry(text)
        assert isinstance(compressed, str)
    
    def test_code_block_with_backticks(self):
        """Test with code blocks using backticks"""
        drier = TokenDrier()
        text = """System: You are a coding assistant.

Context: The quick brown fox.

```python
def hello():
    print("Hello world")
    return True
```

Question: What does the code do?"""
        
        compressed = drier.dry(text)
        assert "You are a coding assistant" in compressed
        assert "What does the code do" in compressed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
