from pydantic import BaseModel, Field
from typing import Optional, List

class DrierConfig(BaseModel):
    target_tokenizer: str = Field(default="cl100k_base", description="Tokenizer of the target LLM")
    
    default_compression_ratio: float = Field(default=0.5, ge=0.1, le=0.9, description="Target retention ratio")
    condition_window: int = Field(default=512, description="Sliding window size")
    
    keep_system_prompt: bool = Field(default=True, description="Always preserve system prompt intact")
    protected_keywords: List[str] = Field(default_factory=list, description="Substrings that must never be compressed")
    min_context_length: int = Field(default=100, description="Minimum context length to trigger compression")
    
    enable_dedup: bool = Field(default=True, description="Enable OpenHuman-style deduplication")
    dedup_threshold: float = Field(default=0.85, ge=0.5, le=1.0, description="Similarity threshold for deduplication")
    min_sentence_length: int = Field(default=15, description="Minimum sentence length for deduplication")
    merge_duplicates: bool = Field(default=True, description="Merge duplicate lines with count (e.g., \"line *3\")")
    min_duplicate_count: int = Field(default=2, description="Minimum count to trigger merge")
