import tiktoken
import re
from typing import List, Optional
from .config import DrierConfig
from .parser import TextNode

class TokenDrierEngine:
    def __init__(self, config: DrierConfig):
        self.config = config
        self.target_encoder = tiktoken.get_encoding(config.target_tokenizer)
        
        if self.config.enable_dedup:
            from .dedup import OpenHumanDeduplicator
            self.deduplicator = OpenHumanDeduplicator(
                similarity_threshold=config.dedup_threshold,
                min_sentence_length=config.min_sentence_length,
                merge_duplicates=config.merge_duplicates,
                min_duplicate_count=config.min_duplicate_count
            )

    def allocate_budgets(self, nodes: List[TextNode], global_target_ratio: float) -> List[TextNode]:
        total_tokens = 0
        protected_tokens = 0
        compressible_nodes = []
        
        for node in nodes:
            node.original_token_count = len(self.target_encoder.encode(node.text))
            total_tokens += node.original_token_count
            
            if node.priority == 5 or (self.config.keep_system_prompt and node.node_type == "system"):
                node.target_token_count = node.original_token_count
                protected_tokens += node.original_token_count
            else:
                compressible_nodes.append(node)
        
        if not compressible_nodes:
            return nodes
        
        remaining_budget = int(total_tokens * global_target_ratio) - protected_tokens
        
        if remaining_budget <= 0:
            for node in compressible_nodes:
                node.target_token_count = max(1, int(node.original_token_count * 0.1))
            return nodes
        
        compressible_tokens = sum(node.original_token_count for node in compressible_nodes)
        ratio = remaining_budget / compressible_tokens
        
        for node in compressible_nodes:
            node.target_token_count = max(1, int(node.original_token_count * ratio))
        
        return nodes

    def _score_sentence(self, sentence: str) -> float:
        score = 0.0
        
        if any(kw.lower() in sentence.lower() for kw in self.config.protected_keywords):
            score += 10.0
        
        score += len([w for w in sentence.split() if w.istitle()]) * 0.5
        
        score += len([w for w in sentence.split() if len(w) > 6]) * 0.3
        
        if len(sentence) < 20:
            score -= 2.0
        
        return max(0, score)

    def dry_node(self, node: TextNode) -> str:
        if node.priority == 5 or (self.config.keep_system_prompt and node.node_type == "system"):
            return node.text
        
        sentences = re.split(r'(?<=[.!?])\s+', node.text)
        
        scored_sentences = [(sentence, self._score_sentence(sentence)) for sentence in sentences]
        
        scored_sentences.sort(key=lambda x: -x[1])
        
        target_chars = int(len(node.text) * (node.target_token_count / node.original_token_count)) if node.original_token_count > 0 else 0
        
        result = []
        current_length = 0
        
        for sentence, score in scored_sentences:
            if current_length + len(sentence) <= target_chars or score >= 5.0:
                result.append(sentence)
                current_length += len(sentence) + 1
        
        result.sort(key=lambda s: node.text.find(s))
        
        return ' '.join(result).strip()

    def compress(self, raw_text: str, target_ratio: Optional[float] = None) -> str:
        if len(raw_text) < self.config.min_context_length:
            return raw_text
        
        ratio = target_ratio if target_ratio is not None else self.config.default_compression_ratio
        
        from .parser import ContextParser
        nodes = ContextParser.parse_rag_prompt(raw_text)
        
        if self.config.enable_dedup:
            nodes = self.deduplicator.deduplicate_nodes(nodes)
        
        nodes = self.allocate_budgets(nodes, ratio)
        
        compressed_blocks = []
        for node in nodes:
            compressed_text = self.dry_node(node)
            compressed_blocks.append(compressed_text)
        
        return "\n\n".join(compressed_blocks)
