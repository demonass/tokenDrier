import re
import hashlib
from typing import List, Tuple, Optional, Dict

class SimHashDeduplicator:
    def __init__(self, hash_bits: int = 64, similarity_threshold: float = 0.85):
        self.hash_bits = hash_bits
        self.similarity_threshold = similarity_threshold
        self.fingerprints = set()
    
    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = text.split()
        return tokens
    
    def _hash_token(self, token: str) -> int:
        h = hashlib.md5(token.encode('utf-8'))
        return int(h.hexdigest(), 16)
    
    def compute_fingerprint(self, text: str) -> int:
        tokens = self._tokenize(text)
        if not tokens:
            return 0

        weights = {}
        for token in tokens:
            weights[token] = weights.get(token, 0) + 1

        vector = [0] * self.hash_bits
        for token, weight in weights.items():
            h = self._hash_token(token)
            for i in range(self.hash_bits):
                bit = (h >> i) & 1
                if bit == 1:
                    vector[i] += weight
                else:
                    vector[i] -= weight

        fingerprint = 0
        for i in range(self.hash_bits):
            if vector[i] > 0:
                fingerprint |= (1 << i)

        return fingerprint
    
    def _hamming_distance(self, fp1: int, fp2: int) -> int:
        return bin(fp1 ^ fp2).count('1')
    
    def is_duplicate(self, text: str) -> bool:
        fingerprint = self.compute_fingerprint(text)
        
        for existing_fp in self.fingerprints:
            distance = self._hamming_distance(fingerprint, existing_fp)
            similarity = 1 - (distance / self.hash_bits)
            if similarity >= self.similarity_threshold:
                return True
        
        self.fingerprints.add(fingerprint)
        return False
    
    def deduplicate(self, texts: List[str]) -> List[str]:
        self.fingerprints.clear()
        unique_texts = []
        
        for text in texts:
            if not self.is_duplicate(text):
                unique_texts.append(text)
        
        return unique_texts

class OpenHumanDeduplicator:
    def __init__(self, 
                 similarity_threshold: float = 0.85,
                 min_sentence_length: int = 15,
                 max_sentence_length: int = 500,
                 merge_duplicates: bool = True,
                 min_duplicate_count: int = 2):
        self.simhash = SimHashDeduplicator(
            hash_bits=64, 
            similarity_threshold=similarity_threshold
        )
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        self.merge_duplicates = merge_duplicates
        self.min_duplicate_count = min_duplicate_count
    
    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        filtered = []
        for s in sentences:
            s = s.strip()
            if self.min_sentence_length <= len(s) <= self.max_sentence_length:
                filtered.append(s)
        return filtered
    
    def _count_duplicates(self, sentences: List[str]) -> List[Tuple[str, int]]:
        counts: Dict[str, int] = {}
        fingerprint_cache: Dict[str, int] = {}

        def get_fingerprint(text: str) -> int:
            if text not in fingerprint_cache:
                fingerprint_cache[text] = self.simhash.compute_fingerprint(text)
            return fingerprint_cache[text]

        for sentence in sentences:
            fingerprint = get_fingerprint(sentence)

            found = False
            for existing_sentence in list(counts.keys()):
                existing_fp = get_fingerprint(existing_sentence)
                distance = self.simhash._hamming_distance(fingerprint, existing_fp)
                similarity = 1 - (distance / self.simhash.hash_bits)

                if similarity >= self.simhash.similarity_threshold:
                    counts[existing_sentence] += 1
                    found = True
                    break

            if not found:
                counts[sentence] = 1

        return [(sentence, count) for sentence, count in counts.items()]
    
    def _merge_with_counts(self, sentences_with_counts: List[Tuple[str, int]]) -> List[str]:
        result = []
        for sentence, count in sentences_with_counts:
            if self.merge_duplicates and count >= self.min_duplicate_count:
                merged = f"{sentence} *{count}"
                result.append(merged)
            else:
                result.append(sentence)
        return result
    
    def deduplicate_text(self, text: str) -> str:
        sentences = self._split_sentences(text)
        
        if self.merge_duplicates:
            sentences_with_counts = self._count_duplicates(sentences)
            sentences_with_counts.sort(key=lambda x: -x[1])
            unique_sentences = self._merge_with_counts(sentences_with_counts)
        else:
            unique_sentences = self.simhash.deduplicate(sentences)
        
        return ' '.join(unique_sentences)
    
    def deduplicate_nodes(self, nodes: List) -> List:
        self.simhash.fingerprints.clear()
        
        for node in nodes:
            if hasattr(node, 'text') and hasattr(node, 'node_type'):
                if node.node_type == 'context':
                    original_length = len(node.text)
                    node.text = self.deduplicate_text(node.text)
                    if hasattr(node, 'original_token_count'):
                        node.original_token_count = len(node.text.split())
        
        return nodes
