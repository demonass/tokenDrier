from pydantic import BaseModel
from typing import List, Literal
import re

class TextNode(BaseModel):
    node_type: Literal["system", "context", "question", "code"]
    text: str
    priority: int
    original_token_count: int = 0
    target_token_count: int = 0

class ContextParser:
    @staticmethod
    def parse_rag_prompt(text: str) -> List[TextNode]:
        nodes = []
        patterns = [
            (r'^System:\s*(.*?)\s*(?=\n\n|\nQuestion:|\nContext:|$)', 'system', 5, re.MULTILINE | re.DOTALL),
            (r'^Question:\s*(.*?)\s*(?=\n\n|\nSystem:|\nContext:|$)', 'question', 5, re.MULTILINE | re.DOTALL),
            (r'^Context:\s*(.*?)\s*(?=\n\n|\nSystem:|\nQuestion:|$)', 'context', 2, re.MULTILINE | re.DOTALL),
            (r'```(\w*)\n(.*?)```', 'code', 4, re.DOTALL),
        ]
        
        remaining_text = text
        
        for pattern, node_type, priority, flags in patterns:
            matches = list(re.finditer(pattern, remaining_text, flags))
            for match in matches:
                content = match.group(1).strip() if node_type != 'code' else match.group(2).strip()
                nodes.append(TextNode(
                    node_type=node_type,
                    text=content,
                    priority=priority
                ))
                remaining_text = remaining_text[:match.start()] + remaining_text[match.end():]
        
        remaining_text = remaining_text.strip()
        if remaining_text:
            nodes.append(TextNode(
                node_type="context",
                text=remaining_text,
                priority=2
            ))
        
        nodes.sort(key=lambda x: [x.node_type != 'system', x.node_type != 'question', x.node_type != 'code', x.priority])
        return nodes
