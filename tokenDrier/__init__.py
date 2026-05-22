from .config import DrierConfig
from .compressor import TokenDrierEngine

class TokenDrier:
    def __init__(self, **kwargs):
        self.config = DrierConfig(**kwargs)
        self.engine = TokenDrierEngine(self.config)
    
    def dry(self, text: str, target_ratio: float = None) -> str:
        return self.engine.compress(text, target_ratio=target_ratio)
