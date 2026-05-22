import random
from typing import List, Tuple

class SimPPLEvaluator:
    def __init__(self, config):
        self.config = config
    
    def compute_token_losses(self, text: str) -> Tuple[List[str], List[float]]:
        tokens = list(text)
        losses = [0.0]
        
        for i, token in enumerate(tokens[1:], 1):
            if token.isalnum():
                losses.append(random.uniform(0.5, 3.0))
            elif token in '.!?':
                losses.append(random.uniform(2.0, 4.0))
            else:
                losses.append(random.uniform(0.1, 1.0))
        
        return tokens, losses
