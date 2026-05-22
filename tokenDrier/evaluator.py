import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple

class PPLEvaluator:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.eval_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(config.eval_model).to(config.device)
        self.model.eval()

    @torch.no_grad()
    def compute_token_losses(self, text: str) -> Tuple[List[str], List[float]]:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.config.device)
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        losses = [0.0] + loss.tolist()
        tokens = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in inputs["input_ids"][0]]
        
        return tokens, losses
