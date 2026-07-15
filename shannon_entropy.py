# Required packages: pip install torch transformers


import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def calculate_token_entropy(model, tokenizer, prompt: str) -> float:
    """
    Generates the next token distribution for a given prompt
    and calculates its shannon entropy in bits.
    """

    #Encode the prompt into tensor IDs
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    #pass inputs through the model to get the logit predictions
    with torch.no_grad():
        outputs = model(**inputs)
    #Extract logits for very last token in the input sequence
    next_token_logits = outputs.logits[0, -1, :]

    #Convert raw logits into a clean probability distribution
    probs = F.softmax(next_token_logits, dim=-1)

    #Filter out zeros to avoid log(0) errors resulting in NaN
    non_zero_probs = probs[probs > 0]

    #Calculate Shannon Entropy H = -sum(p * log2(p))
    entropy = -torch.sum(non_zero_probs * torch.log2(non_zero_probs))
    return entropy.item()
