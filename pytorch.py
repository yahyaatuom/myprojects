import torch

soft = torch.nn.Softmax(dim=-1)
logits = torch.tensor([[1.0,3.0,0.5,1.5],[-1.0,2.0,1.0,0.0]])
probabilities = soft(logits)

print(f'Output Probabilities:\n   {probabilities}\n')
print(f'Sum of probabilites for item 1: {probabilities[0].sum()}')