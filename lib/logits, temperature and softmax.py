import torch, torch.nn.functional as F

# 01. assume this is a module claculating logits (originally)
logits = torch.tensor([2.0, 1.0, 0.1, -1.0])  # Example logits

# 02. add temperature scaling (assume temperature is 0.7 to stable the softmax)
temperature = 0.7
scaled_logits = logits / temperature

# 03. apply softmax to get probabilities
probabailities = F.softmax(scaled_logits, dim=-1)

# 04. according this to the probabilities, we can sample an index
next_token = torch.multinomial(probabailities, num_samples=1)
