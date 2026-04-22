import torch
import torch.nn as nn
import torch.optim as optim

word_to_ix = {
    "love": 0,
    "hate": 1,
    "coding": 2,
    "pizza": 3
}

embeddings = nn.Embedding(num_embeddings=4, embedding_dim=2)

data = [
    ("love", "coding", 1.0),
    ("love", "pizza", 0.2),
    ("hate", "pizza", 1.0),
]

def cosine(a, b):
    return nn.functional.cosine_similarity(a, b, dim=0)

optimizer = optim.SGD(embeddings.parameters(), lr=1)

for epoch in range(100):
    total_loss = 0
    
    for w1, w2, target in data:
        idx1 = torch.tensor(word_to_ix[w1])
        idx2 = torch.tensor(word_to_ix[w2])
        v1 = embeddings(idx1)
        v2 = embeddings(idx2)
        sim = cosine(v1, v2)
        loss = (sim - target) ** 2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"epoch: {epoch} loss: {total_loss}")