import torch
from v1.Model import tokenizer, V1_Base as model


device = "cuda" if torch.cuda.is_available() else "cpu"

text = f"<:User:> {input("User: ")}\n <:Model:> "
x = tokenizer.encode(text)
x = torch.tensor(x).unsqueeze(0).to(device)

temperature = 0.1
generated = []

with torch.no_grad():
    for i in range(100):
        logits = model(x)
        logits = logits[:, -1, :] / temperature

        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)

        x = torch.cat((x, next_token), dim=-1)
        generated.append(next_token.item())

print(tokenizer.decode(generated))