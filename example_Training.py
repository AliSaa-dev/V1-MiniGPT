from v1.Model import V1_Base, tokenizer
from datetime import datetime
from torch import optim, nn
import torch
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def LOG(*args):
    print(f"[{datetime.now()}][LOGS] {args}")

LOG("Loading Dataset...")

dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    name="sample-10BT",
    split="train",
    streaming=True
)

LOG("Dataset has been loaded.")

model = V1_Base.to(device)

LOG("Initiate the training loop...")

optimizer = optim.AdamW(V1_Base.parameters(), lr=1e-5)

criterion = nn.CrossEntropyLoss()

CONTEXT_LENGTH = 2048
OPTIMIZER_STEP = 0

model.train()

LOG("Training loop initiated.")
LOG(f"Initiated with {CONTEXT_LENGTH} context length.")
LOG(f"Starting from step {OPTIMIZER_STEP}")

for i, data in enumerate(dataset):
    data = data["text"]
    tokens = tokenizer.encoder(data)

    if not len(tokens) <= CONTEXT_LENGTH:
        start = torch.randint(0, len(tokens) - CONTEXT_LENGTH + 1, (1,)).item()
        tokens = tokens[start:start + CONTEXT_LENGTH]
        tokens = tokens.to(torch.long).to(device)

    x = tokens[:-1].unsqueeze(0)
    y = tokens[1:].unsqueeze(0)

    logits = model(x)
    loss = criterion(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    nn.utils.clip_grad_norm_(model.parameters(), 1)

    if i % 100 == 0:
        LOG(f"Hit {i} steps, Saving...")
        torch.save(model.state_dict(),"V1_Base-BackUp.pt")

LOG(f"Training finished on {len(dataset)} examples.")
LOG("Saving...")

torch.save(model.state_dict(), "V1_Base.pt")
