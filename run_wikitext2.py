import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re, math, random, warnings
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cpu"
print("Using device:", device)

print("Loading WikiText-2...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9.,!?;:'\s]", " ", text)
    text = re.sub(r"([.,!?;:])", r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()

def process_split(split):
    tokens = []
    for example in dataset[split]:
        cleaned = clean_text(example["text"])
        if cleaned:
            tokens.extend(cleaned.split())
    return tokens

print("Tokenizing...")
train_tokens = process_split("train")
val_tokens = process_split("validation")
test_tokens = process_split("test")
print(f"Train tokens: {len(train_tokens):,}")
print(f"Val tokens:   {len(val_tokens):,}")
print(f"Test tokens:  {len(test_tokens):,}")

def build_vocab(tokens, min_freq=3):
    counter = Counter(tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, f in counter.items():
        if f >= min_freq:
            vocab[w] = len(vocab)
    return vocab, {i: w for w, i in vocab.items()}, counter

vocab, idx_to_word, _ = build_vocab(train_tokens, min_freq=10)
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size:,}")

def tokens_to_ids(tokens, vocab):
    return [vocab.get(t, vocab["<UNK>"]) for t in tokens]

# Use first 500K train tokens, full val/test
train_ids = tokens_to_ids(train_tokens[:300000], vocab)
val_ids = tokens_to_ids(val_tokens, vocab)
test_ids = tokens_to_ids(test_tokens, vocab)

class LanguageModelDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.token_ids = token_ids
        self.seq_len = seq_len
    def __len__(self):
        return len(self.token_ids) - self.seq_len
    def __getitem__(self, idx):
        return (torch.tensor(self.token_ids[idx:idx+self.seq_len], dtype=torch.long),
                torch.tensor(self.token_ids[idx+self.seq_len], dtype=torch.long))

SEQ_LEN = 20
BATCH_SIZE = 128
train_loader = DataLoader(LanguageModelDataset(train_ids, SEQ_LEN), BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(LanguageModelDataset(val_ids, SEQ_LEN), BATCH_SIZE)
test_loader = DataLoader(LanguageModelDataset(test_ids, SEQ_LEN), BATCH_SIZE)
print(f"Train batches: {len(train_loader):,}")
print(f"Val batches:   {len(val_loader):,}")
print(f"Test batches:  {len(test_loader):,}")

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        out, _ = self.lstm(self.embedding(x))
        return self.fc(out[:, -1, :])

model = LSTMLanguageModel(vocab_size, 128, 256, num_layers=2, dropout=0.3)

print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

def train_one_epoch(model, loader):
    model.train()
    total = 0
    for bx, by in loader:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        total += loss.item()
    return total / len(loader)

def evaluate(loader):
    model.eval()
    total = 0
    with torch.no_grad():
        for bx, by in loader:
            total += criterion(model(bx), by).item()
    return total / len(loader)

EPOCHS = 8
print("\nTraining...")
best_val_loss = float("inf")
best_state = None
patience = 0

for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader)
    val_loss = evaluate(val_loader)
    scheduler.step(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = model.state_dict()
        patience = 0
    else:
        patience += 1
    lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.6f}")
    if patience >= 2:
        print(f"Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(best_state)
test_loss = evaluate(test_loader)
print(f"\nBest Val Loss: {best_val_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Perplexity: {math.exp(test_loss):.2f}")
