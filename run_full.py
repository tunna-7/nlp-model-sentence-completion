import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re, math, random, urllib.request, warnings
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

DATA_PATH = "tiny_shakespeare.txt"
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists(DATA_PATH):
    urllib.request.urlretrieve(url, DATA_PATH)
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total characters:", len(raw_text))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9.,!?;:'\s]", " ", text)
    text = re.sub(r"([.,!?;:])", r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()

tokens = clean_text(raw_text).split()
print("Total tokens:", len(tokens))

def build_vocab(tokens, min_freq=2):
    counter = Counter(tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, f in counter.items():
        if f >= min_freq:
            vocab[w] = len(vocab)
    return vocab, {i: w for w, i in vocab.items()}, counter

vocab, idx_to_word, _ = build_vocab(tokens, min_freq=3)
vocab_size = len(vocab)
print("Vocabulary size:", vocab_size)

def tokens_to_ids(tokens, vocab):
    return [vocab.get(t, vocab["<UNK>"]) for t in tokens]

token_ids = tokens_to_ids(tokens, vocab)

def split_data(token_ids, train_ratio=0.8, val_ratio=0.1):
    n = len(token_ids)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return token_ids[:train_end], token_ids[train_end:val_end], token_ids[val_end:]

train_ids, val_ids, test_ids = split_data(token_ids)
print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

class LanguageModelDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.token_ids = token_ids
        self.seq_len = seq_len
    def __len__(self):
        return len(self.token_ids) - self.seq_len
    def __getitem__(self, idx):
        return (torch.tensor(self.token_ids[idx:idx+self.seq_len], dtype=torch.long),
                torch.tensor(self.token_ids[idx+self.seq_len], dtype=torch.long))

SEQ_LEN = 15
BATCH_SIZE = 64
train_loader = DataLoader(LanguageModelDataset(train_ids, SEQ_LEN), BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(LanguageModelDataset(val_ids, SEQ_LEN), BATCH_SIZE, shuffle=False, drop_last=False)
test_loader = DataLoader(LanguageModelDataset(test_ids, SEQ_LEN), BATCH_SIZE, shuffle=False, drop_last=False)
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

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

model = LSTMLanguageModel(vocab_size, 128, 256, num_layers=2, dropout=0.3).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total = 0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        total += loss.item()
    return total / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    total = 0
    with torch.no_grad():
        for bx, by in loader:
            total += criterion(model(bx.to(device)), by.to(device)).item()
    return total / len(loader)

EPOCHS = 10
print("\nTraining...")
best_val_loss = float('inf')
best_model = None
patience_counter = 0
early_stop_patience = 2
for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss = evaluate(model, val_loader, criterion)
    scheduler.step(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
    print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    if patience_counter >= early_stop_patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(best_model)
test_loss = evaluate(model, test_loader, criterion)
print(f"\nFinal Test Loss: {test_loss:.4f}")
print(f"Final Test Perplexity: {math.exp(test_loss):.2f}")
