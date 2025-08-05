"""
Vanilla Transformer baseline for IMDb 4K‑token sentiment classification.
====================================================
* Purpose: Provide a speed/accuracy comparison target for Epsilon HQSA.
* Key differences from Epsilon:
  - Standard multi‑head softmax self‑attention (no binning, no halting).
  - No Jacobian diagnostics or KL routing loss.
  - Simpler training loop (no scheduler warnings, optional cosine LR).
* Usage (example):
    python vanilla_baseline.py --max_seq_length 4096 --batch_size 2 --num_epochs 3
"""

import argparse, time, os
from datasets import load_dataset
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import math

# ---------------------------------------------------------------------
# Data utilities (re‑uses tokenizer + dataset logic from data_utils.py)
# ---------------------------------------------------------------------

def get_dataloaders(batch_size: int, max_length: int, model_name: str):
    ds = load_dataset("imdb")
    train_val = ds["train"].train_test_split(test_size=0.1, shuffle=True, seed=42, stratify_by_column="label")
    train_raw, val_raw = train_val["train"], train_val["test"]

    tok = AutoTokenizer.from_pretrained(model_name)

    def tok_fn(ex):
        return tok(ex["text"], truncation=True, padding="max_length", max_length=max_length)

    train_ds = train_raw.map(tok_fn, batched=True).remove_columns(["text"])
    val_ds   = val_raw.map(tok_fn, batched=True).remove_columns(["text"])
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
            tok.vocab_size, tok.pad_token_id)

# ---------------------------------------------------------------------
# Vanilla Transformer Encoder (Post‑LN)
# ---------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a tensor for positional encodings
        pe = torch.zeros(max_len, d_model)
        
        # Create a tensor for positions (0, 1, ..., max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # Shape: [max_len, 1]
        
        # Calculate the div_term.
        # Each element in div_term corresponds to a pair of sin/cos dimensions.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # Shape: [ceil(d_model/2)]
        
        # Apply sin to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices (1, 3, 5, ...)
        # For `cos`, we only need `d_model // 2` terms from `div_term`.
        if d_model > 1: # There are odd indices only if d_model > 1
            num_cos_terms = d_model // 2
            if num_cos_terms > 0: # Check if there are any cosine terms to compute
                pe[:, 1::2] = torch.cos(position * div_term[:num_cos_terms])
        
        pe = pe.unsqueeze(1) # This makes it (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)].squeeze(1)
        return self.dropout(x)

class VanillaBlock(nn.Module):
    def __init__(self, d_model:int, n_heads:int, ffn_dim:int, dropout:float=0.1):
        super().__init__()
        self.ln1   = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2   = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        y,_ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), key_padding_mask=attn_mask)
        x = x + self.drop1(y)
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x

class VanillaTransformer(nn.Module):
    def __init__(self, vocab:int, d_model:int, n_heads:int, num_layers:int, ffn_dim:int, num_classes:int, pad_token:int, max_seq_length_for_pe: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=pad_token)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.0, max_len=max_seq_length_for_pe)
        self.blocks= nn.ModuleList([VanillaBlock(d_model,n_heads,ffn_dim) for _ in range(num_layers)])
        self.ln_out= nn.LayerNorm(d_model)
        self.head  = nn.Linear(d_model, num_classes)

    def forward(self, ids, mask):
        x = self.embed(ids)
        x = self.pos_encoder(x)
        for blk in self.blocks:
            x = blk(x, attn_mask=~mask.bool()) 
        x = self.ln_out(x)
        mask_expanded = mask.unsqueeze(-1).float()
        summed = (x * mask_expanded).sum(1)
        count = mask_expanded.sum(1).clamp(min=1e-9)
        pooled = summed / count
        return self.head(pooled)

# ---------------------------------------------------------------------
# Train / Eval loop ----------------------------------------------------
# ---------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading data...")
    train_dl, val_dl, vocab, pad = get_dataloaders(args.batch_size, args.max_seq_length, args.tokenizer_name)
    print(f"Vocab size: {vocab}, Pad token ID: {pad}")
    print(f"Train batches: {len(train_dl)}, Val batches: {len(val_dl)}")

    print("Initializing model...")
    model = VanillaTransformer(vocab, args.d_model, args.num_heads, args.num_layers, args.ffn_dim, 2, pad, args.max_seq_length).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    opt   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(train_dl)*args.num_epochs) if args.cosine else None
    crit  = nn.CrossEntropyLoss()

    print("Starting training...")
    # Create output directory for checkpoints
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    best_val_acc = 0.0  # Track best validation accuracy to decide when to save
    for epoch in range(args.num_epochs):
        model.train(); t0=time.time(); loss_acc=0; n=0
        train_iterator = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        for batch in train_iterator:
            ids, am, lbl = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)
            
            opt.zero_grad()
            logits = model(ids, am)
            loss = crit(logits, lbl)
            loss.backward()
            opt.step()
            
            if sched: 
                sched.step()
            
            loss_acc += loss.item()
            n += 1
            train_iterator.set_postfix({'train_loss': f"{loss.item():.4f}"})
            
        avg_train_loss = loss_acc / n if n > 0 else 0
        print(f"Epoch {epoch+1} Train loss: {avg_train_loss:.4f}  Time: {time.time()-t0:.1f}s")

        # Eval
        model.eval(); val_loss=0; correct=0; total=0
        val_iterator = tqdm(val_dl, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]")
        with torch.no_grad():
            for batch in val_iterator:
                ids, am, lbl = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)
                logits = model(ids, am)
                loss = crit(logits, lbl)
                val_loss += loss.item()
                preds = logits.argmax(1)
                correct += (preds==lbl).sum().item()
                total += lbl.size(0)
                val_iterator.set_postfix({'val_loss': f"{loss.item():.4f}", 'val_acc': f"{(preds==lbl).float().mean():.4f}"})

        avg_val_loss = val_loss / len(val_dl) if len(val_dl) > 0 else 0
        avg_val_acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1} Val loss: {avg_val_loss:.4f}  Val acc: {avg_val_acc:.4f}")

        # --- Checkpoint: Save best model so far ---
        if avg_val_acc > best_val_acc and args.output_dir:
            best_val_acc = avg_val_acc
            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  New best model saved to {ckpt_path}")

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--tokenizer_name", default="bert-base-uncased")
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=12)
    p.add_argument("--ffn_dim", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--cosine", action="store_true", help="Use cosine LR scheduler")
    p.add_argument("--output_dir", type=str, default="./vanilla_model_output", help="Directory to save best model checkpoints")
    args=p.parse_args(); train(args)