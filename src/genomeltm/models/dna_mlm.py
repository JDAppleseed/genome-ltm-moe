import torch
import torch.nn as nn

DNA_VOCAB = {"A":0, "C":1, "G":2, "T":3, "N":4, "[MASK]":5}
ID2TOK = {v:k for k,v in DNA_VOCAB.items()}

class DNATokenEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int = 6):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.tok(ids)

class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        h = self.ln2(x)
        x = x + self.ff(h)
        return x

class DNAMaskedLM(nn.Module):
    """
    Safe foundation component:
    - Learns representations of DNA sequence via masked-token prediction.
    - NOT a trait->genome generator.
    - Use for embeddings, variant-effect scoring (via delta-likelihood), region annotation.
    """
    def __init__(self, d_model: int = 512, n_layers: int = 8, n_heads: int = 8, d_ff: int = 2048, vocab_size: int = 6):
        super().__init__()
        self.emb = DNATokenEmbedding(d_model=d_model, vocab_size=vocab_size)
        self.pos = nn.Embedding(4096, d_model)  # for windowed training; extend as needed
        self.blocks = nn.ModuleList([SimpleTransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        ids: (B, L) token ids containing [MASK] tokens
        returns logits: (B, L, vocab)
        """
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L).clamp(max=self.pos.num_embeddings-1)
        x = self.emb(ids) + self.pos(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        logits = self.head(x)
        return logits
