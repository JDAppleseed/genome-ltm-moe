import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from genomeltm.models.dna_mlm import DNAMaskedLM, DNA_VOCAB
from genomeltm.data.genome_sequence_dataset import GenomeSequenceWindowDataset

def collate(batch):
    ids = torch.stack([b.ids for b in batch], dim=0)
    masked_ids = torch.stack([b.masked_ids for b in batch], dim=0)
    # create labels: -100 where not masked
    labels = ids.clone()
    not_masked = masked_ids != DNA_VOCAB["[MASK]"]
    labels[not_masked] = -100
    return masked_ids, labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    # Synthetic placeholder sequences for scaffolding.
    # Replace with FASTA/contigs indexing in real use.
    sequences = [
        "ACGT" * 2000,
        "AAAACCCCGGGGTTTT" * 500,
        "ACGTTGCATGTCAGTCA" * 500,
    ]

    ds = GenomeSequenceWindowDataset(sequences, window_len=2048, mask_prob=0.15, seed=0)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=0, collate_fn=collate)

    model = DNAMaskedLM(d_model=512, n_layers=8, n_heads=8, d_ff=2048).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    it = iter(dl)
    for step in range(1, args.steps + 1):
        try:
            masked_ids, labels = next(it)
        except StopIteration:
            it = iter(dl)
            masked_ids, labels = next(it)

        masked_ids = masked_ids.to(args.device)
        labels = labels.to(args.device)

        logits = model(masked_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"step={step} loss={loss.item():.4f}")

    print("Done (scaffold). Save checkpoints in a real run.")

if __name__ == "__main__":
    main()
