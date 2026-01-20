# Compute and scaling: B200 now, GB300 later

## Key constraints
- Raw reads massively increase token volume vs VCF-only modeling.
- Ultra-long context requires memory-heavy state or retrieval systems.
- MoE introduces communication overhead (routing, expert parallelism).

## Hardware notes
### B200 (cloud accessible)
- B200 SXM6 180GB appears in Lambda pricing/instances pages.
- Price reference: $4.99 per GPU-hour (plus tax); instance includes large CPU/RAM.
- This is suitable for early frontier prototyping and scaled expert training.

### GB300-class (future)
- NVIDIA positions Blackwell Ultra/GB300 for larger context lengths and higher HBM per GPU.
- Higher HBM is directly useful for ultra-long-context inference/training.

## Parallelism plan
- Data parallel (across samples / genome tiles)
- Expert parallel (MoE experts sharded)
- Sequence/tile parallel (split long sequences across GPUs)
- Pipeline parallel (stack partitions)
- Activation checkpointing to fit long contexts

## Target cluster sizes (planning)
Prototype:
- 32–128 GPUs (B200)
Frontier run:
- 256–2048 GPUs (B200/GB300-class), depending on context, expert count, and dataset scale
