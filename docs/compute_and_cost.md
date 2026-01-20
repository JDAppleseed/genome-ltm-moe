# Compute and Cost (Scaling Stack)

## Logged Throughput Metrics
The distributed training loop logs the following per-interval metrics:
- steps/sec
- tokens/sec (or bp/sec if you map tokens to bases)
- data wait time (seconds spent waiting on input)

Source of truth: `src/genomaic/train/profiling.py`.

## Estimating Cost per Genome / Epoch
1. Measure tokens/sec on your target cluster (use a representative shard size).
2. Multiply tokens/sec by total wall time to estimate tokens per epoch.
3. Convert tokens → bases (bp) using your tokenizer ratio (e.g., 1 token = 1 bp).
4. Cost per epoch ≈ (wall time hours) × (hourly GPU price).
5. Cost per genome ≈ cost per epoch / genomes per epoch.

## DeepSpeed + ZeRO Defaults
The default MoE configuration uses ZeRO-2 (`configs/deepspeed/zero2_moe.json`).
Dense baselines use ZeRO-3 (`configs/deepspeed/zero3_dense.json`).

## Practical Tips
- Stage large datasets near compute (parallel FS or object store).
- Use `checkpoint_freq` and `eval_freq` to balance cost vs. observability.
