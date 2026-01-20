# MacBook-Oriented Orchestration Workflow

## Control Plane Principles
- The laptop is a control plane only (no training data routed through it).
- Large datasets must be staged near compute (cluster scratch or object storage).
- Use SSH + rsync/git for code, and separate data staging tools for datasets.

## Recommended Workflow
1. **Develop locally** and commit changes.
2. **Sync code** to the cluster:
   - `scripts/remote/sync_repo.sh <ssh-host> <remote-path>`
3. **Stage data near compute**:
   - `scripts/data/stage_to_scratch.sh <source> <dest>`
4. **Submit job**:
   - `scripts/remote/submit_slurm.sh <ssh-host> <remote-path> <sbatch-args>`
5. **Monitor logs**:
   - `scripts/remote/tail_logs.sh <ssh-host> <log-path>`
6. **Fetch artifacts**:
   - `scripts/remote/fetch_artifacts.sh <ssh-host> <remote-artifacts> <local-dest>`

## Constraints
- No wet-lab automation or genome synthesis/editing.
- Data access must be compliant with IRB/DUA.
- Large data must remain near compute; no laptop staging for real runs.
