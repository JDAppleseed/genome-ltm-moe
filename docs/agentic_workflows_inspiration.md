# Agentic Workflows Inspiration (CRISPR-GPT → GenomAIc)

We adopt the multi-agent orchestration pattern:
- Planner: decomposes goals into deterministic tasks (ingest → analyze → benchmark → report)
- Executor: runs sequential, reproducible tasks with provenance tracking
- Tool Providers: dry-lab utilities (ingest, benchmark, report)
- Data-access policy: enforces path-based constraints from YAML configuration

We explicitly exclude wet-lab automation.
