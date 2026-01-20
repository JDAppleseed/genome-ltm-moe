# Agentic Workflows Inspiration (CRISPR-GPT → GenomAIc)

We adopt the multi-agent orchestration pattern:
- Planner: decomposes user goals into tasks (data ingest, alignment, eval, reporting)
- Executor: runs deterministic state machines for each task
- Tool Providers: wraps external tools (alignment, QC, benchmark runners, literature search)
- User Proxy: captures user constraints + approvals (IRB/DUA where needed)

We explicitly exclude wet-lab automation.
