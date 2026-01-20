import argparse
from pathlib import Path

from genomeltm.agents.executor import Executor
from genomeltm.agents.planner import Planner
from genomeltm.eval.abstention import main as abstention_main
from genomeltm.utils.config import load_yaml


def main():
    p = argparse.ArgumentParser(prog="genomeltm", description="GenomeLTM-MoE research CLI (scaffold)")
    sub = p.add_subparsers(dest="cmd")

    dry = sub.add_parser("dry-run", help="Run a small dry pipeline on synthetic inputs")
    dry.add_argument("--out", type=str, default="runs/dry", help="Output directory")

    abst = sub.add_parser("eval-abstention", help="Compute risk/coverage for abstention heads")
    abst.add_argument("--config", type=str, default="configs/abstention_eval.yaml")
    abst.add_argument("--input", type=str, default=None)
    abst.add_argument("--output", type=str, default=None)

    agentic = sub.add_parser("agentic-run", help="Run planner/executor orchestration")
    agentic.add_argument("--config", type=str, default="configs/agentic.yaml")
    agentic.add_argument("--goal", type=str, default="default")

    args = p.parse_args()
    if args.cmd == "dry-run":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "README.txt").write_text("Dry-run placeholder. Wire real pipeline here.\n")
        print(f"Wrote: {out/'README.txt'}")
    elif args.cmd == "eval-abstention":
        abstention_args = ["--config", args.config]
        if args.input:
            abstention_args.extend(["--input", args.input])
        if args.output:
            abstention_args.extend(["--output", args.output])
        abstention_main(abstention_args)
    elif args.cmd == "agentic-run":
        cfg = load_yaml(args.config)
        planner = Planner(cfg)
        executor = Executor(cfg)
        plan = planner.plan(args.goal)
        records = executor.execute(plan)
        for record in records:
            print(f"{record.task}: {record.status} {record.artifacts}")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
