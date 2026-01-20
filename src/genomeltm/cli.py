import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser(prog="genomeltm", description="GenomeLTM-MoE research CLI (scaffold)")
    sub = p.add_subparsers(dest="cmd")

    dry = sub.add_parser("dry-run", help="Run a small dry pipeline on synthetic inputs")
    dry.add_argument("--out", type=str, default="runs/dry", help="Output directory")

    args = p.parse_args()
    if args.cmd == "dry-run":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "README.txt").write_text("Dry-run placeholder. Wire real pipeline here.\n")
        print(f"Wrote: {out/'README.txt'}")
    else:
        p.print_help()

if __name__ == "__main__":
    main()
