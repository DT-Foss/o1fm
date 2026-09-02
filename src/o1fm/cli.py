"""O1FM command-line entry point."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .demo import run_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="o1fm")
    subcommands = parser.add_subparsers(dest="command", required=True)
    demo = subcommands.add_parser(
        "demo", help="run the trained multivariate O1/Crystal proof"
    )
    demo.add_argument("--epochs", type=int, default=20)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--state-dir")
    demo.add_argument("--output", help="optional JSON result path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "demo":  # pragma: no cover - argparse owns commands.
        raise AssertionError("unsupported command")
    result = run_demo(
        seed=args.seed,
        epochs=args.epochs,
        state_dir=args.state_dir,
    )
    encoded = json.dumps(
        result,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    if args.output is not None:
        destination = Path(args.output).expanduser().absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


__all__ = ["build_parser", "main"]
