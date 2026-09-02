"""O1FM command-line entry point."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .demo import run_demo
from .open_teacher_demo import run_open_teacher_demo


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
    demo.add_argument(
        "--no-conformal",
        action="store_true",
        help="disable the directed O1 conformal field",
    )
    teacher = subcommands.add_parser(
        "teacher-demo",
        help="run a real open-weight teacher through O1FM",
    )
    teacher.add_argument("--teacher", choices=("toto2", "chronos2"), default="toto2")
    teacher.add_argument("--checkpoint")
    teacher.add_argument("--local-checkpoint")
    teacher.add_argument("--no-local", action="store_true")
    teacher.add_argument("--device", default="cpu")
    teacher.add_argument("--seed", type=int, default=23)
    teacher.add_argument("--quality-tolerance", type=float, default=0.005)
    teacher.add_argument("--coverage-tolerance", type=float, default=0.05)
    teacher.add_argument("--state-dir")
    teacher.add_argument("--output", help="optional JSON result path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        result = run_demo(
            seed=args.seed,
            epochs=args.epochs,
            state_dir=args.state_dir,
            enable_conformal=not args.no_conformal,
        )
    elif args.command == "teacher-demo":
        result = run_open_teacher_demo(
            teacher_name=args.teacher,
            checkpoint=args.checkpoint,
            local_checkpoint=args.local_checkpoint,
            enable_local_teacher=not args.no_local,
            device=args.device,
            seed=args.seed,
            quality_tolerance=args.quality_tolerance,
            coverage_tolerance=args.coverage_tolerance,
            state_dir=args.state_dir,
        )
    else:  # pragma: no cover - argparse owns commands.
        raise AssertionError("unsupported command")
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
