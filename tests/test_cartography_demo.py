from __future__ import annotations

from pathlib import Path

import pytest

from o1fm.cartography_demo import run_cartography_demo
from o1fm.cli import build_parser


def test_cartography_cli_defaults_to_covariate_capable_teacher() -> None:
    arguments = build_parser().parse_args(
        ("cartography-demo", "--atlas-dir", "atlas.causal")
    )
    assert arguments.teacher == "chronos2"


def test_toto_intervention_demo_fails_before_model_loading(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not support known-future"):
        run_cartography_demo(
            teacher_name="toto2",
            atlas_dir=tmp_path / "unused.causal",
        )
