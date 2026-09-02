from __future__ import annotations

from pathlib import Path

import pytest

from o1fm.cartography_demo import run_cartography_demo
from o1fm.cli import build_parser
from o1fm.intervention_compiler_demo import _FAMILIES, run_intervention_compiler_demo
from o1fm.synthetic import generate_synthetic_dataset


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


def test_compiler_cli_and_bounds_fail_before_model_loading(tmp_path: Path) -> None:
    arguments = build_parser().parse_args(
        ("compiler-demo", "--atlas-dir", "atlas.causal")
    )
    assert arguments.training_windows == 8
    assert arguments.evaluation_windows == 4
    with pytest.raises(ValueError, match="at least four train"):
        run_intervention_compiler_demo(
            atlas_dir=tmp_path / "unused.causal",
            training_windows=3,
        )


def test_compiler_families_build_exact_future_plans() -> None:
    inputs = generate_synthetic_dataset().online[0].inputs
    for family in _FAMILIES:
        plan = family.plan(inputs, 0.25)
        assert plan.edited_covariates == (family.covariate,)
