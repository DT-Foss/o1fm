from __future__ import annotations

import json

from o1fm import cli


def test_demo_dispatch_keeps_teacher_only_arguments_out(
    monkeypatch,
    capsys,
) -> None:
    captured = {}

    def fake_demo(**kwargs):
        captured.update(kwargs)
        return {"schema": "demo"}

    monkeypatch.setattr(cli, "run_demo", fake_demo)
    assert cli.main(["demo", "--epochs", "3", "--seed", "4"]) == 0
    assert captured == {
        "seed": 4,
        "epochs": 3,
        "state_dir": None,
        "enable_conformal": True,
    }
    assert json.loads(capsys.readouterr().out) == {"schema": "demo"}


def test_teacher_demo_dispatches_local_and_policy_controls(
    monkeypatch,
    capsys,
) -> None:
    captured = {}

    def fake_teacher(**kwargs):
        captured.update(kwargs)
        return {"schema": "teacher"}

    monkeypatch.setattr(cli, "run_open_teacher_demo", fake_teacher)
    assert (
        cli.main(
            [
                "teacher-demo",
                "--teacher",
                "toto2",
                "--local-checkpoint",
                "local/4m",
                "--quality-tolerance",
                "0.004",
                "--coverage-tolerance",
                "0.02",
                "--atlas-dir",
                "/tmp/o1fm-atlas",
            ]
        )
        == 0
    )
    assert captured["teacher_name"] == "toto2"
    assert captured["local_checkpoint"] == "local/4m"
    assert captured["enable_local_teacher"] is True
    assert captured["quality_tolerance"] == 0.004
    assert captured["coverage_tolerance"] == 0.02
    assert captured["atlas_dir"] == "/tmp/o1fm-atlas"
    assert json.loads(capsys.readouterr().out) == {"schema": "teacher"}
