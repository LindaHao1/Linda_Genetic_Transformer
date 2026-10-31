from argparse import Namespace
from pathlib import Path

import yaml

from virtuoso_optimization_GA.transformer_synthesis import build_job, validate


def test_build_job_injects_targets_and_seed(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({
        "mode": "oct_transformer",
        "project": {"parent_output_dir": str(tmp_path)},
        "geometry_range": {
            "pri_inner_r": {"min": 10, "max": 100},
            "pri_width": {"min": 1, "max": 20},
            "sec_inner_r": {"min": 10, "max": 100},
            "sec_width": {"min": 1, "max": 20},
        },
    }))
    args = Namespace(
        base_config=base, lp=1e-9, ls=2e-9, k=0.6, qp=15.0, qs=16.0,
        pri_inner_r=50.0, pri_width=8.0, sec_inner_r=40.0, sec_width=7.0,
        population=10, generations=8, workers=2, run_name="unit_test",
    )
    validate(args)
    cfg = build_job(args)
    assert cfg["target"] == {"Lp": 1e-9, "Ls": 2e-9, "k": 0.6, "Qp": 15.0, "Qs": 16.0}
    assert cfg["optimizer"]["seed_design"]["pri_inner_r"] == 50.0
    assert cfg["optimizer"]["n_parallel"] == 2
    assert cfg["outputs"]["run_name"] == "unit_test"
