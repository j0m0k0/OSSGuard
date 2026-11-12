"""Reproduction entry points for heuristic and fixed-pool experiments."""

from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from .evaluation import evaluate_pair
from .learning import train_fixed_pool, train_specialists
from .model import GameConfig
from .policies import heuristic_maintainers, known_contributors


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_heuristic_baseline(output: Path, episodes: int = 300, seed: int = 0,
                           config: GameConfig | None = None) -> list[dict]:
    cfg = config or GameConfig()
    # The random contributor is included in the full heuristic matrix.
    from .policies import HeuristicPolicy
    contributors = [HeuristicPolicy("Random", "contributor"), *known_contributors(),
                    HeuristicPolicy("Greedy", "contributor")]
    maintainers = heuristic_maintainers()
    rows = []
    for i, contributor in enumerate(contributors):
        for j, maintainer in enumerate(maintainers):
            rc, rm, metrics = evaluate_pair(cfg, contributor, maintainer, episodes,
                                             seed + i * 1009 + j * 9176)
            rows.append({"contributor": contributor.name, "maintainer": maintainer.name,
                         "contributor_return_mean": rc, "maintainer_return_mean": rm,
                         **metrics})
    _write_csv(output / "heuristic_payoffs.csv", rows)
    return rows


def run_fixed_pool(output: Path, method: str = "mss-unif", epochs: int = 100,
                   steps_per_epoch: int = 5_000, evaluation_episodes: int = 300,
                   seed: int = 0, temperature: float = 0.25,
                   config: GameConfig | None = None) -> list[dict]:
    cfg = config or GameConfig()
    output.mkdir(parents=True, exist_ok=True)
    contributors = known_contributors()
    rows = []
    if method == "specialists":
        policies = train_specialists(cfg, steps_per_epoch * epochs, seed)
        for idx, (contributor, policy) in enumerate(zip(contributors, policies)):
            rc, rm, metrics = evaluate_pair(cfg, contributor, policy,
                                             evaluation_episodes, seed + 90_000 + idx)
            rows.append({"method": method, "contributor": contributor.name,
                         "maintainer_return_mean": rm, "contributor_return_mean": rc,
                         **metrics})
            import torch
            torch.save({"state_dict": policy.network.state_dict(), "role": policy.role,
                        "config": cfg.to_dict(), "seed": seed + idx},
                       output / f"specialist_{contributor.name.lower()}.pt")
        _write_csv(output / "specialists_known_pool.csv", rows)
        return rows

    policy, history = train_fixed_pool(method, epochs, steps_per_epoch, seed, cfg,
                                       temperature, evaluation_episodes)
    for idx, contributor in enumerate(contributors):
        rc, rm, metrics = evaluate_pair(cfg, contributor, policy,
                                         evaluation_episodes, seed + 90_000 + idx)
        rows.append({"method": method, "contributor": contributor.name,
                     "contributor_return_mean": rc, "maintainer_return_mean": rm,
                     **metrics})
    _write_csv(output / f"{method}_known_pool.csv", rows)
    _write_csv(output / f"{method}_training.csv", history)
    unseen_rows = []
    for idx, contributor in enumerate(known_contributors(unseen=True)):
        rc, rm, metrics = evaluate_pair(cfg, contributor, policy, evaluation_episodes,
                                         seed + 190_000 + idx)
        unseen_rows.append({"method": method, "contributor": contributor.name,
                            "contributor_return_mean": rc, "maintainer_return_mean": rm,
                            **metrics})
    _write_csv(output / f"{method}_unseen_transfer.csv", unseen_rows)

    if method == "mss-unif":
        signal_rows = []
        for idx in range(6):
            p_clean, p_exploit = 0.5 - idx * 0.1, 0.5 + idx * 0.1
            signal_cfg = GameConfig(**{**cfg.to_dict(), "signal_clean": p_clean,
                                       "signal_exploit": p_exploit})
            for j, contributor in enumerate(contributors):
                rc, rm, metrics = evaluate_pair(signal_cfg, contributor, policy,
                                                 evaluation_episodes,
                                                 seed + 290_000 + idx * 13 + j)
                signal_rows.append({"signal_clean": p_clean, "signal_exploit": p_exploit,
                                    "contributor": contributor.name,
                                    "contributor_return_mean": rc,
                                    "maintainer_return_mean": rm, **metrics})
        _write_csv(output / "mss-unif_signal_sweep.csv", signal_rows)
    import torch
    torch.save({"state_dict": policy.network.state_dict(), "role": policy.role,
                "config": cfg.to_dict(), "seed": seed}, output / f"{method}_maintainer.pt")
    return rows


def summarize_rows(rows: list[dict]) -> dict[str, float]:
    values = np.asarray([r["maintainer_return_mean"] for r in rows], dtype=float)
    return {"mean_maintainer_return": float(values.mean()),
            "sd_across_contributors": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
