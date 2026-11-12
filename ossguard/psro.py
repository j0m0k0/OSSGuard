"""Empirical-game OSSGuard-PSRO for two-player general-sum policies."""

from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import nashpy as nash

from .evaluation import evaluate_pair
from .learning import DQNPolicy, train_response_oracle
from .model import GameConfig
from .policies import HeuristicPolicy


def solve_meta_game(uc: np.ndarray, um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve small games by support enumeration and large games by regret matching."""
    game = nash.Game(uc, um)
    if max(uc.shape) <= 12:
        equilibria = game.support_enumeration()
        try:
            x, y = next(equilibria)
        except (StopIteration, ValueError, ZeroDivisionError):
            x, y = _regret_matching(uc, um)
    else:
        # Support enumeration is exponential in support size. Regret matching
        # yields average mixed strategies for the growing empirical game.
        x, y = _regret_matching(uc, um)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x = np.maximum(x, 0); y = np.maximum(y, 0)
    return x / x.sum(), y / y.sum()


def _regret_matching(uc: np.ndarray, um: np.ndarray,
                     iterations: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Compute average strategies with simultaneous external-regret matching."""
    rows, cols = uc.shape
    reg_c = np.zeros(rows); reg_m = np.zeros(cols)
    x = np.full(rows, 1.0 / rows); y = np.full(cols, 1.0 / cols)
    sum_x = np.zeros(rows); sum_y = np.zeros(cols)
    for _ in range(iterations):
        payoff_c = uc @ y
        payoff_m = x @ um
        reg_c += payoff_c - float(x @ payoff_c)
        reg_m += payoff_m - float(payoff_m @ y)
        pos_c, pos_m = np.maximum(reg_c, 0), np.maximum(reg_m, 0)
        x = pos_c / pos_c.sum() if pos_c.sum() else np.full(rows, 1.0 / rows)
        y = pos_m / pos_m.sum() if pos_m.sum() else np.full(cols, 1.0 / cols)
        sum_x += x; sum_y += y
    return sum_x / iterations, sum_y / iterations


def restricted_metrics(uc: np.ndarray, um: np.ndarray, x: np.ndarray,
                       y: np.ndarray) -> dict[str, float]:
    vc, vm = float(x @ uc @ y), float(x @ um @ y)
    contributor_gain = float(np.max(uc @ y) - vc)
    maintainer_gain = float(np.max(x @ um) - vm)
    return {"contributor_payoff": vc, "maintainer_payoff": vm,
            "contributor_deviation_gain": max(0.0, contributor_gain),
            "maintainer_deviation_gain": max(0.0, maintainer_gain),
            "nashconv": max(0.0, contributor_gain) + max(0.0, maintainer_gain)}


def run_psro(output: Path, iterations: int = 200, steps_per_oracle: int = 5_000,
             evaluation_episodes: int = 300, seed: int = 0,
             config: GameConfig | None = None) -> list[dict[str, float]]:
    """Run expanding-population PSRO from the paper's seven heuristic families."""
    cfg = config or GameConfig()
    output.mkdir(parents=True, exist_ok=True)
    policy_dir = output / "policies"
    policy_dir.mkdir(exist_ok=True)
    names = ["Random", "Periodic", "Gate", "Ramp", "ToughStart", "ToughEnd", "Greedy"]
    contributors = [HeuristicPolicy(name, "contributor") for name in names]
    maintainers = [HeuristicPolicy(name, "maintainer") for name in names]
    for p in contributors + maintainers:
        if p.name == "Periodic": p.period = 5
        elif p.name == "Gate": p.threshold, p.prob = 25, 0.5
        elif p.name == "Ramp": p.rate = 0.5
        elif p.name == "ToughStart": p.tough_start = 15
        elif p.name == "ToughEnd": p.tough_end = 75

    uc = np.zeros((len(contributors), len(maintainers)), dtype=float)
    um = np.zeros_like(uc)
    for i, c in enumerate(contributors):
        for j, m in enumerate(maintainers):
            uc[i, j], um[i, j], _ = evaluate_pair(cfg, c, m, evaluation_episodes,
                                                   seed + i * 1009 + j * 9176)

    history: list[dict[str, float]] = []
    meta_history: list[dict[str, float | str]] = []
    for iteration in range(1, iterations + 1):
        x, y = solve_meta_game(uc, um)
        new_m = train_response_oracle("maintainer", contributors, cfg,
                                      steps=steps_per_oracle, seed=seed + 2 * iteration,
                                      opponent_weights=x)
        new_c = train_response_oracle("contributor", maintainers, cfg,
                                      steps=steps_per_oracle, seed=seed + 2 * iteration + 1,
                                      opponent_weights=y)
        import torch
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in new_m.network.state_dict().items()},
                    "role": new_m.role, "seed": seed + 2 * iteration,
                    "game_config": cfg.to_dict()}, policy_dir / f"maintainer_{iteration:03d}.pt")
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in new_c.network.state_dict().items()},
                    "role": new_c.role, "seed": seed + 2 * iteration + 1,
                    "game_config": cfg.to_dict()}, policy_dir / f"contributor_{iteration:03d}.pt")

        # Evaluate each response oracle against every policy in the opposing population.
        contributor_column = []
        maintainer_column = []
        for i, c in enumerate(contributors):
            rc, rm, _ = evaluate_pair(cfg, c, new_m, evaluation_episodes,
                                       seed + iteration * 100_003 + i)
            contributor_column.append(rc)
            maintainer_column.append(rm)
        new_row_c, new_row_m = [], []
        for j, m in enumerate(maintainers):
            rc, rm, _ = evaluate_pair(cfg, new_c, m, evaluation_episodes,
                                       seed + iteration * 100_019 + j)
            new_row_c.append(rc)
            new_row_m.append(rm)
        rc, rm, _ = evaluate_pair(cfg, new_c, new_m, evaluation_episodes,
                                   seed + iteration * 100_043)
        uc = np.pad(uc, ((0, 1), (0, 1)))
        um = np.pad(um, ((0, 1), (0, 1)))
        uc[:-1, -1] = contributor_column
        um[:-1, -1] = maintainer_column
        uc[-1, :-1] = new_row_c
        um[-1, :-1] = new_row_m
        uc[-1, -1], um[-1, -1] = rc, rm
        contributors.append(new_c)
        maintainers.append(new_m)

        x, y = solve_meta_game(uc, um)
        row = {"iteration": float(iteration), "contributors": float(len(contributors)),
               "maintainers": float(len(maintainers)),
               **restricted_metrics(uc, um, x, y)}
        history.append(row)
        meta_history.extend({"iteration": iteration, "player": "contributor",
                             "policy_index": i, "probability": float(prob)}
                            for i, prob in enumerate(x))
        meta_history.extend({"iteration": iteration, "player": "maintainer",
                             "policy_index": j, "probability": float(prob)}
                            for j, prob in enumerate(y))
        if iteration % 10 == 0 or iteration == iterations:
            with (output / "psro_progress.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(history[0]))
                writer.writeheader(); writer.writerows(history)
            with (output / "psro_meta_strategies.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(meta_history[0]))
                writer.writeheader(); writer.writerows(meta_history)
    if history:
        np.savez_compressed(output / "psro_payoffs.npz", contributor_payoffs=uc,
                            maintainer_payoffs=um)
        np.savez_compressed(output / "psro_final_meta_strategies.npz", contributor=x,
                            maintainer=y)
    return history
