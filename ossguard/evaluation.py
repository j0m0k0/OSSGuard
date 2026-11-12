"""Seeded policy evaluation and payoff matrices."""

from __future__ import annotations

from typing import Sequence
import numpy as np

from .env import OSSGuardEnv
from .model import GameConfig


def evaluate_pair(cfg: GameConfig | None, contributor, maintainer, episodes: int = 300,
                  seed: int = 0) -> tuple[float, float, dict[str, float]]:
    """Estimate mean returns and episode-level behavior for one policy pair."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    env = OSSGuardEnv(cfg or GameConfig(), seed=seed)
    rng = np.random.default_rng(seed + 1)
    returns_c: list[float] = []
    returns_m: list[float] = []
    counts = {"contribute": 0, "exploit": 0, "accept": 0, "inspect": 0,
              "episodes": 0, "detected_episodes": 0}
    for episode in range(episodes):
        obs_c, _ = env.reset(seed + episode)
        if hasattr(contributor, "reset"):
            contributor.reset()
        if hasattr(maintainer, "reset"):
            maintainer.reset()
        total_c = total_m = 0.0
        while not env.done:
            action_c = contributor.act(obs_c, rng)
            obs_m = env.begin_turn(action_c)
            action_m = maintainer.act(obs_m, rng)
            result = env.end_turn(action_m)
            counts["contribute" if action_c == 0 else "exploit"] += 1
            counts["accept" if action_m == 0 else "inspect"] += 1
            total_c += result.contributor_reward
            total_m += result.maintainer_reward
            obs_c = result.contributor_observation
            if result.terminated:
                counts["detected_episodes"] += 1
        returns_c.append(total_c)
        returns_m.append(total_m)
    counts["episodes"] = episodes
    denom = max(episodes, 1)
    mean_c, mean_m = float(np.mean(returns_c)), float(np.mean(returns_m))
    counts = {key: float(value) / denom if key not in ("episodes", "detected_episodes")
              else float(value) for key, value in counts.items()}
    counts["contributor_reward_sd"] = float(np.std(returns_c, ddof=1)) if episodes > 1 else 0.0
    counts["maintainer_reward_sd"] = float(np.std(returns_m, ddof=1)) if episodes > 1 else 0.0
    return mean_c, mean_m, counts


def payoff_matrix(contributors: Sequence, maintainers: Sequence, cfg: GameConfig,
                  episodes: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    uc = np.zeros((len(contributors), len(maintainers)))
    um = np.zeros_like(uc)
    for i, contributor in enumerate(contributors):
        for j, maintainer in enumerate(maintainers):
            uc[i, j], um[i, j], _ = evaluate_pair(cfg, contributor, maintainer,
                                                   episodes=episodes,
                                                   seed=seed + i * 1009 + j * 9176)
    return uc, um
