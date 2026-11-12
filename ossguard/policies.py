"""Rule-based policy families used in the paper's evaluations."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .env import CLEAN, EXPLOIT, SUSPICIOUS


@dataclass
class HeuristicPolicy:
    name: str
    role: str
    period: int = 5
    threshold: int = 25
    rate: float = 0.5
    tough_start: int = 15
    tough_end: int = 75
    prob: float = 0.5
    turn: int = 0

    def __post_init__(self) -> None:
        if self.role not in ("contributor", "maintainer"):
            raise ValueError("role must be 'contributor' or 'maintainer'")
        if self.period < 1:
            raise ValueError("period must be positive")
        if not 0 <= self.prob <= 1 or not 0 <= self.rate <= 1:
            raise ValueError("probability parameters must lie in [0, 1]")

    def reset(self) -> None:
        self.turn = 0

    def act(self, observation: np.ndarray, rng: np.random.Generator) -> int:
        privilege = int(round(float(observation[0])))
        signal = int(observation[1]) if self.role == "maintainer" else CLEAN
        if self.name == "Random":
            self.turn += 1
            return int(rng.integers(0, 2))
        if self.name == "Periodic":
            action = int((self.turn + 1) % self.period == 0)
            self.turn += 1
            return action
        if self.name == "Gate":
            action = int(privilege >= self.threshold and rng.random() < self.prob)
            self.turn += 1
            return action
        if self.name == "Ramp":
            action = int(rng.random() < self.rate * privilege / 100)
            self.turn += 1
            return action
        if self.name == "ToughStart":
            action = int(privilege < self.tough_start or rng.random() < 1 - privilege / 100)
            self.turn += 1
            return action
        if self.name == "ToughEnd":
            action = int(rng.random() < privilege / 100 if privilege < self.tough_end else True)
            self.turn += 1
            return action
        if self.name == "Greedy":
            self.turn += 1
            return EXPLOIT if self.role == "contributor" else int(signal == SUSPICIOUS)
        self.turn += 1
        raise ValueError(f"unknown heuristic: {self.name}")


def known_contributors(unseen: bool = False) -> list[HeuristicPolicy]:
    if unseen:
        return [HeuristicPolicy("Periodic", "contributor", period=8),
                HeuristicPolicy("Gate", "contributor", threshold=50),
                HeuristicPolicy("Ramp", "contributor", rate=0.75),
                HeuristicPolicy("ToughStart", "contributor", tough_start=25),
                HeuristicPolicy("ToughEnd", "contributor", tough_end=90)]
    return [HeuristicPolicy("Periodic", "contributor", period=5),
            HeuristicPolicy("Gate", "contributor", threshold=25),
            HeuristicPolicy("Ramp", "contributor", rate=0.5),
            HeuristicPolicy("ToughStart", "contributor", tough_start=15),
            HeuristicPolicy("ToughEnd", "contributor", tough_end=75)]


def heuristic_maintainers() -> list[HeuristicPolicy]:
    return [HeuristicPolicy("Random", "maintainer"),
            HeuristicPolicy("Periodic", "maintainer", period=5),
            HeuristicPolicy("Gate", "maintainer", threshold=25),
            HeuristicPolicy("Ramp", "maintainer", rate=0.5),
            HeuristicPolicy("ToughStart", "maintainer", tough_start=15),
            HeuristicPolicy("ToughEnd", "maintainer", tough_end=75),
            HeuristicPolicy("Greedy", "maintainer")]
