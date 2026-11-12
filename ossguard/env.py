"""Turn-based OSSGuard game environment."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .model import GameConfig, rewards

CONTRIBUTE, EXPLOIT = 0, 1
ACCEPT, INSPECT = 0, 1
CLEAN, SUSPICIOUS = 0, 1


@dataclass(frozen=True)
class Step:
    contributor_observation: np.ndarray
    maintainer_observation: np.ndarray
    contributor_action: int
    maintainer_action: int
    contributor_reward: float
    maintainer_reward: float
    terminated: bool
    truncated: bool
    info: dict


class OSSGuardEnv:
    """Finite-horizon game; a detected exploit moves to an absorbing terminal state."""

    def __init__(self, config: GameConfig | None = None, seed: int | None = None):
        self.config = config or GameConfig()
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.privilege = 0
        self.t = 0
        self.done = False
        self.last_signal = CLEAN
        self._pending_action: int | None = None
        self._pending_privilege: int | None = None
        return self._contributor_obs(), self._maintainer_obs()

    def _contributor_obs(self) -> np.ndarray:
        return np.asarray([self.privilege], dtype=np.float32)

    def _maintainer_obs(self) -> np.ndarray:
        return np.asarray([self.privilege, float(self.last_signal)],
                          dtype=np.float32)

    def begin_turn(self, contributor_action: int) -> np.ndarray:
        """Apply contributor action and emit the noisy signal observed by the maintainer."""
        if self.done:
            raise RuntimeError("episode is over; call reset() before stepping again")
        if contributor_action not in (CONTRIBUTE, EXPLOIT):
            raise ValueError("contributor_action must be 0 (contribute) or 1 (exploit)")
        if self._pending_action is not None:
            raise RuntimeError("resolve the current turn before starting another")
        p_suspicious = (self.config.signal_exploit if contributor_action == EXPLOIT
                        else self.config.signal_clean)
        self.last_signal = int(self.rng.random() < p_suspicious)
        self._pending_action = contributor_action
        self._pending_privilege = self.privilege
        return self._maintainer_obs()

    def end_turn(self, maintainer_action: int) -> Step:
        """Resolve the maintainer response to the current signal."""
        if self.done or self._pending_action is None or self._pending_privilege is None:
            raise RuntimeError("begin_turn() must be called before end_turn()")
        if maintainer_action not in (ACCEPT, INSPECT):
            raise ValueError("maintainer_action must be 0 (accept) or 1 (inspect)")
        contributor_action = self._pending_action
        prior_privilege = self._pending_privilege
        rc, rm = rewards(prior_privilege, contributor_action, maintainer_action, self.config)

        detected = contributor_action == EXPLOIT and maintainer_action == INSPECT
        if detected:
            self.done = True
        else:
            self.privilege = min(self.privilege + 1, self.config.max_privilege)
        self.t += 1
        truncated = self.t >= self.config.horizon and not self.done
        self.done = self.done or truncated
        self._pending_action = None
        self._pending_privilege = None
        return Step(self._contributor_obs(), self._maintainer_obs(), contributor_action,
                    maintainer_action, rc, rm, terminated=detected, truncated=truncated,
                    info={"privilege_before": prior_privilege, "privilege_after": self.privilege,
                          "signal": self.last_signal, "detected": detected})

    def step(self, contributor_action: int, maintainer_action: int) -> Step:
        """Convenience transition for callers that already selected both actions."""
        self.begin_turn(contributor_action)
        return self.end_turn(maintainer_action)
