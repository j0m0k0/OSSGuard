"""Paper-aligned game configuration and reward definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import expm1
from typing import Any


@dataclass(frozen=True)
class GameConfig:
    """Parameters for the finite-horizon OSSGuard stochastic game.

    ``signal_clean`` and ``signal_exploit`` are P(suspicious | C) and
    P(suspicious | X), respectively.
    """

    horizon: int = 200
    max_privilege: int = 100
    signal_clean: float = 0.30
    signal_exploit: float = 0.60
    omega_min: float = 3.0
    beta_contributor: float = 3.0
    beta_maintainer: float = 3.0
    exploit_multiplier_contributor: float = 3.0
    exploit_multiplier_maintainer: float = 3.0

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.max_privilege < 1:
            raise ValueError("horizon and max_privilege must be positive")
        for name in ("signal_clean", "signal_exploit"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.omega_min <= 0:
            raise ValueError("omega_min must be positive")
        if self.beta_contributor <= 0 or self.beta_maintainer <= 0:
            raise ValueError("beta parameters must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def privilege_value(privilege: int, max_privilege: int, omega_min: float, beta: float) -> float:
    """Compute Ωβ(s) = Ωmin + (exp(βs/Smax) - 1)/(exp(β) - 1)."""
    if max_privilege <= 0 or not 0 <= privilege <= max_privilege:
        raise ValueError("privilege must be in [0, max_privilege]")
    return omega_min + expm1(beta * privilege / max_privilege) / expm1(beta)


def rewards(privilege: int, contributor_action: int, maintainer_action: int,
            config: GameConfig) -> tuple[float, float]:
    """Return (contributor, maintainer) rewards.

    Actions: contributor 0=contribute, 1=exploit; maintainer 0=accept, 1=inspect.
    """
    if contributor_action == 1:
        contributor_value = privilege_value(privilege, config.max_privilege,
                                             config.omega_min, config.beta_contributor)
        maintainer_value = privilege_value(privilege, config.max_privilege,
                                            config.omega_min, config.beta_maintainer)
        rc = config.exploit_multiplier_contributor * contributor_value
        rc = rc if maintainer_action == 0 else -rc
        rm = (-config.exploit_multiplier_maintainer * maintainer_value
              if maintainer_action == 0 else config.exploit_multiplier_maintainer * maintainer_value)
        return rc, rm

    maintainer_value = privilege_value(privilege, config.max_privilege,
                                        config.omega_min, config.beta_maintainer)
    return 0.0, maintainer_value if maintainer_action == 0 else -maintainer_value
