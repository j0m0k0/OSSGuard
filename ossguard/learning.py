"""DQN response oracles and fixed-pool population training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from .env import OSSGuardEnv
from .model import GameConfig
from .policies import HeuristicPolicy, known_contributors


class QNetwork(nn.Module):
    def __init__(self, observation_size: int, action_count: int = 2,
                 hidden_sizes: tuple[int, int] = (120, 84)):
        super().__init__()
        first, second = hidden_sizes
        self.layers = nn.Sequential(nn.Linear(observation_size, first), nn.ReLU(),
                                    nn.Linear(first, second), nn.ReLU(),
                                    nn.Linear(second, action_count))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@dataclass
class DQNConfig:
    learning_rate: float = 2.5e-4
    replay_capacity: int = 100_000
    batch_size: int = 128
    gamma: float = 0.95
    target_update_period: int = 2_000
    train_frequency: int = 4
    tau: float = 0.01
    learning_starts: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    hidden_sizes: tuple[int, int] = (120, 84)


class DQNPolicy:
    """Frozen greedy policy compatible with the heuristic policy interface."""

    def __init__(self, network: QNetwork, role: str, device: str = "cpu"):
        self.network = network.eval()
        self.role = role
        self.device = device

    def act(self, observation: np.ndarray, rng: np.random.Generator) -> int:
        del rng
        with torch.no_grad():
            x = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.network(x).argmax(dim=1).item())


Transition = tuple[np.ndarray, int, float, np.ndarray, bool]


def _soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_p, online_p in zip(target.parameters(), online.parameters()):
            target_p.lerp_(online_p, tau)


def train_response_oracle(
    role: str,
    opponents: Sequence[HeuristicPolicy | DQNPolicy],
    config: GameConfig | None = None,
    steps: int = 5_000,
    seed: int = 0,
    dqn: DQNConfig | None = None,
    opponent_weights: Sequence[float] | None = None,
    on_loss: Callable[[float], None] | None = None,
    initial_state: dict[str, torch.Tensor] | None = None,
) -> DQNPolicy:
    """Train one DQN best response, sampling and freezing an opponent per episode."""
    if role not in ("contributor", "maintainer"):
        raise ValueError("role must be 'contributor' or 'maintainer'")
    if not opponents:
        raise ValueError("at least one opponent is required")
    if steps < 1:
        raise ValueError("steps must be positive")

    cfg = config or GameConfig()
    hparams = dqn or DQNConfig()
    rng = np.random.default_rng(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs_size = 1 if role == "contributor" else 2
    online = QNetwork(obs_size, hidden_sizes=hparams.hidden_sizes).to(device)
    if initial_state is not None:
        online.load_state_dict(initial_state)
    target = QNetwork(obs_size, hidden_sizes=hparams.hidden_sizes).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=hparams.learning_rate)
    replay: deque[Transition] = deque(maxlen=hparams.replay_capacity)
    env = OSSGuardEnv(cfg, seed=seed)
    weights = None if opponent_weights is None else np.asarray(opponent_weights, dtype=float)
    if weights is not None:
        if len(weights) != len(opponents) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("opponent_weights must be nonnegative and match the opponent pool")
        weights = weights / weights.sum()

    global_step = 0
    episode_opponent: HeuristicPolicy | DQNPolicy = opponents[0]
    episode_step = 0
    pending_maintainer: tuple[np.ndarray, int, float] | None = None
    losses: list[float] = []

    while global_step < steps:
        if episode_step == 0:
            pick = int(rng.choice(len(opponents), p=weights))
            episode_opponent = opponents[pick]
            if hasattr(episode_opponent, "reset"):
                episode_opponent.reset()
            obs_c, _ = env.reset()
            pending_maintainer = None

        epsilon_fraction = min(1.0, global_step / max(1, steps * 0.5))
        epsilon = hparams.epsilon_start + epsilon_fraction * (hparams.epsilon_end - hparams.epsilon_start)

        if role == "contributor":
            state = obs_c.copy()
            action = (int(rng.integers(2)) if rng.random() < epsilon else
                      int(online(torch.as_tensor(state, device=device).unsqueeze(0)).argmax(1).item()))
            maint_obs = env.begin_turn(action)
            maint_action = episode_opponent.act(maint_obs, rng)
            result = env.end_turn(maint_action)
            next_obs = result.contributor_observation.copy()
            replay.append((state, action, result.contributor_reward, next_obs,
                           result.terminated or result.truncated))
            obs_c = next_obs
        else:
            # Contributor policy chooses first; signal then defines the DQN's state.
            contributor_action = episode_opponent.act(obs_c, rng)
            state = env.begin_turn(contributor_action).copy()
            action = (int(rng.integers(2)) if rng.random() < epsilon else
                      int(online(torch.as_tensor(state, device=device).unsqueeze(0)).argmax(1).item()))
            result = env.end_turn(action)
            if pending_maintainer is not None:
                old_state, old_action, old_reward = pending_maintainer
                replay.append((old_state, old_action, old_reward, state.copy(), False))
            if result.terminated or result.truncated:
                replay.append((state, action, result.maintainer_reward,
                               np.zeros_like(state), True))
                pending_maintainer = None
            else:
                pending_maintainer = (state, action, result.maintainer_reward)
            obs_c = result.contributor_observation

        global_step += 1
        episode_step += 1

        if global_step >= hparams.learning_starts and global_step % hparams.train_frequency == 0 and len(replay) >= hparams.batch_size:
            batch = random.sample(replay, hparams.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            s = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
            a = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
            r = torch.as_tensor(rewards, dtype=torch.float32, device=device)
            ns = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=device)
            done = torch.as_tensor(dones, dtype=torch.float32, device=device)
            with torch.no_grad():
                y = r + hparams.gamma * (1 - done) * target(ns).max(dim=1).values
            q = online(s).gather(1, a).squeeze(1)
            loss = nn.functional.mse_loss(q, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            if on_loss:
                on_loss(float(loss.item()))
        if global_step % hparams.target_update_period == 0:
            _soft_update(target, online, hparams.tau)

        if env.done:
            env.reset()
            episode_step = 0
    return DQNPolicy(online, role, device=device)


def train_fixed_pool(
    method: str = "mss-unif", epochs: int = 100, steps_per_epoch: int = 5_000,
    seed: int = 0, config: GameConfig | None = None, temperature: float = 0.25,
    evaluation_episodes: int = 300,
) -> tuple[DQNPolicy, list[dict[str, float]]]:
    """Train an MSS maintainer response over the paper's fixed contributor pool."""
    contributors = known_contributors()
    weights = np.full(len(contributors), 1.0 / len(contributors))
    losses: list[float] = []
    policy: DQNPolicy | None = None
    records: list[dict[str, float]] = []
    if method not in ("mss-unif", "mss-hard", "ibr"):
        raise ValueError("method must be mss-unif, mss-hard, or ibr")
    if epochs < 1 or steps_per_epoch < 1:
        raise ValueError("epochs and steps_per_epoch must be positive")
    if method == "mss-hard" and temperature <= 0:
        raise ValueError("temperature must be positive for mss-hard")
    for epoch in range(epochs):
        if method == "ibr":
            idx = min(len(contributors) - 1, epoch * len(contributors) // epochs)
            train_opponents = [contributors[idx]]
            train_weights = None
        else:
            train_opponents = contributors
            train_weights = weights
        policy = train_response_oracle("maintainer", train_opponents, config,
                                       steps=steps_per_epoch, seed=seed + epoch,
                                       opponent_weights=train_weights,
                                       on_loss=losses.append,
                                       initial_state=(policy.network.state_dict() if policy else None))
        if method == "mss-hard":
            from .evaluation import evaluate_pair
            returns = [evaluate_pair(cfg=config, contributor=opp, maintainer=policy,
                                     episodes=evaluation_episodes, seed=seed + 100_000 + epoch * 17 + j)[1]
                       for j, opp in enumerate(contributors)]
            logits = -np.asarray(returns) / max(temperature, 1e-8)
            logits -= logits.max()
            weights = np.exp(logits) / np.exp(logits).sum()
        if policy is not None:
            records.append({"epoch": float(epoch + 1), "loss": float(np.mean(losses[-100:])) if losses else 0.0})
    assert policy is not None
    return policy, records


def train_specialists(config: GameConfig | None = None, steps: int = 500_000,
                      seed: int = 0) -> list[DQNPolicy]:
    """Train one independent maintainer response for each known contributor."""
    policies = []
    for idx, contributor in enumerate(known_contributors()):
        policies.append(train_response_oracle("maintainer", [contributor], config,
                                              steps=steps, seed=seed + idx))
    return policies
