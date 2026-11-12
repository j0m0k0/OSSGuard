"""Command-line interface for the paper's reproducible experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch

from .experiments import run_fixed_pool, run_heuristic_baseline
from .model import GameConfig
from .psro import run_psro


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="ossguard", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output", type=Path, default=Path("results"))
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--episodes", type=int, default=300)
    commands.add_parser("heuristics", parents=[common],
                        help="evaluate the heuristic policy matrix")
    train = commands.add_parser("fixed-pool", parents=[common],
                                help="train a fixed-pool maintainer response")
    train.add_argument("--method", choices=["mss-unif", "mss-hard", "ibr", "specialists"],
                       default="mss-unif")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--steps-per-epoch", type=int, default=5_000)
    train.add_argument("--temperature", type=float, default=0.25)
    psro = commands.add_parser("psro", parents=[common], help="run expanding-population PSRO")
    psro.add_argument("--iterations", type=int, default=200)
    psro.add_argument("--steps-per-oracle", type=int, default=5_000)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    _seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = GameConfig()
    metadata = {"command": args.command, "seed": args.seed, "game_config": cfg.to_dict(),
                "episodes_per_pair": args.episodes}
    if args.command == "heuristics":
        run_heuristic_baseline(args.output, args.episodes, args.seed, cfg)
    elif args.command == "fixed-pool":
        metadata.update({"method": args.method, "epochs": args.epochs,
                         "steps_per_epoch": args.steps_per_epoch,
                         "mss_hard_temperature": args.temperature})
        run_fixed_pool(args.output, args.method, args.epochs, args.steps_per_epoch,
                       args.episodes, args.seed, args.temperature, cfg)
    else:
        metadata.update({"iterations": args.iterations,
                         "steps_per_oracle": args.steps_per_oracle})
        run_psro(args.output, args.iterations, args.steps_per_oracle,
                 args.episodes, args.seed, cfg)
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                                    encoding="utf-8")
    print(f"Wrote experiment artifacts to {args.output.resolve()}")


if __name__ == "__main__":
    main()
