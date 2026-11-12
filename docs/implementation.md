# Implementation notes

This page maps the mathematical model and experiment protocol in the OSSGuard paper to the source code. It also records choices that are necessary to make executable experiments from settings the paper leaves open.

## Game definition

The contributor observes privilege `s ∈ {0, …, Smax}` and chooses a contribution `C` or exploit `X`. The maintainer observes privilege and the noisy signal `z`, then chooses accept `A` or inspect `I`. The binary signal parameters are

`P(z = suspicious | C) = pC` and `P(z = suspicious | X) = pX`.

The privilege-value function and reward cases are implemented in `ossguard/model.py`. A detected exploit (`X, I`) ends the episode; all other action pairs raise privilege by one, capped at `Smax`. Episodes are truncated after the configured finite horizon. The environment splits each turn into `begin_turn` and `end_turn` so the maintainer can choose after seeing the signal.

The maintainer observation is `[s, z]`; the contributor observation is `[s]`, matching the paper's state definition. Heuristic periodic policies retain an episode-local clock, since the paper's observation specification does not list time as a state observation but its periodic baseline depends on elapsed steps.

## Heuristic baselines

Contributor and maintainer versions use the same rule families, with the direction of action mapped to exploit or inspect. Defaults follow the paper: period 5, Gate threshold 25 and probability 0.5, Ramp rate 0.5, Tough Start threshold 15, and Tough End threshold 75. Greedy always exploits as a contributor and inspects only on suspicious signals as a maintainer. The fixed training pool is Periodic(5), Gate(25), Ramp(0.5), ToughStart(15), and ToughEnd(75). Transfer evaluation uses Periodic(8), Gate(50), Ramp(0.75), ToughStart(25), and ToughEnd(90).

## DQN and fixed-pool methods

The response oracle is a Q-network with hidden widths 120 and 84 and ReLU activations. It uses the paper's Adam, replay, discount, target update, learning-start, and epsilon schedule values documented in the README. For maintainer training, one contributor policy is drawn from the current mixture at the beginning of each episode and remains fixed through that episode. MSS-Hard computes softmax weights from negative maintainer returns with temperature `0.25` by default; the paper describes a temperature but does not state its numerical value, so this value is an explicit configurable implementation default. IBR cycles through the five known contributors in the fixed order, allocating equal portions of the requested epoch count to each. Specialists use independent DQN models, one per known contributor.

The signal sweep freezes the trained MSS-Unif policy and evaluates six matched signal pairs from `[0.5, 0.5]` through `[0.0, 1.0]`, as specified in the paper.

## PSRO and equilibrium analysis

The expanding-population mode begins with all seven heuristic families for both players, evaluates the empirical payoff matrices, solves the restricted game, trains one DQN response per player against the opposing meta-strategy, then adds the response policies and evaluates the new payoff row and column. The reported NashConv is computed from the final mixed profile and best unilateral deviations within the current restricted population.

The paper does not state which restricted-game solver or equilibrium selection policy generated its tables. This implementation selects the first Nashpy support-enumerated equilibrium for games with at most 12 strategies on each side. For larger games it uses 500 simultaneous external-regret-matching updates and reports NashConv against those resulting mixtures. This is a documented reproducibility choice, not a claim that every resulting mixture is a Nash equilibrium.

## Reproducibility boundaries

The default parameters and episode counts reproduce the paper's stated protocol, but the paper's exact trained checkpoints, random seeds, and all implementation details are not supplied here. Thus, reruns should be treated as independent reproductions and not expected to match every decimal in the paper. Every CLI run writes its seed, game parameters, and run-scale arguments to `run_metadata.json`.
