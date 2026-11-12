"""Computational forms of the paper's posterior-threshold characterization."""

from __future__ import annotations


def posterior_exploit_probability(prior: float, p_signal_given_exploit: float,
                                  p_signal_given_clean: float) -> float:
    """Bayes update P(X | z) from a prior and the likelihood of one signal."""
    for name, value in (("prior", prior), ("p_signal_given_exploit", p_signal_given_exploit),
                        ("p_signal_given_clean", p_signal_given_clean)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    numerator = prior * p_signal_given_exploit
    denominator = numerator + (1.0 - prior) * p_signal_given_clean
    if denominator == 0:
        raise ValueError("the observed signal has zero probability under the prior")
    return numerator / denominator


def inspection_threshold(clean_acceptance_advantage: float,
                         exploit_inspection_advantage: float) -> float:
    """Return ΔC/(ΔC+ΔX) under the theorem's reward-dominance assumptions."""
    if clean_acceptance_advantage < 0 or exploit_inspection_advantage < 0:
        raise ValueError("both advantage terms must be nonnegative")
    total = clean_acceptance_advantage + exploit_inspection_advantage
    if total <= 0:
        raise ValueError("the threshold is undefined when both advantages are zero")
    return clean_acceptance_advantage / total


def bayes_optimal_response(posterior_exploit: float, threshold: float) -> str:
    """Choose inspect at or above the posterior threshold, accept below it."""
    if not 0 <= posterior_exploit <= 1 or not 0 <= threshold <= 1:
        raise ValueError("posterior and threshold must lie in [0, 1]")
    return "inspect" if posterior_exploit >= threshold else "accept"
