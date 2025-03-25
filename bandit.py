import numpy as np
from typing import List, Optional, Union, Iterable

np.random.seed(42)


class Bandit:
    """
    A Multi-Armed Bandit model supporting:
        - Gaussian Bandit (normally distributed arms)
        - Bernoulli Bandit (Bernoulli-distributed arms)
        - Reward-gap Bandit (introducing a structured reward difference)
    """

    def __init__(self, K: int, dist_type: str, means: Optional[Iterable[float]] = None, 
                 delta: Optional[float] = None):
        """
        Initializes the Bandit.

        Args:
            K (int): Number of arms.
            dist_type (str): Either 'gaussian' or 'bernoulli'.
            means (Optional[Iterable[float]]): User-defined means (or probabilities for Bernoulli).
            delta (Optional[float]): The reward gap Δ for 'reward-gap' mode.
        """
        if dist_type not in {'gaussian', 'bernoulli'}:
            raise ValueError(f"Invalid dist_type '{dist_type}'. Expected 'gaussian' or 'bernoulli'.")

        self.dist_type = dist_type
        self.K = K

        if means is not None:
            self._validate_means(means)
            self.means = list(means) if self.dist_type == 'gaussian' else None
            self.p = list(means) if self.dist_type == 'bernoulli' else None

        elif delta is not None:
            if delta <= 0:
                raise ValueError("Reward gap mode requires a positive delta value.")
            self.delta = delta
            self._initialize_reward_gap()

        else:
            self._random_init()

    def _validate_means(self, means: Iterable[float]) -> None:
        """Ensures the provided means/probabilities are valid."""
        if not isinstance(means, (list, np.ndarray)):
            raise TypeError("Means must be a list or a numpy array.")
        if len(means) != self.K:
            raise ValueError(f"For K={self.K} arms, {len(means)} values provided.")

    def _random_init(self) -> None:
        """Randomly initializes means for Gaussian bandits and probabilities for Bernoulli bandits."""
        if self.dist_type == 'gaussian':
            self.means = np.random.normal(0, 1, self.K).tolist()
        elif self.dist_type == 'bernoulli':
            self.p = np.random.uniform(0, 1, self.K).tolist()

    def _initialize_reward_gap(self) -> None:
        """Applies the reward-gap modification."""
        if self.dist_type == 'gaussian':
            base_means = np.random.normal(0, 1, self.K).tolist()
            mu_star = max(base_means)
            self.means = [mu_star - k * self.delta for k in range(self.K)]

        elif self.dist_type == 'bernoulli':
            base_p = np.random.uniform(0, 1, self.K).tolist()
            p_star = max(base_p)
            self.p = [max(p_star - k * self.delta, 0) for k in range(self.K)]

    def perform_action(self, i: int) -> float:
        """
        Simulates pulling arm i.

        Args:
            i (int): The index of the arm to pull.

        Returns:
            float: The reward obtained.
        """
        if not (0 <= i < self.K):
            raise IndexError(f"Invalid arm index {i}. Must be in range [0, {self.K-1}].")

        if self.dist_type == 'gaussian':
            return np.random.normal(self.means[i], 1)
        elif self.dist_type == 'bernoulli':
            return np.random.binomial(1, self.p[i])

        raise RuntimeError("Invalid mode. This should never happen.")

    def __repr__(self) -> str:
        """Returns a readable string representation of the Bandit model."""
        if self.dist_type == 'gaussian':
            means_str = ", ".join(f"{m:.2f}" for m in self.means)
            return f"Bandit(dist_type='gaussian', K={self.K}, means=[{means_str}])"
        elif self.dist_type == 'bernoulli':
            p_str = ", ".join(f"{p:.2f}" for p in self.p)
            return f"Bandit(dist_type='bernoulli', K={self.K}, p=[{p_str}])"
        return "Bandit(dist_type='unknown')"
