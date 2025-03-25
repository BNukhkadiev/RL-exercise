import numpy as np
from abc import ABC, abstractmethod

class BanditAgent(ABC):
    def __init__(self, K: int):
        self.K = K
        self.counts = np.zeros(K)  # Number of times each arm is pulled
        self.values = np.zeros(K)  # Estimated mean reward for each arm

    @abstractmethod
    def select_arm(self) -> int:
        pass

    def update(self, chosen_arm: int, reward: float) -> None:
        """Updates estimates with new reward."""
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        # Q_hat(a_t) = Q_hat(a_t) + 1/T(a) * (X_t - Q_hat(a_t))
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / n


class ETCAgent(BanditAgent):
    def __init__(self, K: int, m: int, horizon: int):
        super().__init__(K)
        self.m = m
        self.horizon = horizon
        self.total_count = 0
        self.best_arm = None
        self.exploration_done = False

    def select_arm(self) -> int:
        self.total_count += 1

        if self.total_count < self.m * self.K:
            # Explore each arm m times
            arm = (self.total_count - 1) % self.K
            if self.total_count == (self.K * self.m) - 1:
                self.best_arm = np.argmax(self.values)  # Choose the best estimated arm
            return arm
        else:
            # Commit to the best arm
            return self.best_arm


class EpsilonGreedyAgent(BanditAgent):
    def __init__(self, K: int, epsilon: float):
        super().__init__(K)
        self.epsilon = epsilon

    def select_arm(self) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.K)
        else:
            return np.argmax(self.values)


class GreedyAgent(BanditAgent):
    """
    Very retarded agent
    """
    def __init__(self, K: int):
        super().__init__(K)

    def select_arm(self) -> int:
        return np.argmax(self.values)
    

class ExploreThenEpsilonGreedyAgent(BanditAgent):
    def __init__(self, K: int, d: float, C: float):
        super().__init__(K)
        self.d = d
        self.C = C
        self.t = 0  # Total time steps

    def epsilon(self) -> float:
        """Compute ε_t = min{1, CK / (d^2 * t)}"""
        if self.t == 0:
            return 1.0  # To avoid division by zero
        return min(1.0, (self.C * self.K) / (self.d ** 2 * self.t))

    def select_arm(self) -> int:
        self.t += 1
        eps = self.epsilon()
        if np.random.rand() < eps:
            # Exploration
            return np.random.randint(self.K)
        else:
            # Exploitation
            return int(np.argmax(self.values))


class UCB1Agent(BanditAgent):
    def __init__(self, K: int):
        super().__init__(K)
        self.total_count = 0  # To keep track of total time steps

    def select_arm(self) -> int:
        self.total_count += 1
        # Ensure all arms are pulled at least once
        for arm in range(self.K):
            if self.counts[arm] == 0:
                return arm

        # Compute UCB for all arms
        ucb_values = self.values + np.sqrt(2 * np.log(self.total_count) / self.counts)
        return np.argmax(ucb_values)


class UCBAgent(BanditAgent):
    def __init__(self, K: int, delta: float):
        super().__init__(K)
        self.delta = delta
        self.total_counts = 0  # Time step t

    def select_arm(self) -> int:
        self.total_counts += 1
        ucb_values = np.zeros(self.K)
        
        for a in range(self.K):
            if self.counts[a] == 0:
                ucb_values[a] = float('inf')
            else:
                bonus = np.sqrt((2 * np.log(1 / self.delta)) / self.counts[a])
                ucb_values[a] = self.values[a] + bonus

        return int(np.argmax(ucb_values))


class SubGaussianUCBAgent(BanditAgent):
    def __init__(self, K: int, sigma: float):
        super().__init__(K)
        self.sigma = sigma
        self.total_pulls = 0

    def select_arm(self) -> int:
        self.total_pulls += 1
        ucb_values = np.zeros(self.K)

        for a in range(self.K):
            if self.counts[a] == 0:
                ucb_values[a] = float('inf')  # Force exploration
            else:
                bonus = np.sqrt((4 * self.sigma ** 2 * np.log(self.total_pulls)) / self.counts[a])
                ucb_values[a] = self.values[a] + bonus

        return int(np.argmax(ucb_values))
    

