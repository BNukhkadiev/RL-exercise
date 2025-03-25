import numpy as np


def simulate(bandit, agent, horizon: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Simulates interaction between bandit and agent.

    Returns:
        rewards: Array of obtained rewards.
        regret: Array of cumulative regret.
        optimal_choices: Binary array indicating if optimal arm was chosen.
    """
    rewards = np.zeros(horizon)
    regret = np.zeros(horizon)
    optimal_choices = np.zeros(horizon)
    optimal_arm = np.argmax(bandit.means)
    optimal_mean = bandit.means[optimal_arm]
    estimates = []
    for t in range(horizon):
        arm = agent.select_arm()
        reward = bandit.perform_action(arm)
        agent.update(arm, reward)

        rewards[t] = reward
        regret[t] = (optimal_mean - bandit.means[arm]) if bandit.dist_type == 'gaussian' else 0.0
        optimal_choices[t] = 1 if arm == optimal_arm else 0
        estimates.append(agent.values.copy())

        if t > 0:
            regret[t] += regret[t - 1]  # Cumulative regret

    return rewards, regret, optimal_choices, estimates