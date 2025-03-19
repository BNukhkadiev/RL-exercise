

def simulate(bandit, agent, horizon: int) -> (np.ndarray, np.ndarray):
    """
    Simulates interaction between bandit and agent.
    
    Args:
        bandit: Instance of Bandit class.
        agent: Instance of Agent class.
        horizon: Number of rounds.
    
    Returns:
        rewards: Array of obtained rewards.
        regret: Array of cumulative regret.
    """
    rewards = np.zeros(horizon)
    regret = np.zeros(horizon)
    optimal_mean = max(bandit.means)

    for t in range(horizon):
        arm = agent.select_arm()
        reward = bandit.perform_action(arm)
        agent.update(arm, reward)

        rewards[t] = reward
        regret[t] = (optimal_mean - bandit.means[arm]) if bandit.dist_type == 'gaussian' else 0.0
        if t > 0:
            regret[t] += regret[t - 1]  # Cumulative regret

    return rewards, regret


import matplotlib.pyplot as plt

# Example configuration
K = 10
horizon = 10000
epsilon = 0.1
num_simulations = 50

all_regrets = []

for _ in range(num_simulations):
    bandit = Bandit(K=K, dist_type='gaussian')
    agent = EpsilonGreedyAgent(K=K, epsilon=epsilon)
    _, regret = simulate(bandit, agent, horizon)
    all_regrets.append(regret)

mean_regret = np.mean(all_regrets, axis=0)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(mean_regret, label=f"Epsilon-Greedy (ε={epsilon})")
plt.xlabel("Time steps")
plt.ylabel("Cumulative Regret")
plt.title("Epsilon-Greedy Regret over Time (Gaussian Bandit)")
plt.legend()
plt.grid()
plt.show()
