import argparse
import gymnasium as gym
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

# Load the agent module
parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="agent.py")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)

# Initialize the environment
try:
    env = gym.make(args.env, is_slippery=False)
    print("Loaded ", args.env)
except:
    file_name, env_name = args.env.split(":")
    gym.envs.register(
        id=env_name + "-v0",
        entry_point=args.env,
    )
    env = gym.make(env_name + "-v0")
    print("Loaded", args.env)


# Constants
action_dim = env.action_space.n
state_dim = env.observation_space.n
num_runs = 5
num_episodes = 100
max_steps_per_episode = 100
Q_VALUE = 0.1

# Initialize list to hold cumulative rewards for each agent across runs
rewards_all = []

# Create agents with random Q-table initialization
for agent_class in [agentfile.QLearningAgent, agentfile.SARSA_Agent, agentfile.ExpectedSARSA_Agent, agentfile.Double_QLearningAgent]:
    agent_rewards = []  # To store rewards from each run for the current agent
    
    for run in range(num_runs):
        #q_table = np.random.uniform(low=Q_MIN, high=Q_MAX, size=(state_dim, action_dim))
        q_table = np.full(shape=(state_dim, action_dim), fill_value = Q_VALUE)
        agent = agent_class(state_dim, action_dim, q_table=np.copy(q_table))
        
        rewards_all_episodes = []  # To store rewards for this run
        
        # Training loop for a single run
        for episode in range(num_episodes):
            state, info = env.reset()
            done = False
            total_rewards = 0

            for step in range(max_steps_per_episode):
                action = agent.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.observe(state, action, reward, next_state, done)
                state = next_state
                total_rewards += reward

                if done:
                    break
            
            rewards_all_episodes.append(total_rewards)
        
        agent_rewards.append(rewards_all_episodes)  # Store results of this run
    
    rewards_all.append(agent_rewards)  # Store results for the current agent

# Calculate means and 95% confidence intervals
mean_rewards = []
ci_rewards = []

for agent_rewards in rewards_all:
    cumulative_rewards_per_run = [np.cumsum(rewards) for rewards in agent_rewards]
    
    # Convert to numpy array to calculate mean and CI
    cumulative_rewards_per_run = np.array(cumulative_rewards_per_run)
    
    mean_cumulative_rewards = np.mean(cumulative_rewards_per_run, axis=0)
    std_cumulative_rewards = np.std(cumulative_rewards_per_run, axis=0)
    
    # Calculate the 95% confidence interval (1.96 * std / sqrt(n))
    ci_cumulative_rewards = 1.96 * std_cumulative_rewards / np.sqrt(num_runs)
    
    mean_rewards.append(mean_cumulative_rewards)
    ci_rewards.append(ci_cumulative_rewards)

# Colors for each agent
colors = ['blue', 'green', 'red', 'purple']

# Create subplots: 2 rows, 2 columns
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

agent_names = ['QLearningAgent', 'SARSA_Agent', 'ExpectedSARSA_Agent', 'Double_QLearningAgent']
agent_classes = [agentfile.QLearningAgent, agentfile.SARSA_Agent, agentfile.ExpectedSARSA_Agent, agentfile.Double_QLearningAgent]

for i, ax in enumerate(axs.flatten()):
    ax.plot(mean_rewards[i], label=f'{agent_classes[i].__name__}', color=colors[i], linewidth=2)
    ax.fill_between(np.arange(num_episodes), 
                    mean_rewards[i] - ci_rewards[i], 
                    mean_rewards[i] + ci_rewards[i], 
                    color=colors[i], 
                    alpha=0.2)
    ax.set_title(f'{agent_classes[i].__name__}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.legend()

plt.tight_layout()
plt.savefig('cumulative_rewards_subplots_with_colors.png')
plt.show()


# Q_TESTS = [(-0.5, -0.25), (0.01, 0.02), (0.1, 0.2), (0.5, 1.0), (1.0, 2.0), (5.0, 10.0)]
Q_TESTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]


q_rewards = []

last_q_tables = []

# for Q_MIN, Q_MAX in Q_TESTS:
for Q_VALUE in Q_TESTS:
    num_episodes = 100
    max_steps_per_episode = 100
    q_table = np.full(shape=(state_dim, action_dim), fill_value = Q_VALUE)
    # q_table = np.random.uniform(low=Q_MIN, high=Q_MAX, size=(state_dim, action_dim))
    rewards_all_episodes = []
    agent = agentfile.QLearningAgent(state_dim, action_dim, q_table=q_table)
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        total_rewards = 0

        for step in range(max_steps_per_episode):
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.observe(state, action, reward, next_state, done)

            state = next_state
            total_rewards += reward

            if done:
                break

        rewards_all_episodes.append(total_rewards)
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Total Reward = {total_rewards}")
            #env.render()
        
    print("Training completed.")
    q_rewards.append(rewards_all_episodes)
    last_q_tables.append(agent.q_table)

# Plot cumulative rewards
for i, rewards in enumerate(q_rewards):
    cumulative_rewards = np.cumsum(rewards)
  #  plt.plot(cumulative_rewards, label=f"Q_MIN={Q_TESTS[i][0]}, Q_MAX={Q_TESTS[i][1]}")
    plt.plot(cumulative_rewards, label=f"Q_VALUE={Q_TESTS[i]}")
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')

plt.legend()
plt.savefig('cumulative_rewards_Q_DIFF.png')
plt.show()
#save plot


def plot_q_table(q_table, env):
    # Compute the sum of Q-values for the alpha scale (transparency)
    q_values_sum = np.sum(q_table, axis=1).reshape(4, 4)

    # Create the plot with a more visually appealing figure size
    fig, ax = plt.subplots(figsize=(6, 6))

    # Use a more visually appealing color map, e.g., "plasma"
    cmap = plt.get_cmap("plasma")
    norm = plt.Normalize(vmin=np.min(q_values_sum), vmax=np.max(q_values_sum))
    
    # Plot the squares with the alpha scale
    for i in range(4):
        for j in range(4):
            rect = plt.Rectangle([j, i], 1, 1, color=cmap(norm(q_values_sum[i, j])), alpha=0.8)
            ax.add_patch(rect)

            # Display the actual Q-value in the square (average of Q-values)
            avg_q_value = np.mean(q_table[i * 4 + j])
            ax.text(j + 0.5, i + 0.5, f'{avg_q_value:.2f}', ha='center', va='center', color='white', fontsize=14, fontweight='bold')

    # Get the greedy policy (argmax of Q-values)
    greedy_policy = np.argmax(q_table, axis=1).reshape(4,4)
    
    # Plot arrows inside the squares, avoiding overlap with the numbers
    for i in range(4):
        for j in range(4):
            action = greedy_policy[i, j]
            if action == 0:  # left
                dx, dy = -0.25, 0
            elif action == 1:  # down
                dx, dy = 0, 0.25
            elif action == 2:  # right
                dx, dy = 0.25, 0
            elif action == 3:  # up
                dx, dy = 0, -0.25
            ax.arrow(j + 0.5, i + 0.5, dx, dy, head_width=0.1, head_length=0.1, fc='black', ec='black')

    # Set plot limits and remove axis labels
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set title and adjust layout
    plt.title("Greedy policy", fontsize=16, pad=20)
    plt.gca().invert_yaxis()  # Flip y-axis to match grid layout
    plt.tight_layout()
    plt.show()

# Example usage:
plot_q_table(last_q_tables[0], env)