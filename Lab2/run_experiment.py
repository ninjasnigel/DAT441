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

action_dim = env.action_space.n
state_dim = env.observation_space.n
Q_MIN = 0.1
Q_MAX = 0.2
q_table = np.random.uniform(low=Q_MIN, high=Q_MAX, size=(state_dim, action_dim))

agents = [
    agentfile.QLearningAgent(state_dim, action_dim, q_table=np.copy(q_table)),
    agentfile.SARSA_Agent(state_dim, action_dim, q_table=np.copy(q_table)),
    agentfile.ExpectedSARSA_Agent(state_dim, action_dim, q_table=np.copy(q_table)),
    agentfile.Double_QLearningAgent(state_dim, action_dim, q_table=np.copy(q_table)),
]

print("Agents found and instantiated: ", agents)

for agent in agents:
    # Training loop
    num_episodes = 1000
    max_steps_per_episode = 100
    rewards_all_episodes = []

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

    # Plot cumulative rewards
    cumulative_rewards = np.cumsum(rewards_all_episodes)
    plt.plot(cumulative_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.show()