import numpy as np

class Agent(object):
    """The world's simplest agent!"""
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space

    def observe(self, observation, reward, done):
        #Add your code here
        #print(f"Observation: {observation}, Reward: {reward}, Done: {done}")
        pass
    def act(self, observation):
        #Add your code here
        return np.random.randint(self.action_space)
    
class QLearningAgent(Agent):
    def __init__(self, state_space, action_space, gamma=0.95, alpha=0.8, epsilon=1.0):
        super().__init__(state_space, action_space)
        self.q_table = np.zeros((state_space, action_space))
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def map_state_to_index(self, state):
        # This is an example for mapping a tuple state to an index
        return hash(state) % self.state_space  # Ensure it fits within the table size

    def observe(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def act(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            # Explore: select a random action
            action = np.random.randint(self.action_space)
        else:
            # Exploit: select the action with the highest Q-value for the current state
            action = np.argmax(self.q_table[state])
        return action


