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
    def __init__(self, state_space, action_space, gamma=0.95, alpha=0.1, epsilon=0.05):
        super().__init__(state_space, action_space)
        self.q_table = np.zeros((state_space, action_space))
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)  # Exploration
        return np.argmax(self.q_table[state])  # Exploitation

    def observe(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + (1 - done) * self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error