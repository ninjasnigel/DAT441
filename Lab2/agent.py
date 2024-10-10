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
    
class SARSA_Agent(Agent):
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

        if np.random.rand() < self.epsilon:
            next_action = np.random.randint(self.action_space)  
        else:
            next_action = np.argmax(self.q_table[state])

        self.q_table[state, action] += self.alpha * (reward +  self.gamma * self.q_table[next_state, next_action] - self.q_table[state, action])



class ExpectedSARSA_Agent(Agent):
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
        # Calculate the expected value of the next state
        expected_value = 0
        for next_action in range(self.action_space):
            if next_action == np.argmax(self.q_table[next_state]):
                prob = 1 - self.epsilon + (self.epsilon / self.action_space)  # Probability of taking the best action
            else:
                prob = self.epsilon / self.action_space  # Probability of taking a suboptimal action
            
            expected_value += prob * self.q_table[next_state, next_action]

        # Update Q-value using the Expected SARSA update rule
        self.q_table[state, action] += self.alpha * (reward + self.gamma * expected_value - self.q_table[state, action])


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
        self.q_table[state, action] += self.alpha * (reward +  self.gamma * self.q_table[next_state, best_next_action] - self.q_table[state, action])


class Double_QLearningAgent(Agent):
    def __init__(self, state_space, action_space, gamma=0.95, alpha=0.1, epsilon=0.05, coin=0.5):
        super().__init__(state_space, action_space)
        self.q_table_1 = np.zeros((state_space, action_space))
        self.q_table_2 = np.zeros((state_space, action_space))
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.coin = coin

    def act(self, state):

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)  # Exploration
        else:
            combined_q_tables = (self.q_table_1[state] + self.q_table_2[state]) / 2
            return np.argmax(combined_q_tables)  # Exploitation
        
    def observe(self, state, action, reward, next_state, done):
        if np.random.rand() < self.coin:
             self.q_table_1[state][action] += self.alpha * (reward + self.gamma * self.q_table_2[next_state][np.argmax(self.q_table_1[next_state])]- self.q_table_1[state][action])
        else:
             self.q_table_2[state][action] += self.alpha * (reward + self.gamma * self.q_table_1[next_state][np.argmax(self.q_table_2[next_state])] - self.q_table_2[state][action])