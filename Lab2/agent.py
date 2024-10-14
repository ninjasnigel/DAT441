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
        self.q_table = np.random.uniform(low=0.1, high=0.2, size=(state_space, action_space))
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)
        return np.argmax(self.q_table[state])

    def observe(self, state, action, reward, next_state, done):
        
        if np.random.rand() < self.epsilon:
            next_action = np.random.randint(self.action_space)  
        else:
            next_action = np.argmax(self.q_table[next_state])

        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q_table[next_state, next_action]

        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])

class ExpectedSARSA_Agent(Agent):
    def __init__(self, state_space, action_space, gamma=0.95, alpha=0.1, epsilon=0.05):
        super().__init__(state_space, action_space)
        self.q_table = np.random.uniform(low=0.01, high=0.02, size=(state_space, action_space))
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)  # Exploration
        return np.argmax(self.q_table[state])  # Exploitation

    def observe(self, state, action, reward, next_state, done):
        # Handle terminal state
        if done:
            expected_value = 0
        else:
            # Calculate the expected value of the next state
            best_next_action = np.argmax(self.q_table[next_state])
            expected_value = 0
            for next_action in range(self.action_space):
                if next_action == best_next_action:
                    prob = (1 - self.epsilon) + (self.epsilon / self.action_space)
                else:
                    prob = self.epsilon / self.action_space
                expected_value += prob * self.q_table[next_state, next_action]
        
        # Update Q-value using the Expected SARSA update rule
        td_target = reward + self.gamma * expected_value
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error



class QLearningAgent(Agent):
    def __init__(self, state_space, action_space, gamma=0.95, alpha=0.8, epsilon=0.05):
        super().__init__(state_space, action_space)
        self.q_table = np.random.uniform(low=0.1, high=0.2, size=(state_space, action_space))
        #self.q_table = np.zeros((state_space, action_space))
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

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

        # self.q_table[state, action] += self.alpha * (reward +  self.gamma * self.q_table[next_state, best_next_action] - self.q_table[state, action])

class Double_QLearningAgent(Agent):
    def __init__(self, state_space, action_space, gamma=0.95, alpha=0.1, epsilon=0.05, coin=0.5):
        super().__init__(state_space, action_space)
        self.q_table_1 = np.random.uniform(low=0.1, high=0.2, size=(state_space, action_space))
        self.q_table_2 = np.random.uniform(low=0.1, high=0.2, size=(state_space, action_space))
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.coin = coin

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)
        else:
            combined_q_tables = (self.q_table_1[state] + self.q_table_2[state]) / 2
            return np.argmax(combined_q_tables)

    def observe(self, state, action, reward, next_state, done):
        if np.random.rand() < self.coin:
            if done:
                target = reward
            else:
                target = reward + self.gamma * self.q_table_2[next_state][np.argmax(self.q_table_1[next_state])]
            self.q_table_1[state][action] += self.alpha * (target - self.q_table_1[state][action])
        else:
            if done:
                target = reward
            else:
                target = reward + self.gamma * self.q_table_1[next_state][np.argmax(self.q_table_2[next_state])]
            self.q_table_2[state][action] += self.alpha * (target - self.q_table_2[state][action])
