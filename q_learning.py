import numpy as np

class QLearning:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob

        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_prob:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
       
        best_next_action = np.argmax(self.q_table[next_state, :])
        self.q_table[state, action] = (1 - self.learning_rate) * self.q_table[state, action] + \
                                      self.learning_rate * (reward + self.discount_factor * self.q_table[next_state, best_next_action])

def run_q_learning():
    num_states = 6
    num_actions = 2
    agent = QLearning(num_states, num_actions)
    num_episodes = 10

    for episode in range(num_episodes):
        state = 0
        total_reward = 0

        while state != num_states - 1:
            action = agent.choose_action(state)
            next_state = state + action
            reward = 0 if next_state != num_states - 1 else 1  
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

if __name__ == "__main__":
    run_q_learning()
