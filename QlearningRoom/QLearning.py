import numpy as np

class QLearning(object):
    def __init__(self, R, goal_state=5):
        self.Q = np.zeros(R.shape)  # Initialize Q-table with zeros
        self.R = R  # Reward matrix
        self.num_states = R.shape[0]
        self.gamma = 0.8  # Discount factor
        self.goal_state = goal_state
        self.current_state = np.random.randint(0, self.num_states)  # Start at a random state

    def get_reward_from_environment(self, action):
        """Returns the reward for a given action from the R table."""
        return self.R[self.current_state, action]

    def train(self, num_training_episodes):
        """Trains the Q-learning model."""
        self.Q = np.zeros(self.R.shape)  # Reset Q-table
        for i in range(num_training_episodes):
            self.current_state = np.random.randint(0, self.num_states)  # Start from a random state
            valid_action_on_state = -1
            possible_action = -1
            reward = 0

            while True:
                # Choose a random valid action
                while valid_action_on_state == -1:
                    possible_action = np.random.randint(0, self.num_states)  # Pick a random next state
                    reward = self.get_reward_from_environment(possible_action)
                    valid_action_on_state = reward  # Ensure action is valid

                valid_action_on_state = -1

                # Q-table update formula: Q(s,a) = R(s,a) + gamma * max(Q(s',a'))
                next_state = possible_action
                qmax_next_state = self.get_QMax(next_state)
                self.Q[self.current_state, possible_action] = reward + (self.gamma * qmax_next_state)
                self.current_state = possible_action  # Move to next state

                if self.current_state == self.goal_state:
                    break  # Stop when goal state is reached

            print(f"Finished episode {i}, restarting environment")

    def get_QMax(self, next_state):
        """Finds the maximum Q value for the next state."""
        return max(self.Q[next_state, :])
