import numpy as np
import sys
from QLearning import QLearning

def main():
    # Define the reward matrix
    R = np.array([[-1, -1, -1, -1,  0, -1],
                  [-1, -1, -1,  0, -1, 100],
                  [-1, -1, -1,  0, -1, -1],
                  [-1,  0,  0, -1,  0, -1],
                  [ 0, -1, -1,  0, -1, 100],
                  [-1,  0, -1, -1,  0, 100]])

    print("Initial Reward Matrix (R):")
    print(R)

    # Initialize Q-learning with the reward matrix
    qlearn = QLearning(R)

    # Train the model for 20 episodes
    qlearn.train(20)

    # Print the final Q-table
    print("\nFinal Q Table:")
    print(qlearn.Q)

if __name__ == "__main__":
    sys.exit(int(main() or 0))

