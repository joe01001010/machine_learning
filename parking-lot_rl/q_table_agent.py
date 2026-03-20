import random
import sys

from logging_init import configure_logging
LOGGER = configure_logging(__file__)

class QTableAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}
        self.state_method = "get_state"
        LOGGER.info(
            f"Initializing constructor for {self.__class__.__name__}, "
            f"Alpha: {self.alpha}, "
            f"Gamma: {self.gamma}, "
            f"Epsilon: {self.epsilon}, "
            f"Epsilon decay: {self.epsilon_decay}, "
            f"Epsilon Min: {self.epsilon_min}, "
            f"State Method: {self.state_method}"
        )

    def get_q_values(self, state):
        """
        This function takes the state as an argument
        If the state is not in the q table it will return all 0
        Else it will return the Q_table that corresponds to that state
        """
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0, 0.0]
        LOGGER.debug(f"Returning Q values: {self.q_table} from {sys._getframe().f_code.co_name}")
        return self.q_table[state]

    def choose_action(self, state):
        """
        This function takes state as an argument
        This function will check a random value against epsilon to see if the action should be random or not
        This function will choose an action if the random action isnt taken based on the Q table
        This function will randomly choose one if the best actions are tied
        """
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2, 3])
        q_values = self.get_q_values(state)
        max_q = max(q_values)
        best_actions = [i for i, q in enumerate(q_values) if q == max_q]
        LOGGER.debug(f"Returning action: {best_actions} from {sys._getframe().f_code.co_name}")
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        """
        This function takes 5 arguments
        state is the current position of the agent
        action is the action chosen by the agent from that state
        reward is the reward received for choosing the action in the specific state
        next_state is the state the agent will go to after taking that action
        done is a boolean to say if the agent is in the terminal state
        This function will update the q table within the QLearningAgent object
        """
        current_q = self.get_q_values(state)[action]
        max_next_q = 0 if done else max(self.get_q_values(next_state))

        new_q = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )

        self.q_table[state][action] = new_q
        LOGGER.debug(f"Learning: {self.q_table[state][action]} <- {new_q} from {sys._getframe().f_code.co_name}")

    def decay_epsilon(self):
        """
        This function will update the epsilon value of the object
        This function will decay until it hits the minimum
        """
        LOGGER.debug(f"Decaying epsilon: max({self.epsilon_min}, {self.epsilon} * {self.epsilon_decay}) from {sys._getframe().f_code.co_name}")
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)