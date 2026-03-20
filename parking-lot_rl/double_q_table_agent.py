import random

class DoubleQTableAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q1_table = {}
        self.q2_table = {}
        self.state_method = "get_state"

    def get_q_values(self, table, state):
        """
        This function takes two arguments
        table is the current Q table
        state is the current state of the agent
        This function will initialize the q value in the table to all zeros then return it
        or it will return the current q values for that state
        """
        if state not in table:
            table[state] = [0.0, 0.0, 0.0, 0.0]
        return table[state]

    def choose_action(self, state):
        """
        This function takes one argument
        state is the current state of the agent
        This function will first check if the epsilon is higher than a random number
        If epsilon is higher than random it will choose a random action
        This function will get the q value for q1 and q2
        Then this function will combine the corresponding values from the two different tables
        The agent will pull the highest value action or if there is a tie randomly choose one
        """
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2, 3])

        q1 = self.get_q_values(self.q1_table, state)
        q2 = self.get_q_values(self.q2_table, state)
        combined = [q1[i] + q2[i] for i in range(4)]

        max_q = max(combined)
        best_actions = [i for i, q in enumerate(combined) if q == max_q]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        """
        This function takes 5 arguments
        state is the current state of the agent
        action is the action that was selected in that state
        rewards is the reward given for selecting that action in that state
        next_state is the state the agent will be in after performing the action in that original state
        done is a boolean value to say if the agent is terminating the episode
        This function will use a 50% chance to either update Q2 or Q1
        If updating Q1 it will get the values for the state in Q1's Q table
        Then the Q values for the next state from both Q1 and Q2's tables
        If in the terminal state the target will be set to the reward
        Else the agent will find the next best action to perform in the next state
        Then it will set the target to a discounted reward for a specific action in that next state
        vice versa for if Q2 is selected to be updated
        """
        if random.random() < 0.5:
            q1 = self.get_q_values(self.q1_table, state)
            q1_next = self.get_q_values(self.q1_table, next_state)
            q2_next = self.get_q_values(self.q2_table, next_state)

            if done:
                target = reward
            else:
                best_next_action = max(range(4), key=lambda a: q1_next[a])
                target = reward + self.gamma * q2_next[best_next_action]

            q1[action] += self.alpha * (target - q1[action])

        else:
            q2 = self.get_q_values(self.q2_table, state)
            q1_next = self.get_q_values(self.q1_table, next_state)
            q2_next = self.get_q_values(self.q2_table, next_state)

            if done:
                target = reward
            else:
                best_next_action = max(range(4), key=lambda a: q2_next[a])
                target = reward + self.gamma * q1_next[best_next_action]

            q2[action] += self.alpha * (target - q2[action])

    def decay_epsilon(self):
        """
        This function takes no arguments
        This function will update the objects epsilon value the either the min value
        or it will decay the epsilon value by the decay value multiplied by the current epsilon
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)