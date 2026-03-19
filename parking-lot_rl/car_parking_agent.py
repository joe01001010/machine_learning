#!/usr/bin/env python


import argparse
import random
import time
from turtle import window_width


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}

    def get_q_values(self, state):
        """
        This function takes the state as an argument
        If the state is not in the q table it will return all 0
        Else it will return the Q_table that corresponds to that state
        """
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0, 0.0]
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
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
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

    def decay_epsilon(self):
        """
        This function will update the epsilon value of the object
        This function will decay until it hits the minimum
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class ParkingLot:
    def __init__(self, rows, columns, goal):
        self.rows = rows
        self.columns = columns
        self.entrance = (rows - 1, columns - 1)
        self.barriers = [(rows // 2, column) for column in range(1, columns - 1)]
        self.parking_spots = [(barrier[0] + offset, barrier[1]) for barrier in self.barriers for offset in (-1, 1)]
        self.goal = self.parking_spots[goal]
        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.reset()

    def reset(self):
        """
        This function takes no arguments
        This function will update the agent with a fresh start
        This function will return the current position of the agent
        """
        self.agent_location = self.entrance
        self.parked = self.agent_location == self.goal
        return self.get_state()

    def get_state(self):
        """
        This function takes no arguments
        This function will return the location of the agent
        """
        return self.agent_location

    def move_agent(self, action):
        """
        This function takes one argument
        action is the ID 0, 1, 2, 3 of either up, down, left, right
        This function will check to see if the action is out of bounds
        Check if the agent moves into a barrier
        Check if the agent enters a parking space, and if its the correct parking space
        This function will update the agent's location and return a reward as an int if a valid move is made
        """
        curr_row, curr_col = self.get_state()
        row_mod, col_mod = self.actions[action]
        new_row = curr_row + row_mod
        new_col = curr_col + col_mod
        next_position = (new_row, new_col)
        if new_row < 0 or new_row >= self.rows or new_col < 0 or new_col >= self.columns:
            return self.get_state(), -1, False
        elif next_position in self.barriers:
            return self.get_state(), -1, False
        elif next_position in self.parking_spots and next_position != self.goal:
            return self.get_state(), -1, False
        elif next_position == self.goal:
            self.agent_location = next_position
            self.parked = True
            return self.get_state(), 100, True
        else:
            self.agent_location = next_position
            return self.get_state(), -1, False

    def __str__(self):
        """
        This function takes no arguments
        This function will print the current representation of the parking lot
        This function will return a string that represents the parking lot and all entities inside
        """
        parking_lot_status = ""
        for row in range(self.rows):
            for column in range(self.columns):
                if (row, column) == self.agent_location:
                    parking_lot_status += "[A]"
                elif (row, column) in self.parking_spots:
                    parking_lot_status += "[P]"
                elif (row, column) in self.barriers:
                    parking_lot_status += "[X]"
                else:
                    parking_lot_status += "[ ]"
            parking_lot_status += "\n"

        return parking_lot_status

    def __repr__(self):
        return f"Rows: {self.rows}, Columns: {self.columns}, Goal: {self.goal}, Entrance: {self.entrance}"


def train_agent(parking_lot, agent, episodes=2000, max_steps=100, cutoff=50):
    agent_history = {
        'episode_rewards': [],
        'episode_steps': [],
        'success': [],
        'epsilon': [],
    }

    for episode in range(episodes):
        total_reward = 0
        steps_taken = 0
        succeeded = 0
        state = parking_lot.reset()

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = parking_lot.move_agent(action)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps_taken += 1

            if done:
                succeeded = 1
                break

        agent_history['episode_rewards'].append(total_reward)
        agent_history['episode_steps'].append(steps_taken)
        agent_history['success'].append(succeeded)
        agent_history['epsilon'].append(agent.epsilon)
        agent.decay_epsilon()

        if sum(agent_history['success'][-cutoff:]) == cutoff:
            if max(agent_history['episode_steps'][-cutoff:]) - min(agent_history['episode_steps'][-cutoff:]) <= 2:
                break

    print(f"Training successes: {sum(agent_history['success'])}/{len(agent_history['success'])}")
    return agent_history


def test_agent(parking_lot, agent, max_steps=50):
    state = parking_lot.reset()
    path = [state]

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(max_steps):
        print(parking_lot)
        action = agent.choose_action(state)
        next_state, reward, done = parking_lot.move_agent(action)
        path.append(next_state)
        state = next_state
        time.sleep(1)

        if done:
            print(parking_lot)
            break

    agent.epsilon = old_epsilon
    return path, done


def print_training_summary(history, window=100):
    """
    This function takes two arguments
    history is the history of the training pipeline the agent took
    windows it the subset to average over the history
    This function will print the training summary to stdout
    This function doesnt return anything
    """
    rewards = history["episode_rewards"]
    steps = history["episode_steps"]
    successes = history["success"]
    epsilons = history["epsilon"]

    for offset in range(0, len(rewards), window):
        section = window
        if len(rewards) - (offset + window) < window:
            section = len(rewards) - (offset + window)

        average_reward = sum(rewards[offset:offset + section]) / section
        average_steps = sum(steps[offset:offset + section]) / section
        success_rate = sum(successes[offset:offset + section]) / section
        average_epsilon = sum(epsilons[offset:offset + section]) / section

        print(f"Episodes {offset}-{offset + section}: ", end = '')
        print(f"Average Reward={average_reward:.2f}, ", end = '')
        print(f"Average Steps={average_steps:.2f}, ", end = '')
        print(f"Success Rate={success_rate:.2%}, ", end = '')
        print(f"Average Epsilon={average_epsilon:.3f}")
        
        if section != window:
            break


def main(rows, columns):
    for goal in range((columns - 2) * 2):
        parking_lot = ParkingLot(rows, columns, goal)
        agent = QLearningAgent()

        episode_stats = train_agent(parking_lot, agent)
        print_training_summary(episode_stats)

        path, success = test_agent(parking_lot, agent)
        print("Success:", success)
        print(f"Number of steps for optimal path found: {len(path) - 1}")
        print("Path:", path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default=7, type=int)
    parser.add_argument("--columns", default=6, type=int)
    args = parser.parse_args()
    main(args.rows, args.columns)