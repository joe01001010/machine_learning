#!/usr/bin/env python


import argparse
import time

from parking_lot import ParkingLot
from q_table_agent import QTableAgent
from double_q_table_agent import DoubleQTableAgent
from deep_q_network import DeepQNetwork


def get_agent_state(parking_lot, agent):
    """
    This function takes two arguments
    parking_lot is the parking lot object that is the environment for the agent
    agent is the agent that is currently training
    This function will figure out if it is a tabular or network based agent
    This function will return the appropriate state based on the architecture of the agent
    """
    state_method = getattr(agent, "state_method", "get_state")
    return getattr(parking_lot, state_method)()


def train_agent(parking_lot, agent, episodes=2000, max_steps=100, cutoff=50):
    """
    This function takes 5 arguments
    parking_lot is the object that represent the environment the agent is training in
    agent is the object that represents either the q table, double q table, or q network agent
    episodes is the max number of episodes to execute, default is 2000
    max_steps is the maximum amount of steps to perform before continuing to the next episode, default is 100
    cutoff is the number of episodes that need to be won in a row before exiting
    This cutoff also checks to see if the agent is improving and if its not it will exit
    """
    agent_history = {
        'episode_rewards': [],
        'episode_steps': [],
        'success': [],
        'epsilon': [],
        'loss': [],
    }

    for episode in range(episodes):
        total_reward = 0
        steps_taken = 0
        succeeded = 0
        losses = []

        parking_lot.reset()
        state = get_agent_state(parking_lot, agent)

        for step in range(max_steps):
            action = agent.choose_action(state)
            _, reward, done = parking_lot.move_agent(action)
            next_state = get_agent_state(parking_lot, agent)

            loss = agent.learn(state, action, reward, next_state, done)
            if loss is not None:
                losses.append(loss)

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
        agent_history['loss'].append(sum(losses) / len(losses) if losses else None)
        agent.decay_epsilon()

        if len(agent_history['success']) >= cutoff:
            if sum(agent_history['success'][-cutoff:]) == cutoff:
                if max(agent_history['episode_steps'][-cutoff:]) - min(agent_history['episode_steps'][-cutoff:]) <= 2:
                    break

    return agent_history


def test_agent(parking_lot, agent, display=False,  max_steps=50):
    state = parking_lot.reset()
    path = [state]

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(max_steps):
        if display:
            print(parking_lot)
            time.sleep(1)

        action = agent.choose_action(state)
        next_state, reward, done = parking_lot.move_agent(action)
        path.append(next_state)
        state = next_state

        if done:
            if display: print(parking_lot)
            break

    agent.epsilon = old_epsilon
    return path, done


def main(rows, columns, display):
    temp_lot = ParkingLot(rows, columns)
    q_table_agents = dict()
    q_table_agents_stats = dict()
    q_table_agents_tests = dict()
    double_q_table_agents = dict()
    double_q_table_agents_stats = dict()
    double_q_table_agents_tests = dict()
    deep_q_network_agents = dict()
    deep_q_network_agents_stats = dict()
    deep_q_network_tests = dict()

    for goal in range(len(temp_lot.parking_spots)):
        q_table_agent = QTableAgent()
        q_table_agents_stats[goal] = train_agent(ParkingLot(rows, columns, goal), q_table_agent)
        q_table_agents_tests[goal] = test_agent(ParkingLot(rows, columns, goal), q_table_agent, display=display)
        q_table_agents[goal] = q_table_agent

        double_q_table_agent = DoubleQTableAgent()
        double_q_table_agents_stats[goal] = train_agent(ParkingLot(rows, columns, goal), double_q_table_agent)
        double_q_table_agents_tests[goal] = test_agent(ParkingLot(rows, columns, goal), double_q_table_agent, display=display)
        double_q_table_agents[goal] = double_q_table_agent

        deep_q_network_agent = DeepQNetwork()
        deep_q_network_agents_stats[goal] = train_agent(ParkingLot(rows, columns, goal), deep_q_network_agent)
        deep_q_network_tests[goal] = test_agent(ParkingLot(rows, columns, goal), deep_q_network_agent, display=display)
        deep_q_network_agents[goal] = deep_q_network_agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default=7, type=int)
    parser.add_argument("--columns", default=6, type=int)
    parser.add_argument("--display", default=False, type=bool)
    args = parser.parse_args()
    main(args.rows, args.columns, args.display)