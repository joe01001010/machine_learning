#!/usr/bin/env python

import argparse
import time
import sys

from parking_lot import ParkingLot
from q_table_agent import QTableAgent
from double_q_table_agent import DoubleQTableAgent
from deep_q_network import DeepQNetwork
from tensorboard_logger import TensorBoardLogger
from logging_init import configure_logging
LOGGER = configure_logging(__file__)


def get_agent_state(parking_lot, agent):
    """
    This function takes two arguments
    parking_lot is the parking lot object that is the environment for the agent
    agent is the agent that is currently training
    This function will figure out if it is a tabular or network based agent
    This function will return the appropriate state based on the architecture of the agent
    """
    state_method = getattr(agent, "state_method", "get_state")
    LOGGER.debug(f"Returning agent state {getattr(parking_lot, state_method)()} from {sys._getframe().f_code.co_name}")
    return getattr(parking_lot, state_method)()


def train_agent(parking_lot, agent, tb_logger, goal_index, episodes=2000, max_steps=100, cutoff=50):
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
    agent_name = agent.__class__.__name__

    LOGGER.info(
        "Training started | agent=%s goal=%s episodes=%s max_steps=%s cutoff=%s",
        agent_name,
        parking_lot.goal,
        episodes,
        max_steps,
        cutoff,
    )

    for episode in range(episodes):
        total_reward = 0
        steps_taken = 0
        succeeded = 0
        losses = []

        parking_lot.reset()
        state = get_agent_state(parking_lot, agent)

        for step in range(max_steps):
            current_state = state
            action = agent.choose_action(current_state)
            _, reward, done = parking_lot.move_agent(action)
            next_state = get_agent_state(parking_lot, agent)

            loss = agent.learn(current_state, action, reward, next_state, done)
            if loss is not None:
                losses.append(loss)

            LOGGER.debug(
                "Training step | agent=%s episode=%s step=%s state=%s action=%s reward=%s next_state=%s loss=%s",
                agent_name,
                episode + 1,
                step + 1,
                current_state,
                action,
                reward,
                next_state,
                loss if loss is not None else "n/a",
            )
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

        window = 50
        recent_rewards = agent_history['episode_rewards'][-window:]
        recent_steps = agent_history['episode_steps'][-window:]
        recent_successes = agent_history['success'][-window:]

        avg_reward_50 = sum(recent_rewards) / len(recent_rewards)
        avg_steps_50 = sum(recent_steps) / len(recent_steps)
        avg_success_50 = sum(recent_successes) / len(recent_successes)

        tb_logger.log_episode(
            agent_name=agent_name,
            goal_index=goal_index,
            episode=episode,
            reward=total_reward,
            steps=steps_taken,
            success=succeeded,
            epsilon=agent.epsilon,
            loss=(sum(losses) / len(losses) if losses else None),
            avg_reward_50=avg_reward_50,
            avg_steps_50=avg_steps_50,
            avg_success_50=avg_success_50,
        )

        if (episode + 1) % 100 == 0:
            window = min(100, len(agent_history['episode_rewards']))
            recent_losses = [loss for loss in agent_history['loss'][-window:] if loss is not None]
            avg_loss = (
                f"{sum(recent_losses) / len(recent_losses):.4f}"
                if recent_losses else "n/a"
            )
            LOGGER.info(
                "Training progress | agent=%s goal=%s episode=%s avg_reward=%.2f successes_last_%s=%s epsilon=%.4f avg_loss=%s",
                agent_name,
                parking_lot.goal,
                episode + 1,
                sum(agent_history['episode_rewards'][-window:]) / window,
                window,
                sum(agent_history['success'][-window:]),
                agent.epsilon,
                avg_loss,
            )

        if len(agent_history['success']) >= cutoff:
            if sum(agent_history['success'][-cutoff:]) == cutoff:
                if max(agent_history['episode_steps'][-cutoff:]) - min(agent_history['episode_steps'][-cutoff:]) <= 2:
                    LOGGER.info(
                        "Training stopped early | agent=%s goal=%s episode=%s recent_successes=%s",
                        agent_name,
                        parking_lot.goal,
                        episode + 1,
                        cutoff,
                    )
                    break

    LOGGER.info(
        "Training finished | agent=%s goal=%s episodes_run=%s successes=%s final_epsilon=%.4f",
        agent_name,
        parking_lot.goal,
        len(agent_history['success']),
        sum(agent_history['success']),
        agent.epsilon,
    )
    return agent_history


def test_agent(parking_lot, agent, display=False,  max_steps=50):
    """
    This function takes four arguments
    parking_lot is the parking lot object that represents the environment for the agent to explore
    agent is the agent architecture that is being tested
    display is a boolean that will tell whether to display the testing to stdout
    max_steps is how many steps the agent is allowed to take in testing, default=50
    This function will reset the parking lot and use the trained agents to show the converged upon policy
    """
    parking_lot.reset()
    state = get_agent_state(parking_lot, agent)
    path = [parking_lot.get_state()]
    agent_name = agent.__class__.__name__
    done = parking_lot.parked
    reward = 0

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    LOGGER.info(
        "Testing started | agent=%s goal=%s max_steps=%s display=%s",
        agent_name,
        parking_lot.goal,
        max_steps,
        display,
    )

    for step in range(max_steps):
        if display:
            print(parking_lot)
            LOGGER.info(
                "Display snapshot | agent=%s goal=%s\n%s",
                agent_name,
                parking_lot.goal,
                parking_lot,
            )
            time.sleep(1)

        current_state = state
        action = agent.choose_action(current_state)
        next_state, reward, done = parking_lot.move_agent(action)
        path.append(next_state)
        state = get_agent_state(parking_lot, agent)
        LOGGER.debug(
            "Testing step | agent=%s step=%s/%s state=%s action=%s reward=%s next_state=%s",
            agent_name,
            step + 1,
            max_steps,
            current_state,
            action,
            reward,
            next_state,
        )

        if done:
            if display:
                print(parking_lot)
                LOGGER.info(
                    "Display snapshot | agent=%s goal=%s\n%s",
                    agent_name,
                    parking_lot.goal,
                    parking_lot,
                )
            break

    agent.epsilon = old_epsilon
    LOGGER.info(
        "Testing finished | agent=%s goal=%s success=%s steps=%s final_reward=%s path=%s",
        agent_name,
        parking_lot.goal,
        done,
        len(path) - 1,
        reward,
        path,
    )
    return path, done


def main(rows, columns, display):
    LOGGER.info(f"Run started | rows={rows} columns={columns} display={display}")
    temp_lot = ParkingLot(rows, columns)
    tb_logger = TensorBoardLogger()
    q_table_agents = dict()
    q_table_agents_stats = dict()
    double_q_table_agents = dict()
    double_q_table_agents_stats = dict()
    deep_q_network_agents = dict()
    deep_q_network_agents_stats = dict()

    for goal in range(len(temp_lot.parking_spots)):
        LOGGER.info(f"Goal processing started | goal_index={goal} target={temp_lot.parking_spots[goal]}")

        q_table_agent = QTableAgent()
        q_table_agents_stats[goal] = train_agent(ParkingLot(rows, columns, goal), q_table_agent, tb_logger, goal_index=goal)
        q_table_agents[goal] = q_table_agent

        double_q_table_agent = DoubleQTableAgent()
        double_q_table_agents_stats[goal] = train_agent(ParkingLot(rows, columns, goal), double_q_table_agent, tb_logger, goal_index=goal)
        double_q_table_agents[goal] = double_q_table_agent

        deep_q_network_agent = DeepQNetwork()
        deep_q_network_agents_stats[goal] = train_agent(ParkingLot(rows, columns, goal), deep_q_network_agent, tb_logger, goal_index=goal)
        deep_q_network_agents[goal] = deep_q_network_agent

    for goal in range(len(temp_lot.parking_spots)):
        parking_lot_test = ParkingLot(rows, columns, goal)

        for agent_name, agent_dict in [
            ("QTableAgent", q_table_agents),
            ("DoubleQTableAgent", double_q_table_agents),
            ("DeepQNetwork", deep_q_network_agents),
        ]:
            agent_test = agent_dict[goal]
            path, success = test_agent(parking_lot_test, agent_test, display=display)

            tb_logger.log_test_result(
                agent_name=agent_name,
                goal_index=goal,
                success=success,
                steps=len(path) - 1,
                path=path,
            )
    
    tb_logger.close()
    LOGGER.info(f"Run finished | goals_trained={len(temp_lot.parking_spots)}")


if __name__ == '__main__':
    start = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default=7, type=int)
    parser.add_argument("--columns", default=6, type=int)
    parser.add_argument("--display", default=False, type=bool)
    args = parser.parse_args()

    LOGGER.info(f"Calling main | rows={args.rows} columns={args.columns} display={args.display}")
    main(args.rows, args.columns, args.display)

    end = time.perf_counter()
    seconds = end - start
    metric = f"{'Execution Time (Seconds):' if seconds <= 60 else 'Execution Time (Minutes)'}"
    seconds = seconds if seconds <= 60 else seconds / 60
    LOGGER.info(f"{metric} {seconds:.2f}")