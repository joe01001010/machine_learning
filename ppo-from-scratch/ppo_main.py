#!/usr/bin/env python


import gymnasium as gym
import numpy as np
from ppo_classes import Agent


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n,
        batch_size=batch_size,
        alpha=alpha,
        n_epochs=n_epochs,
        input_dims=env.observation_space.shape
    )
    n_games = 300
    figure_file = 'plots/cartpole.png'
    best_score = -np.inf
    score_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)

            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1

            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(f"Episode: {i+1},  Score: {score:.2f}, Avg Score: {avg_score:.2f}, Steps: {n_steps}, Learn Steps: {learn_iters}")
