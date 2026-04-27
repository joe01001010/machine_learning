from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, List

from networks import PolicyNetwork, ValueNetwork
from parking_lot import ACTIONS, ParkingLotEnvironment
from training_utils import (
    clipped,
    discounted_returns,
    normalize,
    selected_goal,
)


@dataclass
class ExperimentResult:
    agent: str
    episodes: int
    avg_reward: float
    avg_steps: float
    success_rate: float
    runtime_seconds: float


def run_reinforce(args: argparse.Namespace) -> ExperimentResult:
    env = ParkingLotEnvironment(
        max_steps=args.max_steps,
        seed=args.seed,
        distance_shaping=args.distance_shaping,
    )

    policy = PolicyNetwork(
        env.feature_size,
        args.hidden_size,
        len(ACTIONS),
        learning_rate=args.reinforce_policy_lr,
        seed=args.seed,
    )

    rewards: List[float] = []
    steps: List[int] = []
    successes: List[int] = []

    start_time = time.perf_counter()

    for _ in range(args.episodes):
        env.reset(selected_goal(args.goal))

        episode_features = []
        episode_actions = []
        episode_legal_actions = []
        episode_rewards = []

        done = False
        info = {"success": False}

        while not done:
            state_features = env.state_features()
            legal_actions = env.legal_actions()
            action = policy.sample_action(state_features, legal_actions)

            _, reward, done, info = env.step(action)

            episode_features.append(state_features)
            episode_actions.append(action)
            episode_legal_actions.append(legal_actions)
            episode_rewards.append(reward)

        returns = discounted_returns(episode_rewards, args.gamma)
        if args.normalize_returns:
            returns = normalize(returns)

        for state_features, action, legal_actions, return_value in zip(
            episode_features,
            episode_actions,
            episode_legal_actions,
            returns,
        ):
            policy.update_log_policy(
                state_features,
                action,
                clipped(return_value, args.advantage_clip),
                legal_actions,
                args.entropy_coef,
            )

        rewards.append(sum(episode_rewards))
        steps.append(env.step_count)
        successes.append(1 if info.get("success", False) else 0)

    runtime = time.perf_counter() - start_time

    return summarize_result(
        agent="REINFORCE",
        episodes=args.episodes,
        rewards=rewards,
        steps=steps,
        successes=successes,
        runtime_seconds=runtime,
        window=args.final_window,
    )


def run_reinforce_baseline(args: argparse.Namespace) -> ExperimentResult:
    env = ParkingLotEnvironment(
        max_steps=args.max_steps,
        seed=args.seed,
        distance_shaping=args.distance_shaping,
    )

    policy = PolicyNetwork(
        env.feature_size,
        args.hidden_size,
        len(ACTIONS),
        learning_rate=args.baseline_policy_lr,
        seed=args.seed,
    )

    value = ValueNetwork(
        env.feature_size,
        args.hidden_size,
        learning_rate=args.baseline_value_lr,
        seed=args.seed + 1,
    )

    rewards: List[float] = []
    steps: List[int] = []
    successes: List[int] = []

    start_time = time.perf_counter()

    for _ in range(args.episodes):
        env.reset(selected_goal(args.goal))

        episode_features = []
        episode_actions = []
        episode_legal_actions = []
        episode_rewards = []

        done = False
        info = {"success": False}

        while not done:
            state_features = env.state_features()
            legal_actions = env.legal_actions()
            action = policy.sample_action(state_features, legal_actions)

            _, reward, done, info = env.step(action)

            episode_features.append(state_features)
            episode_actions.append(action)
            episode_legal_actions.append(legal_actions)
            episode_rewards.append(reward)

        returns = discounted_returns(episode_rewards, args.gamma)

        for state_features, action, legal_actions, return_value in zip(
            episode_features,
            episode_actions,
            episode_legal_actions,
            returns,
        ):
            baseline = value.predict(state_features)
            advantage = return_value - baseline

            policy.update_log_policy(
                state_features,
                action,
                clipped(advantage, args.advantage_clip),
                legal_actions,
                args.entropy_coef,
            )

            value.update(state_features, return_value)

        rewards.append(sum(episode_rewards))
        steps.append(env.step_count)
        successes.append(1 if info.get("success", False) else 0)

    runtime = time.perf_counter() - start_time

    return summarize_result(
        agent="REINFORCE Baseline",
        episodes=args.episodes,
        rewards=rewards,
        steps=steps,
        successes=successes,
        runtime_seconds=runtime,
        window=args.final_window,
    )


def run_actor_critic(args: argparse.Namespace) -> ExperimentResult:
    env = ParkingLotEnvironment(
        max_steps=args.max_steps,
        seed=args.seed,
        distance_shaping=args.distance_shaping,
    )

    policy = PolicyNetwork(
        env.feature_size,
        args.hidden_size,
        len(ACTIONS),
        learning_rate=args.ac_policy_lr,
        seed=args.seed,
    )

    value = ValueNetwork(
        env.feature_size,
        args.hidden_size,
        learning_rate=args.ac_value_lr,
        seed=args.seed + 1,
    )

    rewards: List[float] = []
    steps: List[int] = []
    successes: List[int] = []

    start_time = time.perf_counter()

    for _ in range(args.episodes):
        env.reset(selected_goal(args.goal))

        total_reward = 0.0
        discount_multiplier = 1.0
        done = False
        info = {"success": False}

        while not done:
            state_features = env.state_features()
            legal_actions = env.legal_actions()
            action = policy.sample_action(state_features, legal_actions)

            _, reward, done, info = env.step(action)
            next_features = env.state_features()

            current_value = value.predict(state_features)
            next_value = 0.0 if done else value.predict(next_features)

            target = reward + args.gamma * next_value
            td_error = target - current_value

            policy.update_log_policy(
                state_features,
                action,
                clipped(discount_multiplier * td_error, args.advantage_clip),
                legal_actions,
                args.entropy_coef,
            )

            value.update(state_features, target)

            total_reward += reward
            discount_multiplier *= args.gamma

        rewards.append(total_reward)
        steps.append(env.step_count)
        successes.append(1 if info.get("success", False) else 0)

    runtime = time.perf_counter() - start_time

    return summarize_result(
        agent="One-Step Actor-Critic",
        episodes=args.episodes,
        rewards=rewards,
        steps=steps,
        successes=successes,
        runtime_seconds=runtime,
        window=args.final_window,
    )


def summarize_result(
    agent: str,
    episodes: int,
    rewards: List[float],
    steps: List[int],
    successes: List[int],
    runtime_seconds: float,
    window: int,
) -> ExperimentResult:
    reward_window = rewards[-window:]
    step_window = steps[-window:]
    success_window = successes[-window:]

    return ExperimentResult(
        agent=agent,
        episodes=episodes,
        avg_reward=sum(reward_window) / len(reward_window),
        avg_steps=sum(step_window) / len(step_window),
        success_rate=sum(success_window) / len(success_window),
        runtime_seconds=runtime_seconds,
    )


def print_plain_table(results: List[ExperimentResult]) -> None:
    print("\nPolicy Gradient Results")
    print("-" * 78)
    print(f"{'Agent':<24} {'Ep.':>8} {'Reward':>10} {'Steps':>10} {'Succ.':>10} {'Time (s)':>10}")
    print("-" * 78)

    for result in results:
        print(
            f"{result.agent:<24} "
            f"{result.episodes:>8} "
            f"{result.avg_reward:>10.2f} "
            f"{result.avg_steps:>10.2f} "
            f"{result.success_rate:>10.2f} "
            f"{result.runtime_seconds:>10.2f}"
        )

    print("-" * 78)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--goal", choices=["random", *ParkingLotEnvironment.parking_spots.keys()], default="random")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--advantage-clip", type=float, default=5.0)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--distance-shaping", type=float, default=0.02)
    parser.add_argument("--final-window", type=int, default=50)
    parser.add_argument("--env-label", type=str, default=r"7$\times$6")

    parser.add_argument("--normalize-returns", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--reinforce-policy-lr", type=float, default=0.008)

    parser.add_argument("--baseline-policy-lr", type=float, default=0.02)
    parser.add_argument("--baseline-value-lr", type=float, default=0.04)

    parser.add_argument("--ac-policy-lr", type=float, default=0.015)
    parser.add_argument("--ac-value-lr", type=float, default=0.04)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    runners: List[Callable[[argparse.Namespace], ExperimentResult]] = [
        run_reinforce,
        run_reinforce_baseline,
        run_actor_critic,
    ]

    results = [runner(args) for runner in runners]

    print_plain_table(results)


if __name__ == "__main__":
    main()
