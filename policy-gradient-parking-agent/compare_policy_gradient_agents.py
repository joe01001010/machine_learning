from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, List

from networks import PolicyNetwork, ValueNetwork
from parking_lot import ACTIONS, ParkingLotEnvironment
from training_utils import (
    centered,
    clipped,
    discounted_returns,
    normalize,
    print_policy_demo,
    print_policy_snapshot,
    selected_goal,
)


@dataclass
class ExperimentResult:
    agent: str
    policy: PolicyNetwork
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
        policy=policy,
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
        baselines = [value.predict(state_features) for state_features in episode_features]
        advantages = [
            return_value - baseline
            for return_value, baseline in zip(returns, baselines)
        ]
        if args.center_advantages:
            advantages = centered(advantages)

        for state_features, action, legal_actions, advantage in zip(
            episode_features,
            episode_actions,
            episode_legal_actions,
            advantages,
        ):
            policy.update_log_policy(
                state_features,
                action,
                clipped(advantage, args.advantage_clip),
                legal_actions,
                args.entropy_coef,
            )

        for state_features, return_value in zip(episode_features, returns):
            value.update(state_features, return_value)

        rewards.append(sum(episode_rewards))
        steps.append(env.step_count)
        successes.append(1 if info.get("success", False) else 0)

    runtime = time.perf_counter() - start_time

    return summarize_result(
        agent="REINFORCE Baseline",
        policy=policy,
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
                clipped(td_error, args.advantage_clip),
                legal_actions,
                args.entropy_coef,
            )

            value.update(state_features, target)

            total_reward += reward

        rewards.append(total_reward)
        steps.append(env.step_count)
        successes.append(1 if info.get("success", False) else 0)

    runtime = time.perf_counter() - start_time

    return summarize_result(
        agent="One-Step Actor-Critic",
        policy=policy,
        episodes=args.episodes,
        rewards=rewards,
        steps=steps,
        successes=successes,
        runtime_seconds=runtime,
        window=args.final_window,
    )


def summarize_result(
    agent: str,
    policy: PolicyNetwork,
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
        policy=policy,
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


def print_agent_policies_and_demos(args: argparse.Namespace, results: List[ExperimentResult]) -> None:
    if args.no_policy_output and args.no_demo:
        return

    policy_goals = (
        list(ParkingLotEnvironment.parking_spots)
        if args.policy_goal == "all"
        else [args.policy_goal]
    )

    for result in results:
        env = ParkingLotEnvironment(
            max_steps=args.max_steps,
            seed=args.seed,
            distance_shaping=args.distance_shaping,
        )
        print(f"\n{'=' * 78}\n{result.agent}\n{'=' * 78}")

        if not args.no_policy_output:
            for goal_spot in policy_goals:
                print_policy_snapshot(
                    env,
                    result.policy,
                    goal_spot,
                    show_probabilities=not args.no_policy_probabilities,
                )

        if not args.no_demo:
            print_policy_demo(
                env,
                result.policy,
                args.demo_goal,
                max_steps=args.demo_steps,
            )


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
    parser.add_argument(
        "--policy-goal",
        choices=["all", *ParkingLotEnvironment.parking_spots.keys()],
        default="P8",
        help="Goal used when printing learned policy maps. Use 'all' for every parking spot.",
    )
    parser.add_argument(
        "--no-policy-output",
        action="store_true",
        help="Skip learned policy grid/probability output.",
    )
    parser.add_argument(
        "--no-policy-probabilities",
        action="store_true",
        help="Print only the greedy policy grid, without the probability table.",
    )
    parser.add_argument(
        "--demo-goal",
        choices=list(ParkingLotEnvironment.parking_spots.keys()),
        default="P8",
        help="Goal parking space used for each end-of-training rollout display.",
    )
    parser.add_argument(
        "--demo-steps",
        type=int,
        default=20,
        help="Maximum number of trained-agent steps to display for each rollout.",
    )
    parser.add_argument("--no-demo", action="store_true", help="Skip rollout demos.")

    parser.add_argument("--normalize-returns", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--reinforce-policy-lr", type=float, default=0.008)

    parser.add_argument("--baseline-policy-lr", type=float, default=0.06)
    parser.add_argument("--baseline-value-lr", type=float, default=0.01)
    parser.add_argument("--center-advantages", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--ac-policy-lr", type=float, default=0.08)
    parser.add_argument("--ac-value-lr", type=float, default=0.02)

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
    print_agent_policies_and_demos(args, results)


if __name__ == "__main__":
    main()
