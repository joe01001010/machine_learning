"""REINFORCE with a learned state-value baseline.

This is still a Monte Carlo policy-gradient method: it waits for the full
episode, computes discounted returns G_t, and then updates the policy. The
difference from plain REINFORCE is the learned value network V(s). The policy
uses G_t - V(s) as its advantage estimate, so actions are reinforced only when
the episode return was better than the value network expected.
"""

from __future__ import annotations

import argparse

from networks import PolicyNetwork, ValueNetwork
from parking_lot import ACTIONS, ParkingLotEnvironment
from training_utils import (
    add_common_training_args,
    centered,
    clipped,
    discounted_returns,
    new_success_window,
    print_evaluation,
    print_policy_demo,
    print_policy_snapshot,
    print_progress,
    selected_goal,
)


def train(args: argparse.Namespace) -> PolicyNetwork:
    env = ParkingLotEnvironment(
        max_steps=args.max_steps,
        seed=args.seed,
        distance_shaping=args.distance_shaping,
    )
    policy = PolicyNetwork(
        env.feature_size,
        args.hidden_size,
        len(ACTIONS),
        learning_rate=args.policy_lr,
        seed=args.seed,
    )
    value = ValueNetwork(
        env.feature_size,
        args.hidden_size,
        learning_rate=args.value_lr,
        seed=args.seed + 1,
    )

    rewards = []
    steps = []
    success_window = new_success_window()

    for episode in range(1, args.episodes + 1):
        env.reset(selected_goal(args.goal))
        features = []
        actions = []
        legal_action_sets = []
        episode_rewards = []
        done = False
        info = {"success": False}

        while not done:
            state_features = env.state_features()
            legal_actions = env.legal_actions()
            action = policy.sample_action(state_features, legal_actions)
            _, reward, done, info = env.step(action)

            features.append(state_features)
            actions.append(action)
            legal_action_sets.append(legal_actions)
            episode_rewards.append(reward)

        # Like plain REINFORCE, this uses full-episode returns. The value
        # predictions are taken before any critic update for this episode, so
        # the policy sees one consistent learned baseline for the trajectory.
        returns = discounted_returns(episode_rewards, args.gamma)
        baselines = [value.predict(state_features) for state_features in features]
        advantages = [
            return_value - baseline
            for return_value, baseline in zip(returns, baselines)
        ]
        if args.center_advantages:
            advantages = centered(advantages)

        for state_features, action, legal_actions, advantage in zip(
            features,
            actions,
            legal_action_sets,
            advantages,
        ):
            policy.update_log_policy(
                state_features,
                action,
                clipped(advantage, args.advantage_clip),
                legal_actions,
                args.entropy_coef,
            )

        for state_features, return_value in zip(features, returns):
            value.update(state_features, return_value)

        rewards.append(sum(episode_rewards))
        steps.append(env.step_count)
        success_window.append(1 if info.get("success", False) else 0)
        if not args.quiet and episode % args.eval_every == 0:
            print_progress(episode, rewards, steps, success_window)

    return policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_training_args(parser)
    parser.add_argument("--policy-lr", type=float, default=0.06)
    parser.add_argument("--value-lr", type=float, default=0.01)
    parser.add_argument(
        "--center-advantages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Subtract the episode mean from G_t - V(s_t) before the policy update.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    policy = train(args)
    env = ParkingLotEnvironment(
        max_steps=args.max_steps,
        seed=args.seed,
        distance_shaping=args.distance_shaping,
    )
    print_evaluation(env, policy)
    if not args.no_policy_output:
        policy_goals = (
            list(ParkingLotEnvironment.parking_spots)
            if args.policy_goal == "all"
            else [args.policy_goal]
        )
        for goal_spot in policy_goals:
            print_policy_snapshot(
                env,
                policy,
                goal_spot,
                show_probabilities=not args.no_policy_probabilities,
            )
    if not args.no_demo:
        print_policy_demo(env, policy, args.demo_goal, max_steps=args.demo_steps)


if __name__ == "__main__":
    main()
