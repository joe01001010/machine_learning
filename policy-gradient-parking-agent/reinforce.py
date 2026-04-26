"""
This is the Monte Carlo policy-gradient version. It runs a full episode first,
computes the discounted return G_t for each visited state-action pair, and then
updates only the policy network. There is no value network here, so the raw
episode return is the learning signal. That makes the algorithm simple, but
also higher variance than REINFORCE with a baseline or actor-critic.
"""

from __future__ import annotations

import argparse

from networks import PolicyNetwork
from parking_lot import ACTIONS, ParkingLotEnvironment
from training_utils import (
    add_common_training_args,
    clipped,
    discounted_returns,
    normalize,
    new_success_window,
    print_evaluation,
    print_policy_demo,
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

        # Plain REINFORCE waits until the episode is over, then uses the full
        # discounted return G_t to decide how strongly to reinforce each action.
        returns = discounted_returns(episode_rewards, args.gamma)
        policy_returns = normalize(returns) if args.normalize_returns else returns
        for state_features, action, legal_actions, return_value in zip(
            features,
            actions,
            legal_action_sets,
            policy_returns,
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
        success_window.append(1 if info.get("success", False) else 0)
        if not args.quiet and episode % args.eval_every == 0:
            print_progress(episode, rewards, steps, success_window)

    return policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_training_args(parser)
    parser.add_argument("--policy-lr", type=float, default=0.008)
    parser.add_argument(
        "--normalize-returns",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize episode returns before the policy update.",
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
    if not args.no_demo:
        print_policy_demo(env, policy, args.demo_goal, max_steps=args.demo_steps)


if __name__ == "__main__":
    main()
