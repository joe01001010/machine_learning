"""One-step actor-critic for the assignment parking-lot environment.

The policy network is the actor, and the value network is the critic. Unlike
both REINFORCE implementations, this does not wait for a full episode return.
After each step, it computes the one-step TD error

    delta = reward + gamma * V(next_state) - V(current_state)

and uses that TD error as the actor's advantage signal.
"""

from __future__ import annotations

import argparse

from networks import PolicyNetwork, ValueNetwork
from parking_lot import ACTIONS, ParkingLotEnvironment
from training_utils import (
    add_common_training_args,
    clipped,
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
        total_reward = 0.0
        done = False
        info = {"success": False}

        while not done:
            state_features = env.state_features()
            legal_actions = env.legal_actions()
            action = policy.sample_action(state_features, legal_actions)
            _, reward, done, info = env.step(action)
            next_features = env.state_features()

            # Actor-critic updates immediately after each transition. The
            # critic forms a one-step TD target, and the actor uses the TD
            # error as an advantage estimate for the action just taken.
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
        success_window.append(1 if info.get("success", False) else 0)
        if not args.quiet and episode % args.eval_every == 0:
            print_progress(episode, rewards, steps, success_window)

    return policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_training_args(parser)
    parser.add_argument("--policy-lr", type=float, default=0.08)
    parser.add_argument("--value-lr", type=float, default=0.02)
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
