"""Shared command-line and evaluation helpers for the parking agents."""

from __future__ import annotations

import argparse
from collections import deque
from typing import Deque, Dict, Iterable, List, Sequence, Tuple

from networks import PolicyNetwork, clip
from parking_lot import ACTIONS, ParkingLotEnvironment, ParkingState


ACTION_SYMBOLS = {
    "forward": "F",
    "backward": "B",
    "left": "L",
    "right": "R",
}


def add_common_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--goal", choices=["random", *ParkingLotEnvironment.parking_spots.keys()], default="random")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--advantage-clip", type=float, default=5.0)
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="Entropy bonus that keeps the policy exploratory during training.",
    )
    parser.add_argument(
        "--distance-shaping",
        type=float,
        default=0.02,
        help="Reward added for reducing Manhattan distance to the goal. Use 0 for sparse rewards.",
    )
    parser.add_argument(
        "--demo-goal",
        choices=list(ParkingLotEnvironment.parking_spots.keys()),
        default="P8",
        help="Goal parking space used for the end-of-training rollout display.",
    )
    parser.add_argument(
        "--demo-steps",
        type=int,
        default=20,
        help="Maximum number of trained-agent steps to display after training.",
    )
    parser.add_argument(
        "--policy-goal",
        choices=["all", *ParkingLotEnvironment.parking_spots.keys()],
        default="P8",
        help="Goal used when printing the learned policy. Use 'all' for every parking spot.",
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
    parser.add_argument("--no-demo", action="store_true", help="Skip the end-of-training rollout display.")
    parser.add_argument("--quiet", action="store_true")


def selected_goal(goal_argument: str) -> str | None:
    return None if goal_argument == "random" else goal_argument


def discounted_returns(rewards: Sequence[float], gamma: float) -> List[float]:
    returns = [0.0 for _ in rewards]
    running_return = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running_return = rewards[index] + gamma * running_return
        returns[index] = running_return
    return returns


def normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = variance ** 0.5
    if std < 1e-8:
        return [value - mean for value in values]
    return [(value - mean) / std for value in values]


def centered(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    return [value - mean for value in values]


def clipped(value: float, limit: float | None) -> float:
    return clip(value, limit)


def new_success_window(size: int = 100) -> Deque[int]:
    return deque(maxlen=size)


def print_progress(episode: int, rewards: Sequence[float], steps: Sequence[int], successes: Iterable[int]) -> None:
    success_values = list(successes)
    success_rate = sum(success_values) / max(1, len(success_values))
    reward_window = rewards[-100:]
    step_window = steps[-100:]
    mean_reward = sum(reward_window) / max(1, len(reward_window))
    mean_steps = sum(step_window) / max(1, len(step_window))
    print(
        f"episode={episode:5d} "
        f"success_rate_100={success_rate:5.2f} "
        f"mean_reward_100={mean_reward:7.3f} "
        f"mean_steps_100={mean_steps:6.1f}"
    )


def greedy_rollout(
    env: ParkingLotEnvironment,
    policy: PolicyNetwork,
    goal_spot: str,
) -> Dict[str, object]:
    state = env.reset(goal_spot)
    total_reward = 0.0
    path = [state.position]
    actions: List[str] = []
    done = False
    info: Dict[str, object] = {"success": False}

    while not done:
        features = env.state_features()
        legal_actions = env.legal_actions()
        action = policy.greedy_action(features, legal_actions)
        state, reward, done, info = env.step(action)
        total_reward += reward
        path.append(state.position)
        actions.append(ACTIONS[action])

    return {
        "goal": goal_spot,
        "success": bool(info.get("success", False)),
        "steps": env.step_count,
        "total_reward": total_reward,
        "path": path,
        "actions": actions,
    }


def evaluate_all_goals(env: ParkingLotEnvironment, policy: PolicyNetwork) -> List[Dict[str, object]]:
    return [greedy_rollout(env, policy, goal) for goal in env.parking_spots]


def print_evaluation(env: ParkingLotEnvironment, policy: PolicyNetwork) -> None:
    print("\nGreedy policy evaluation:")
    results = evaluate_all_goals(env, policy)
    successes = sum(1 for result in results if result["success"])
    print(f"successes={successes}/{len(results)}")
    for result in results:
        status = "ok" if result["success"] else "fail"
        actions = ", ".join(result["actions"][:18])
        if len(result["actions"]) > 18:
            actions += ", ..."
        print(
            f"{result['goal']}: {status:4s} "
            f"steps={result['steps']:2d} "
            f"reward={result['total_reward']:6.2f} "
            f"actions=[{actions}]"
        )


def print_policy_snapshot(
    env: ParkingLotEnvironment,
    policy: PolicyNetwork,
    goal_spot: str,
    *,
    show_probabilities: bool = True,
) -> None:
    print(f"\nLearned policy for goal {goal_spot}:")
    print(
        "Legend: F=forward/north, B=backward/south, L=left/west, R=right/east, "
        "En/*=entrance action, G*=target, P*=blocked parking, ###=barrier."
    )
    print(_policy_grid(env, policy, goal_spot))

    if show_probabilities:
        print("\nAction probabilities by state:")
        print(f"{'State':<8} {'Greedy':<7} " + " ".join(f"{action:<10}" for action in ACTIONS))
        print("-" * 58)
        for state in _policy_states(env, goal_spot):
            features = env.state_features(state)
            legal_actions = env.legal_actions(state)
            probabilities = policy.probabilities(features, legal_actions)
            action = policy.greedy_action(features, legal_actions)
            probability_text = " ".join(
                f"{probabilities[action_index]:<10.2f}"
                for action_index in range(len(ACTIONS))
            )
            print(
                f"{_state_label(env, state):<8} "
                f"{ACTIONS[action]:<7} "
                f"{probability_text}"
            )


def _policy_grid(env: ParkingLotEnvironment, policy: PolicyNetwork, goal_spot: str) -> str:
    lines: List[str] = []
    for row in range(env.rows):
        cells: List[str] = []
        for col in range(env.cols):
            position = (row, col)
            label = _policy_cell_label(env, policy, goal_spot, position)
            cells.append(f"{label:^5}")
        lines.append("|" + "|".join(cells) + "|")
    return "\n".join(lines)


def _policy_cell_label(
    env: ParkingLotEnvironment,
    policy: PolicyNetwork,
    goal_spot: str,
    position: Tuple[int, int],
) -> str:
    if position in env.barriers:
        return "###"

    for spot_name, spot_position in env.parking_spots.items():
        if position == spot_position:
            if spot_name == goal_spot:
                return f"G{spot_name[1:]}"
            return spot_name

    state = ParkingState(position[0], position[1], env.start_heading, goal_spot)
    legal_actions = env.legal_actions(state)
    if not legal_actions:
        return "."
    features = env.state_features(state)
    action = policy.greedy_action(features, legal_actions)
    symbol = ACTION_SYMBOLS[ACTIONS[action]]
    if position == env.entrance:
        return f"En/{symbol}"
    return symbol


def _policy_states(env: ParkingLotEnvironment, goal_spot: str) -> List[ParkingState]:
    states: List[ParkingState] = []
    for row in range(env.rows):
        for col in range(env.cols):
            position = (row, col)
            if position == env.parking_spots[goal_spot]:
                continue
            if env.is_driveable(position, goal_spot):
                states.append(ParkingState(row, col, env.start_heading, goal_spot))
    return states


def _state_label(env: ParkingLotEnvironment, state: ParkingState) -> str:
    if state.position == env.entrance:
        return "En"
    return f"({state.row},{state.col})"


def print_policy_demo(
    env: ParkingLotEnvironment,
    policy: PolicyNetwork,
    goal_spot: str,
    *,
    max_steps: int = 20,
) -> None:
    state = env.reset(goal_spot)
    done = False
    info: Dict[str, object] = {"success": False}

    print(f"\nTrained policy rollout for goal {goal_spot}:")
    print("Legend: A^ is the agent facing north, G marks the target parking spot, ### marks barriers.")
    print("\nStart state:")
    print(env.render(state))

    for step_number in range(1, max_steps + 1):
        features = env.state_features()
        legal_actions = env.legal_actions()
        probabilities = policy.probabilities(features, legal_actions)
        action = policy.greedy_action(features, legal_actions)

        probability_text = ", ".join(
            f"{ACTIONS[action_index]}={probabilities[action_index]:.2f}"
            for action_index in legal_actions
        )
        print(f"\nStep {step_number}")
        print(f"Action probabilities: {probability_text}")
        print(f"Chosen action: {ACTIONS[action]}")

        state, reward, done, info = env.step(action)
        print(f"Reward: {reward:.2f}")
        print("Next state:")
        print(env.render(state))

        if done:
            if info.get("success", False):
                print(f"Reached {goal_spot} in {step_number} steps.")
            elif info.get("truncated", False):
                print(f"Stopped after hitting the environment step limit of {env.max_steps}.")
            break

    if not done:
        print(f"\nDemo stopped after {max_steps} displayed steps.")
