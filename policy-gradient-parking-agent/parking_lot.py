"""Parking-lot grid environment shared by the policy-gradient agents.

The default layout is modeled after the assignment figure:

* 6 rows by 7 columns
* entrance in the bottom-right cell
* eight parking spaces labeled P1-P8
* a hatched barrier row between the upper and lower parking spaces

Coordinates are zero-based as (row, column), with row 0 at the top.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Dict, List, Optional, Sequence, Tuple, Union


GridPosition = Tuple[int, int]

HEADINGS: Sequence[str] = ("north", "east", "south", "west")
HEADING_VECTORS: Dict[str, GridPosition] = {
    "north": (-1, 0),
    "east": (0, 1),
    "south": (1, 0),
    "west": (0, -1),
}

ACTIONS: Sequence[str] = ("forward", "backward", "left", "right")


@dataclass(frozen=True)
class ParkingState:
    row: int
    col: int
    heading: str
    goal_spot: str

    @property
    def position(self) -> GridPosition:
        return (self.row, self.col)


class ParkingLotEnvironment:
    """Small deterministic parking-lot environment for policy-gradient demos."""

    rows = 6
    cols = 7
    entrance: GridPosition = (5, 6)
    parking_spots: Dict[str, GridPosition] = {
        "P1": (2, 1),
        "P2": (2, 2),
        "P3": (2, 3),
        "P4": (2, 4),
        "P5": (4, 1),
        "P6": (4, 2),
        "P7": (4, 3),
        "P8": (4, 4),
    }
    barriers = frozenset({(3, 1), (3, 2), (3, 3), (3, 4)})

    def __init__(
        self,
        *,
        max_steps: int = 80,
        seed: Optional[int] = None,
        step_reward: float = -0.01,
        invalid_move_reward: float = -0.05,
        goal_reward: float = 1.0,
        distance_shaping: float = 0.0,
        start_heading: str = "north",
    ) -> None:
        if start_heading not in HEADINGS:
            raise ValueError(f"Unknown heading {start_heading!r}. Use one of {HEADINGS}.")

        self.max_steps = max_steps
        self.step_reward = step_reward
        self.invalid_move_reward = invalid_move_reward
        self.goal_reward = goal_reward
        self.distance_shaping = distance_shaping
        self.start_heading = start_heading
        self.rng = Random(seed)
        self.step_count = 0
        self.state = ParkingState(
            self.entrance[0],
            self.entrance[1],
            self.start_heading,
            "P1",
        )

    @property
    def feature_size(self) -> int:
        # row, col, goal row, goal col, delta row, delta col,
        # four heading bits, and eight goal-spot bits
        return 6 + len(HEADINGS) + len(self.parking_spots)

    def reset(self, goal_spot: Optional[str] = None) -> ParkingState:
        if goal_spot is None:
            goal_spot = self.rng.choice(list(self.parking_spots))
        self._validate_goal(goal_spot)

        self.step_count = 0
        self.state = ParkingState(
            self.entrance[0],
            self.entrance[1],
            self.start_heading,
            goal_spot,
        )
        return self.state

    def step(self, action: Union[int, str]) -> Tuple[ParkingState, float, bool, Dict[str, object]]:
        action_name = self._action_name(action)
        row_delta, col_delta = self._delta_for_action(self.state.heading, action_name)
        next_position = (self.state.row + row_delta, self.state.col + col_delta)
        old_distance = self._distance_to_goal(self.state.position, self.state.goal_spot)

        self.step_count += 1
        info: Dict[str, object] = {
            "action": action_name,
            "goal_spot": self.state.goal_spot,
            "blocked": False,
            "success": False,
            "truncated": False,
        }

        if not self.is_driveable(next_position, self.state.goal_spot):
            next_position = self.state.position
            reward = self.invalid_move_reward
            info["blocked"] = True
        else:
            new_distance = self._distance_to_goal(next_position, self.state.goal_spot)
            reward = self.step_reward + self.distance_shaping * (old_distance - new_distance)

        self.state = ParkingState(
            next_position[0],
            next_position[1],
            self.state.heading,
            self.state.goal_spot,
        )

        done = False
        if self.state.position == self.parking_spots[self.state.goal_spot]:
            reward = self.goal_reward
            done = True
            info["success"] = True
        elif self.step_count >= self.max_steps:
            done = True
            info["truncated"] = True

        return self.state, reward, done, info

    def legal_actions(self, state: Optional[ParkingState] = None) -> List[int]:
        state = state or self.state
        legal: List[int] = []
        for action_index, action_name in enumerate(ACTIONS):
            row_delta, col_delta = self._delta_for_action(state.heading, action_name)
            candidate = (state.row + row_delta, state.col + col_delta)
            if self.is_driveable(candidate, state.goal_spot):
                legal.append(action_index)
        return legal

    def state_features(self, state: Optional[ParkingState] = None) -> List[float]:
        state = state or self.state
        goal_row, goal_col = self.parking_spots[state.goal_spot]
        row_scale = max(1, self.rows - 1)
        col_scale = max(1, self.cols - 1)

        features = [
            state.row / row_scale,
            state.col / col_scale,
            goal_row / row_scale,
            goal_col / col_scale,
            (goal_row - state.row) / row_scale,
            (goal_col - state.col) / col_scale,
        ]
        features.extend(1.0 if state.heading == heading else 0.0 for heading in HEADINGS)
        features.extend(1.0 if state.goal_spot == spot else 0.0 for spot in self.parking_spots)
        return features

    def is_driveable(self, position: GridPosition, goal_spot: Optional[str] = None) -> bool:
        if not self._within_grid(position):
            return False
        if position in self.barriers:
            return False
        if position in self.parking_spots.values():
            return goal_spot is not None and position == self.parking_spots[goal_spot]
        return True

    def render(self, state: Optional[ParkingState] = None) -> str:
        state = state or self.state
        heading_marker = {"north": "^", "east": ">", "south": "v", "west": "<"}[state.heading]
        lines: List[str] = []
        for row in range(self.rows):
            cells: List[str] = []
            for col in range(self.cols):
                position = (row, col)
                label = " . "
                if position == self.entrance:
                    label = "En "
                if position in self.barriers:
                    label = "###"
                for spot_name, spot_position in self.parking_spots.items():
                    if position == spot_position:
                        label = f"{spot_name:<3}"
                        if spot_name == state.goal_spot:
                            label = f"G{spot_name[1]:<2}"
                if position == state.position:
                    label = f"A{heading_marker} "
                cells.append(label)
            lines.append("|" + "|".join(cells) + "|")
        return "\n".join(lines)

    def _validate_goal(self, goal_spot: str) -> None:
        if goal_spot not in self.parking_spots:
            valid = ", ".join(self.parking_spots)
            raise ValueError(f"Unknown goal spot {goal_spot!r}. Use one of: {valid}.")

    def _within_grid(self, position: GridPosition) -> bool:
        row, col = position
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _distance_to_goal(self, position: GridPosition, goal_spot: str) -> int:
        goal_row, goal_col = self.parking_spots[goal_spot]
        return abs(goal_row - position[0]) + abs(goal_col - position[1])

    def _action_name(self, action: Union[int, str]) -> str:
        if isinstance(action, int):
            try:
                return ACTIONS[action]
            except IndexError as exc:
                raise ValueError(f"Action index {action} is out of range.") from exc
        if action not in ACTIONS:
            raise ValueError(f"Unknown action {action!r}. Use one of {ACTIONS}.")
        return action

    def _delta_for_action(self, heading: str, action: str) -> GridPosition:
        heading_index = HEADINGS.index(heading)
        if action == "forward":
            movement_heading = heading
        elif action == "backward":
            movement_heading = HEADINGS[(heading_index + 2) % len(HEADINGS)]
        elif action == "left":
            movement_heading = HEADINGS[(heading_index - 1) % len(HEADINGS)]
        elif action == "right":
            movement_heading = HEADINGS[(heading_index + 1) % len(HEADINGS)]
        else:
            raise ValueError(f"Unknown action {action!r}.")
        return HEADING_VECTORS[movement_heading]


if __name__ == "__main__":
    env = ParkingLotEnvironment(seed=7)
    env.reset("P1")
    print(env.render())
    print("Legal actions:", [ACTIONS[action] for action in env.legal_actions()])
