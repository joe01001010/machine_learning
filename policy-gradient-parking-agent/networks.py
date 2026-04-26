"""Tiny neural networks implemented with the Python standard library only."""

from __future__ import annotations

import math
from random import Random
from typing import Iterable, List, Optional, Sequence, Tuple


def clip(value: float, limit: Optional[float]) -> float:
    if limit is None:
        return value
    return max(-limit, min(limit, value))


class PolicyNetwork:
    """One-hidden-layer softmax policy network."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        *,
        learning_rate: float = 0.02,
        seed: Optional[int] = None,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.rng = Random(seed)

        hidden_limit = math.sqrt(6.0 / (input_size + hidden_size))
        output_limit = math.sqrt(6.0 / (hidden_size + output_size))
        self.w1 = [
            [self.rng.uniform(-hidden_limit, hidden_limit) for _ in range(hidden_size)]
            for _ in range(input_size)
        ]
        self.b1 = [0.0 for _ in range(hidden_size)]
        self.w2 = [
            [self.rng.uniform(-output_limit, output_limit) for _ in range(output_size)]
            for _ in range(hidden_size)
        ]
        self.b2 = [0.0 for _ in range(output_size)]

    def probabilities(self, features: Sequence[float], legal_actions: Optional[Iterable[int]] = None) -> List[float]:
        _, hidden, logits = self._forward(features)
        return self._masked_softmax(logits, legal_actions)

    def sample_action(self, features: Sequence[float], legal_actions: Optional[Iterable[int]] = None) -> int:
        probabilities = self.probabilities(features, legal_actions)
        draw = self.rng.random()
        cumulative = 0.0
        for action, probability in enumerate(probabilities):
            cumulative += probability
            if draw <= cumulative:
                return action
        return len(probabilities) - 1

    def greedy_action(self, features: Sequence[float], legal_actions: Optional[Iterable[int]] = None) -> int:
        probabilities = self.probabilities(features, legal_actions)
        return max(range(len(probabilities)), key=lambda action: probabilities[action])

    def update_log_policy(
        self,
        features: Sequence[float],
        action: int,
        advantage: float,
        legal_actions: Optional[Iterable[int]] = None,
        entropy_weight: float = 0.0,
    ) -> None:
        _, hidden, logits = self._forward(features)
        probabilities = self._masked_softmax(logits, legal_actions)

        delta_logits = [-probability for probability in probabilities]
        delta_logits[action] += 1.0
        scaled_delta = [advantage * delta for delta in delta_logits]

        if entropy_weight:
            entropy_delta = self._entropy_logit_delta(probabilities)
            scaled_delta = [
                policy_delta + entropy_weight * entropy_delta[action_index]
                for action_index, policy_delta in enumerate(scaled_delta)
            ]

        old_w2 = [row[:] for row in self.w2]
        for hidden_index in range(self.hidden_size):
            for action_index in range(self.output_size):
                self.w2[hidden_index][action_index] += (
                    self.learning_rate * hidden[hidden_index] * scaled_delta[action_index]
                )
        for action_index in range(self.output_size):
            self.b2[action_index] += self.learning_rate * scaled_delta[action_index]

        hidden_delta = []
        for hidden_index in range(self.hidden_size):
            downstream = sum(
                old_w2[hidden_index][action_index] * scaled_delta[action_index]
                for action_index in range(self.output_size)
            )
            hidden_delta.append((1.0 - hidden[hidden_index] ** 2) * downstream)

        for input_index, feature in enumerate(features):
            for hidden_index in range(self.hidden_size):
                self.w1[input_index][hidden_index] += (
                    self.learning_rate * feature * hidden_delta[hidden_index]
                )
        for hidden_index in range(self.hidden_size):
            self.b1[hidden_index] += self.learning_rate * hidden_delta[hidden_index]

    def _forward(self, features: Sequence[float]) -> Tuple[List[float], List[float], List[float]]:
        if len(features) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {len(features)}.")

        z1 = []
        for hidden_index in range(self.hidden_size):
            total = self.b1[hidden_index]
            for input_index, feature in enumerate(features):
                total += feature * self.w1[input_index][hidden_index]
            z1.append(total)

        hidden = [math.tanh(value) for value in z1]
        logits = []
        for action_index in range(self.output_size):
            total = self.b2[action_index]
            for hidden_index, hidden_value in enumerate(hidden):
                total += hidden_value * self.w2[hidden_index][action_index]
            logits.append(total)
        return z1, hidden, logits

    def _masked_softmax(self, logits: Sequence[float], legal_actions: Optional[Iterable[int]]) -> List[float]:
        if legal_actions is None:
            legal = set(range(self.output_size))
        else:
            legal = set(legal_actions)
        if not legal:
            legal = set(range(self.output_size))

        max_logit = max(logits[action] for action in legal)
        exp_values = [0.0 for _ in logits]
        for action in legal:
            exp_values[action] = math.exp(logits[action] - max_logit)
        total = sum(exp_values)
        if total == 0.0:
            return [1.0 / self.output_size for _ in logits]
        return [value / total for value in exp_values]

    def _entropy_logit_delta(self, probabilities: Sequence[float]) -> List[float]:
        entropy_terms = [
            probability * (math.log(max(probability, 1e-12)) + 1.0)
            for probability in probabilities
            if probability > 0.0
        ]
        expected_term = sum(entropy_terms)
        return [
            probability * (expected_term - (math.log(max(probability, 1e-12)) + 1.0))
            if probability > 0.0
            else 0.0
            for probability in probabilities
        ]


class ValueNetwork:
    """One-hidden-layer state-value network."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        learning_rate: float = 0.04,
        seed: Optional[int] = None,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.rng = Random(seed)

        hidden_limit = math.sqrt(6.0 / (input_size + hidden_size))
        output_limit = math.sqrt(6.0 / (hidden_size + 1))
        self.w1 = [
            [self.rng.uniform(-hidden_limit, hidden_limit) for _ in range(hidden_size)]
            for _ in range(input_size)
        ]
        self.b1 = [0.0 for _ in range(hidden_size)]
        self.w2 = [self.rng.uniform(-output_limit, output_limit) for _ in range(hidden_size)]
        self.b2 = 0.0

    def predict(self, features: Sequence[float]) -> float:
        _, hidden, value = self._forward(features)
        return value

    def update(self, features: Sequence[float], target: float) -> float:
        _, hidden, value = self._forward(features)
        error = target - value

        old_w2 = self.w2[:]
        for hidden_index in range(self.hidden_size):
            self.w2[hidden_index] += self.learning_rate * error * hidden[hidden_index]
        self.b2 += self.learning_rate * error

        hidden_delta = [
            (1.0 - hidden[hidden_index] ** 2) * old_w2[hidden_index] * error
            for hidden_index in range(self.hidden_size)
        ]
        for input_index, feature in enumerate(features):
            for hidden_index in range(self.hidden_size):
                self.w1[input_index][hidden_index] += (
                    self.learning_rate * feature * hidden_delta[hidden_index]
                )
        for hidden_index in range(self.hidden_size):
            self.b1[hidden_index] += self.learning_rate * hidden_delta[hidden_index]
        return 0.5 * error * error

    def _forward(self, features: Sequence[float]) -> Tuple[List[float], List[float], float]:
        if len(features) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {len(features)}.")

        z1 = []
        for hidden_index in range(self.hidden_size):
            total = self.b1[hidden_index]
            for input_index, feature in enumerate(features):
                total += feature * self.w1[input_index][hidden_index]
            z1.append(total)

        hidden = [math.tanh(value) for value in z1]
        value = self.b2
        for hidden_index, hidden_value in enumerate(hidden):
            value += hidden_value * self.w2[hidden_index]
        return z1, hidden, value
