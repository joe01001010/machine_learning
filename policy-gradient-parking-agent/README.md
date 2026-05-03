# Policy-Gradient Parking Agent

This project implements the parking-lot environment from the assignment figure and three policy-gradient agents:

- `reinforce.py`
- `reinforce_baseline.py`
- `actor_critic.py`

The code uses only the Python standard library. The small policy and value networks are implemented directly in `networks.py`.

## Layout

The shared environment is in `parking_lot.py`. It models the 6 by 7 grid in the PDF:

- entrance: bottom-right cell `(5, 6)`
- parking spaces: `P1`-`P4` on row `2`, columns `1`-`4`
- barrier cells: row `3`, columns `1`-`4`
- parking spaces: `P5`-`P8` on row `4`, columns `1`-`4`

Parking spaces are blocked unless the cell is the current goal parking space.

## Run

From this directory:

```bash
python parking_lot.py
python reinforce.py --episodes 2000
python reinforce_baseline.py --episodes 2000
python actor_critic.py --episodes 2000
```

Compare all three agents and print each learned policy plus a rollout demo:

```bash
python compare_policy_gradient_agents.py --episodes 2000 --policy-goal P8 --demo-goal P8
```

Use `--policy-goal all` to print a policy map for every parking spot, or
`--no-policy-probabilities` to show only the greedy action grid.

Train for one specific parking spot:

```bash
python actor_critic.py --goal P8 --episodes 1000
```

Useful knobs:

```bash
python reinforce_baseline.py --episodes 3000 --gamma 0.99 --policy-lr 0.06 --value-lr 0.01 --hidden-size 32
```

Plain REINFORCE is high variance, so `reinforce.py` defaults to normalized returns, a smaller learning rate, and a small entropy bonus. If a run collapses to one repeated action, increase exploration a bit:

```bash
python reinforce.py --episodes 4000 --demo-goal P8 --entropy-coef 0.03
```

Each training script prints a final greedy evaluation for all eight goal spaces.

`reinforce_baseline.py` computes the learned-baseline advantages from a frozen
episode trajectory, centers `G_t - V(s_t)` before the policy update, and then
fits the value network to the uncentered Monte Carlo returns. That keeps the
critic from changing the advantage estimates midway through the same episode.

After evaluation, each script also prints a text rollout of the trained agent. The rollout shows the grid, the policy's action probabilities for the current legal moves, the greedy action selected, and the next grid state:

```bash
python actor_critic.py --episodes 1000 --demo-goal P8 --demo-steps 12
```

Skip that display with `--no-demo`.

By default the scripts use a small distance-shaping reward (`--distance-shaping 0.02`) so the agents get feedback before they happen to reach a parking space. To run the sparse-reward version, set:

```bash
python reinforce.py --distance-shaping 0 --episodes 2000
```
