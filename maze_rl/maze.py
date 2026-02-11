#!/usr/bin/env python
from collections import defaultdict
import random, os, time

ACTIONS         = ["U", "D", "L", "R"]
SLEEP_DELAY     = 1
STABLE_CRITERIA = 50
ARROWS          = { "U": "↑", "D": "↓", "L": "←", "R": "→" }
ROWS            = 5
COLUMNS         = 5
GOAL            = (ROWS - 1, COLUMNS - 1)
EPISODES        = 10000
WALLS           = {
  (2, 1), (2, 2), (2, 3), (2, 4), (4, 2)
}


class Maze:
  def __init__(self):
    self.rows    = ROWS
    self.columns = COLUMNS
    self.start   = (0, 0)
    self.goal    = GOAL
    self.walls   = WALLS


  def execute_action(self, state, action):
    """
    This function takes two areguments
    state is the current state of the agent
    action is the action selected by the agent
    This function will return 3 values
    the state if an invalid action is selected or the new state if a valid action was selected
    the reward based on the action taken
    a boolean value representing if the terminal state was reached
    """
    row, column = state

    if action == "U": row -= 1
    if action == "D": row += 1
    if action == "L": column -= 1
    if action == "R": column += 1

    if row < 0 or row >= self.rows or column < 0 or column >= self.columns:
      return state, -1, False

    if (row, column) in self.walls:
      return state, -1, False

    new_state = (row, column)

    if new_state == self.goal:
      return new_state, 100, True

    return new_state, -1, False


def random_policy():
  """
  This function takes no arguments
  This function will return a dict of tuples representing all actions with a value assigned to the actions in every state
  This tupple is the probability of choosing an action for each state
  """
  policy = {}

  for row in range(ROWS):
    for column in range(COLUMNS):
      policy[row, column] = {a: 0.25 for a in ACTIONS}

  return policy


def generate_episode(env, policy):
  """
  This function takes two arguments
  env is the environment object that is being used for this agent
  policy is the policy generated for this agent to use
  """
  episode = []
  state = env.start
  done = False

  while not done:
    actions = list(policy[state].keys())
    probs = list(policy[state].values())
    action = random.choices(actions, probs)[0]

    next_state, reward, done = env.execute_action(state, action)
    episode.append((state, action, reward))

    state = next_state

  return episode


def mc_evaluate(Q, returns, episode, gamma=0.8):
  """
  This function takes 4 arguments
  Q is the list of state action pairs with the sum of returns
  returns is the state action pairs that holds the rewards in a list
  episode is the complete number of episodes ran
  gamma is the discount factor
  """
  G = 0
  visited = set()

  for time in reversed(range(len(episode))):
    state, action, reward = episode[time]
    G = gamma * G + reward

    if ((state, action) not in visited):
      visited.add((state, action))

      returns[(state, action)].append(G)
      Q[(state, action)] = sum(returns[(state, action)]) / len(returns[(state, action)])


def improve_policy(policy, Q, epsilon=0.1):
  """
  This function takes two arguments
  policy is the policy the agent is trying to pick actions to align with
  Q is the list of state action tuples for all the episodes
  This function will return the stable policy improvements
  This function will attempt to find the best action and then greedily update the policy with the best action
  """
  stable = True

  for state in policy.keys():

    old_best = max(policy[state], key=policy[state].get)

    qvals = {action: Q[(state, action)] for action in ACTIONS}
    best_action = max(qvals, key=qvals.get)

    epsilon_greedy = len(ACTIONS)
    for action in ACTIONS:
      if action == best_action:
        policy[state][action] = 1 - epsilon + epsilon/epsilon_greedy
      else:
        policy[state][action] = epsilon/epsilon_greedy

    if old_best != best_action:
      stable = False

  return stable


def monte_carlo_control(episodes):
  """
  This function takes one argument being the number of episodes to run to learn a policy
  This function will return the policy and Q table
  This function will create the environment then get the random policy
  This will then get the Q table initialized and returns
  Then this funciton will iterate over the specified episodes to run and evaluade the episodes to improve the policy
  """
  env = Maze()
  policy = random_policy()
  total_episodes = 0

  Q = defaultdict(float)
  returns = defaultdict(list)
  total_steps = 0
  stable_count = 0

  for i in range(episodes):

    episode = generate_episode(env, policy)
    total_steps += len(episode)

    mc_evaluate(Q, returns, episode)

    stable = improve_policy(policy, Q)
    total_episodes += 1

    if stable:
      stable_count += 1
    else:
      stable_count = 0

    if stable_count >= STABLE_CRITERIA:
      return policy, Q, total_episodes, total_steps


  return policy, Q, total_episodes, total_steps


def display_board(rows, columns, walls, goal, agent_location):
  """
  This function takes 5 arguments
  ROWS is the global number of rows in the environmnet
  COLUMNS is the global number of columns in the environment
  WALLS is the global wall locations in the environment
  GOAL is the global goal for the agent to get to
  agent_location is the location of the agent in the maze at a specific time
  This function will show the agent moving through the maze
  This function will clear the screen after each move
  """
  clear_screen()
  for row in range(rows):
    for column in range(columns):
      if (row, column) == agent_location and (row, column) == goal:
        print(" [AG] ", end='')
      else:
        if (row, column) == agent_location:
          print(" [A] ", end='')
        elif (row, column) in walls:
          print(" [#] ", end='')
        elif (row, column) == goal:
          print(" [G] ", end='')
        else:
          print(" [ ] ", end='')
    print()

  time.sleep(SLEEP_DELAY)


def clear_screen():
  """
  This function takes no arguments
  This function doesnt return anything
  This function will simply clear stdout for displaying the maze and updating each step
  """
  if os.name == 'nt':
    os.system('cls')
  else:
    os.system('clear')


def follow_policy(env, policy):
  """
  This function takes two arguments
  env is the object of the environment for the agent to explore
  policy is the optimal policy learned by the agent
  """
  state = env.start
  done = False

  while not done:
    display_board(ROWS, COLUMNS, WALLS, GOAL, state)

    action = max(policy[state], key=policy[state].get)

    state, _, done = env.execute_action(state, action)

  display_board(ROWS, COLUMNS, WALLS, GOAL, state)


def display_policy(policy):
  """
  This function takes the policy as an argument
  This will iterate over each step taken in the optimal policy found
  This will call the display_board function to display the optimal policy in the environment that the agent found
  """
  for step in policy:
    display_board(ROWS, COLUMNS, WALLS, GOAL, step)


def print_policy_display(policy):
  """
  This function takes one argument
  policy is the ranking for which action to take from each state
  This function will return nothing
  This function prints to stdout the optimal policy that was discovered through training
  """
  print("\nOptimal policy discovered through training")
  for row in range(ROWS):
    for column in range(COLUMNS):
      state = (row, column)

      if state in WALLS:
        print(" [#] ", end='')
      elif state == GOAL:
        print(" [G] ", end='')
      else:
        best_action = max(policy[state], key=policy[state].get)
        print(f" [{ARROWS[best_action]}] ", end='')

    print()


def main():
  start_time = time.time()
  policy, Q, total_episodes, total_steps = monte_carlo_control(episodes=EPISODES)
  total_time = f"{(time.time() - start_time):.3f}"
  follow_policy(Maze(), policy)

  print("=" * 100)

  print(f"Total rows: {ROWS}")
  print(f"Total columns: {COLUMNS}")
  print(f"Total time: {total_time}")
  print(f"Total episodes: {total_episodes}")
  print(f"Average steps per episode: {(total_steps / total_episodes):.2f}")

  print_policy_display(policy)  

  print("=" * 100)


if __name__ == '__main__':
  main()
