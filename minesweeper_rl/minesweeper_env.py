# minesweeper_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time

class MinesweeperGame:
    def __init__(self, rows=9, cols=9, num_mines=10):
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.board = None
        self.visible = None
        self.game_over = False
        self.won = False
        self.first_move = True
        self.start_time = None
        

    def start_game(self, rows=None, cols=None, num_mines=None):
        if rows: self.rows = rows
        if cols: self.cols = cols
        if num_mines: self.num_mines = num_mines
            
        self.board = np.full((self.rows, self.cols), -2, dtype=int)  # -2 = hidden
        self.visible = np.zeros((self.rows, self.cols), dtype=bool)
        self.game_over = False
        self.won = False
        self.first_move = True
        self.start_time = time.time()
        
        self.mine_positions = set()
        return self._get_observation()
    

    def select_cell(self, row, col):
        """Only reveal action - no flagging"""
        if self.game_over:
            return self._get_observation()
            
        if self.first_move:
            self._place_mines(row, col)
            self.first_move = False
            
        return self._reveal_cell(row, col)
    

    def _place_mines(self, safe_row, safe_col):
        safe_cells = set()
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r, c = safe_row + dr, safe_col + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    safe_cells.add((r, c))
        
        all_positions = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        mine_positions = [pos for pos in all_positions if pos not in safe_cells]
        
        self.mine_positions = set(random.sample(mine_positions, self.num_mines))
        
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in self.mine_positions:
                    self.board[r, c] = -1  # Mine
                else:
                    count = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                                if (nr, nc) in self.mine_positions:
                                    count += 1
                    self.board[r, c] = count  # Number of adjacent mines
    

    def _reveal_cell(self, row, col):
        """Reveal cell - no flagging checks needed"""
        if (row, col) in self.mine_positions:
            # Hit a mine - game over
            self.game_over = True
            self.visible[row, col] = True
        else:
            # Reveal cell and cascade
            self._reveal_recursive(row, col)
            
            # Check win condition: all non-mine cells revealed
            if np.sum(~self.visible) == self.num_mines:
                self.won = True
                self.game_over = True
                
        return self._get_observation()
    

    def _reveal_recursive(self, row, col):
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return
        if self.visible[row, col]:  # No flag check needed
            return
            
        self.visible[row, col] = True
        
        if self.board[row, col] == 0:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    self._reveal_recursive(row + dr, col + dc)
    

    def _get_observation(self):
        obs = self.board.copy()
        obs[~self.visible] = -2  # Hidden cells
        return obs
    

    def get_game_state(self):
        current_time = time.time() - self.start_time if self.start_time else 0
        return {
            "board": self._get_observation(),
            "game_over": self.game_over,
            "won": self.won,
            "mines_remaining": self.num_mines,  # No flagging, so all mines remain until end
            "time_elapsed": int(current_time),
            "visible": self.visible.copy()
        }


class MinesweeperEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 60}
    

    def __init__(self, rows=9, cols=9, num_mines=10, render_mode=None):
        super().__init__()
        
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.render_mode = render_mode
        self.steps = 0
        
        self.game = MinesweeperGame(rows, cols, num_mines)
        
        # Action space: only (row, col) - no action_type since only reveal
        self.action_space = spaces.MultiDiscrete([rows, cols])
        
        # Update observation space to be a Dict space
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=-2, high=8, shape=(rows, cols), dtype=int),
            'visible': spaces.Box(low=0, high=1, shape=(rows, cols), dtype=bool),
            'flagged': spaces.Box(low=0, high=1, shape=(rows, cols), dtype=bool)
        })
    

    def _get_obs(self):
        """
        Returns a structured observation dictionary.
        """
        return {
            'board': self.game.board.copy(),
            'visible': self.game.visible.copy(),
            'flagged': np.zeros((self.rows, self.cols), dtype=bool)  # Placeholder since your game doesn't have flags yet
        }


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.game.start_game(self.rows, self.cols, self.num_mines)
        self.previous_visible = self.game.visible.copy()
        info = {
            "mines_remaining": self.num_mines,
            "game_over": False,
            "won": False,
            "time_elapsed": 0
        }
        return self._get_obs(), info  # Return the rich observation
    

    def step(self, action):
        # Action is just (row, col) - no action_type
        row, col = action
        
        observation = self.game.select_cell(int(row), int(col))
        
        reward = self._calculate_reward()
        
        terminated = self.game.game_over
        truncated = False
        
        game_state = self.game.get_game_state()
        info = {
            "mines_remaining": game_state["mines_remaining"],
            "game_over": game_state["game_over"],
            "won": game_state["won"],
            "time_elapsed": game_state["time_elapsed"],
            "valid_actions": self._get_valid_actions()
        }
        self.previous_visible = self.game.visible.copy()
        self.steps += 1
        return self._get_obs(), reward, terminated, truncated, info
    

    def _calculate_reward(self):
        # Terminal rewards
        if self.game.won:
            return 20.0
        elif self.game.game_over and not self.game.won:
            return -5.0

        reward = 0.0

        # Reward for newly revealed tiles
        newly_revealed = np.logical_and(self.game.visible, ~self.previous_visible)
        num_new = np.sum(newly_revealed)

        # Give a reward for revealing new tiles
        # Or give a penalty for clicking a tile that doesnt end the game but also doesnt make any progress
        if num_new > 0:
            reward += num_new * 0.3

            # give an extra bonus for revealing cells near numbers
            for row, column in zip(*np.where(newly_revealed)):
                if self.game.board[row, column] == 0:
                    reward += 0.2
                forced_move_bonus = self._check_forced_move_pattern(row, column)
                reward += forced_move_bonus
        else:
            reward -= 0.1

        # Bonus for revealing tiles adjacent to known numbers
        for r, c in zip(*np.where(newly_revealed)):
            neighbors = [
                (r + dr, c + dc)
                for dr in [-1, 0, 1]
                for dc in [-1, 0, 1]
                if 0 <= r + dr < self.rows and 0 <= c + dc < self.cols
            ]
            for nr, nc in neighbors:
                if self.game.visible[nr, nc] and self.game.board[nr, nc] > 0:
                    reward += 0.05  # local structure bonus

        # Optional: small time penalty to encourage efficiency
        reward -= 0.001 * self.steps  # slowly discourages long episodes

        return reward
    
    
    def _get_valid_actions(self):
        """Only reveal actions on hidden cells"""
        valid_actions = []
        for row in range(self.rows):
            for col in range(self.cols):
                if not self.game.visible[row, col]:
                    valid_actions.append((row, col))
        return valid_actions
    

    def render(self):
        """Simplified rendering - no flags"""
        obs = self.game._get_observation()
        print("\n" + "=" * (self.cols * 2 + 3))
        for r in range(self.rows):
            row_str = "|"
            for c in range(self.cols):
                cell = obs[r, c]
                if cell == -2:
                    row_str += ". "
                elif cell == -1:
                    row_str += "* "
                elif cell == 0:
                    row_str += "  "
                else:
                    row_str += f"{cell} "
            row_str += "|"
            print(row_str)
        print("=" * (self.cols * 2 + 3))
            
        state = self.game.get_game_state()
        print(f"Mines: {state['mines_remaining']}")
        if state['game_over']:
            print("GAME OVER -", "YOU WIN!" if state['won'] else "YOU LOSE!")
    

    def close(self):
        pass


    def get_action_space_size(self):
        return self.rows * self.cols


    def get_observation_space_size(self):
        return self.rows * self.cols


    def _check_forced_move_pattern(self, r, c):
        """Reward for revealing cells that create obvious patterns"""
        bonus = 0.0
        
        # Check if this creates a 1-1 or 1-2 pattern
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.rows and 0 <= nc < self.cols and 
                    self.game.visible[nr, nc] and self.game.board[nr, nc] > 0):
                    
                    # Count hidden neighbors for this number
                    hidden_count = 0
                    for dr2 in [-1, 0, 1]:
                        for dc2 in [-1, 0, 1]:
                            nr2, nc2 = nr + dr2, nc + dc2
                            if (0 <= nr2 < self.rows and 0 <= nc2 < self.cols and 
                                not self.game.visible[nr2, nc2]):
                                hidden_count += 1
                    
                    # If this number now has exactly the right number of hidden cells
                    if hidden_count == self.game.board[nr, nc]:
                        bonus += 0.3  # Pattern recognition bonus     
        return bonus