import sys
import numpy as np

from logging_init import configure_logging
LOGGER = configure_logging(__file__)

class ParkingLot:
    def __init__(self, rows, columns, goal=1):
        self.rows = rows
        self.columns = columns
        self.entrance = (rows - 1, columns - 1)
        self.barriers = [(rows // 2, column) for column in range(1, columns - 1)]
        self.parking_spots = [(barrier[0] + offset, barrier[1]) for barrier in self.barriers for offset in (-1, 1)]
        self.goal = self.parking_spots[goal]
        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.reset()
        LOGGER.info(
            f"Initializing constructor for {self.__class__.__name__}"
            f"Rows: {self.rows}"
            f"Columns: {self.columns}"
            f"Entrance: {self.entrance}"
            f"Barriers: {self.barriers}"
            f"Parking Spots: {self.parking_spots}"
            f"Goal: {self.goal}"
            f"Actions: {self.actions}"
        )

    def reset(self):
        """
        This function takes no arguments
        This function will update the agent with a fresh start
        This function will return the current position of the agent
        """
        self.agent_location = self.entrance
        self.parked = self.agent_location == self.goal
        LOGGER.debug(f"Resetting state and returning {self.get_state()} from {sys._getframe().f_code.co_name}")
        return self.get_state()

    def get_state(self):
        """
        This function takes no arguments
        This function will return the location of the agent
        """
        LOGGER.debug(f"Getting state and returning: {self.agent_location} from {sys._getframe().f_code.co_name}")
        return self.agent_location

    def get_dqn_state(self):
        """
        This function takes no arguments
        This function will return 5 channels representing the environment
        First channel for free spaces
        Second channel for agent location
        Third channel for barriers
        Fourth channel for parking spots
        Fifth channel for the goal
        """
        state = np.zeros((5, self.rows, self.columns), dtype=np.float32)
        state[0, :, :] = 1.0
        for row, col in self.barriers:
            state[0, row, col] = 0.0
            state[2, row, col] = 1.0

        for row, col in self.parking_spots:
            state[0, row, col] = 0.0
            if (row, col) == self.goal:
                state[4, row, col] = 1.0
            else:
                state[3, row, col]

        agent_row, agent_col = self.agent_location
        state[0, agent_row, agent_col] = 0.0
        state[1, agent_row, agent_col] = 1.0

        LOGGER.debug(
            f"Getting DQN state and returning: {state} from {sys._getframe().f_code.co_name}"
        )
        return state

    def move_agent(self, action):
        """
        This function takes one argument
        action is the ID 0, 1, 2, 3 of either up, down, left, right
        This function will check to see if the action is out of bounds
        Check if the agent moves into a barrier
        Check if the agent enters a parking space, and if its the correct parking space
        This function will update the agent's location and return a reward as an int if a valid move is made
        """
        curr_row, curr_col = self.get_state()
        row_mod, col_mod = self.actions[action]
        new_row = curr_row + row_mod
        new_col = curr_col + col_mod
        next_position = (new_row, new_col)
        if new_row < 0 or new_row >= self.rows or new_col < 0 or new_col >= self.columns:
            LOGGER.debug(f"new_row: {new_row} >= self.rows: {self.rows} or new_col: {new_col} >= self.columns: {self.columns} from {sys._getframe().f_code.co_name}")
            return self.get_state(), -5, False
        elif next_position in self.barriers:
            LOGGER.debug(f"Next position was in barriers: {next_position} barriers: {self.barriers} from {sys._getframe().f_code.co_name}")
            return self.get_state(), -5, False
        elif next_position in self.parking_spots and next_position != self.goal:
            LOGGER.debug(f"Next position was in wrong parking spot: {next_position} barriers: {self.parking_spots}, goal: {self.goal} from {sys._getframe().f_code.co_name}")
            return self.get_state(), -10, False
        elif next_position == self.goal:
            LOGGER.debug(f"Agent found the goal, updating agents location and setting parked to true from {sys._getframe().f_code.co_name}")
            self.agent_location = next_position
            self.parked = True
            return self.get_state(), 100, True
        else:
            LOGGER.debug(f"Agent made a valid move but it was not the goal, move: {next_position}, goal: {self.goal} from {sys._getframe().f_code.co_name}")
            self.agent_location = next_position
            return self.get_state(), -1, False

    def __str__(self):
        """
        This function takes no arguments
        This function will print the current representation of the parking lot
        This function will return a string that represents the parking lot and all entities inside
        """
        parking_lot_status = ""
        for row in range(self.rows):
            for column in range(self.columns):
                if (row, column) == self.agent_location:
                    parking_lot_status += "[A]"
                elif (row, column) in self.parking_spots:
                    parking_lot_status += "[P]"
                elif (row, column) in self.barriers:
                    parking_lot_status += "[X]"
                else:
                    parking_lot_status += "[ ]"
            parking_lot_status += "\n"

        return parking_lot_status

    def __repr__(self):
        return f"Rows: {self.rows}, Columns: {self.columns}, Goal: {self.goal}, Entrance: {self.entrance}"
