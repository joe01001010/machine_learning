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

    def reset(self):
        """
        This function takes no arguments
        This function will update the agent with a fresh start
        This function will return the current position of the agent
        """
        self.agent_location = self.entrance
        self.parked = self.agent_location == self.goal
        return self.get_state()

    def get_state(self):
        """
        This function takes no arguments
        This function will return the location of the agent
        """
        return self.agent_location

    def get_dqn_state(self):
        """
        This function takes no arguments
        This function will return the location of the agent
        The location of the goal
        The distance from the agent to the goal
        """
        agent_row, agent_col = self.agent_location
        goal_row, goal_col = self.goal
        return (
            agent_row / (self.rows - 1),
            agent_col / (self.columns - 1),
            goal_row / (self.rows - 1),
            goal_col / (self.columns - 1),
            (goal_row - agent_row) / (self.rows - 1),
            (goal_col - agent_col) / (self.columns - 1)
        )

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
            return self.get_state(), -5, False
        elif next_position in self.barriers:
            return self.get_state(), -5, False
        elif next_position in self.parking_spots and next_position != self.goal:
            return self.get_state(), -10, False
        elif next_position == self.goal:
            self.agent_location = next_position
            self.parked = True
            return self.get_state(), 100, True
        else:
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