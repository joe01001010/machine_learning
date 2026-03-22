import random
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from logging_init import configure_logging
LOGGER = configure_logging(__file__)


class BoardCNN(nn.Module):
    def __init__(self, rows, columns, action_size):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        conv_output_size = 64 * rows * columns

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)


class _QModel(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        LOGGER.info(
            f"Initializing constructor for {self.__class__.__name__}, "
            f"Network: {self.network}"
        )

    def forward(self, x):
        return self.network(x)


class DeepQNetwork:
    def __init__(self,
        rows=7,
        columns=6,
        action_size=4,
        alpha=0.0005,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.997,
        epsilon_min=0.05,
        batch_size=64,
        memory_size=20000,
        target_update_frequency=250,
    ):
        self.rows = rows
        self.columns = columns
        self.action_size = action_size

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

        self.memory = deque(maxlen=memory_size)
        self.training_steps = 0
        self.state_method = "get_dqn_state"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = BoardCNN(rows, columns, action_size).to(self.device)
        self.target_network = BoardCNN(rows, columns, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)
        self.loss_function = nn.MSELoss()

        LOGGER.info(
            f"Initializing constructor for {self.__class__.__name__}, "
            f"Rows: {self.rows}, "
            f"Columns: {self.columns}, "
            f"Action size: {self.action_size}, "
            f"Alpha: {self.alpha}, "
            f"Gamma: {self.gamma}, "
            f"Epsilon: {self.epsilon}, "
            f"Epsilon Decay: {self.epsilon_decay}, "
            f"Epsilon Min: {self.epsilon_min}, "
            f"Batch Size: {self.batch_size}, "
            f"Target Update Frequency: {self.target_update_frequency}, "
            f"State Method: {self.state_method}, "
            f"Device: {self.device}"
        )


    def _state_to_array(self, state):
        """
        This function takes one argument
        state is the current state of the agent
        This function will conver the state into a numpy array
        """
        LOGGER.debug(f"Converting state into array from {sys._getframe().f_code.co_name}: "
            f"{state} -> {np.array(state, dtype=np.float32)}"
        )
        return np.array(state, dtype=np.float32)


    def choose_action(self, state):
        """
        This function takes one argument
        state is the current state of the agent
        This function will choose a random action if random is less than epsilon value
        Else this function will convert the state to a numpy array and then a tensor
        This function will then pass the state tensor to the neural network which will return q values
        This function will return an integer that is either the q value or dimension
        """
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2, 3])

        state_array = self._state_to_array(state)
        state_tensor = torch.tensor(state_array, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())


    def _store_experience(self, state, action, reward, next_state, done):
        """
        This function takes five arguments
        state is the current state of the agent
        action is the action selected in that state
        reward is the reward given to the agent for choosing that action in that state
        next_state is the state the agent will be in after executing that action from that state
        done is whether the agent is entering the terminal state/ending the episode
        This function will append the objects memory with a tuple of the state converted to an array
        with the action and reward and next state as an array with the done boolean
        This function doesnt return anything
        """
        self.memory.append((
            self._state_to_array(state),
            action,
            reward,
            self._state_to_array(next_state),
            done
        ))
        LOGGER.debug(f"Adding memory: {self._state_to_array(state)}, {action}, {reward}, {self._state_to_array(next_state)}, {done}")


    def _train_batch(self):
        """
        This function takes no arguments
        This function will see if the memory samples is larger than batch_size and if not it will skip training the batch
        If there are more memory samples than batch size it will use random.sample to get a random selection of the memories
        This will then break them up into states, actions, rewards, next states, and dones
        Then leverage a pytorch library to format the data to be trained on
        The current Q values will be calculated based on the states and actions
        This function will get the next Q values based on the target network
        The loss will be calculated based on the current and target Q values
        The optimizer will then update the neural network weights using back propogation
        If the training steps are at a point to update the target network the q values will be loaded into the target network
        This function will return a float of the loss calculated by the current and target Q values
        """
        if len(self.memory) < self.batch_size:
            LOGGER.debug(f"Returning none from: {sys._getframe().f_code.co_name}")
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(dim=1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = self.loss_function(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_steps += 1

        if self.training_steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        LOGGER.debug(f"Returning {float(loss.item())} from: {sys._getframe().f_code.co_name}")
        return float(loss.item())


    def learn(self, state, action, reward, next_state, done):
        """
        This function takes 5 arguments
        state is the current state of the agent
        action is the selected action in that state
        reward is the reward given to that agent for selecting that action in that state
        next_state is the state the agent will be in after performing that action
        done is if the agent entered the terminal state
        This function will store the experiences 
        """
        self._store_experience(state, action, reward, next_state, done)
        LOGGER.debug(
            f"Learning from {sys._getframe().f_code.co_name}: "
            f"state={state}, action={action}, reward={reward}, next_state={next_state}, done={done}"
        )
        return self._train_batch()


    def decay_epsilon(self):
        """
        This function takes no arguments
        This function will either choose the minimmum epsilon value set or decay the epsilon
        """
        LOGGER.debug(f"Decaying epsilon: self.epsilon = max({self.epsilon_min}, {self.epsilon} * {self.epsilon_decay}) from {sys._getframe().f_code.co_name}")
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
