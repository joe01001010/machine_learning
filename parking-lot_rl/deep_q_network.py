import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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

    def forward(self, x):
        return self.network(x)


class DeepQNetwork:
    def __init__(self,
        state_size=6,
        action_size=4,
        alpha=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        memory_size=10000,
        batch_size=32,
        target_update_frequency=25
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.state_method = "get_dqn_state"

        self.memory = deque(maxlen=memory_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = _QModel(state_size, action_size).to(self.device)
        self.target_network = _QModel(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)
        self.loss_function = nn.MSELoss()

        self.training_steps = 0


    def _state_to_array(self, state):
        """
        This function takes one argument
        state is the current state of the agent
        This function will conver the state into a numpy array
        """
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
        return self._train_batch()


    def decay_epsilon(self):
        """
        This function takes no arguments
        This function will either choose the minimmum epsilon value set or decay the epsilon
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)