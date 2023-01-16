# This does not yet work, but I'm trying to make a DQN that can handle continuous actions

import torch
import numpy as np
from collections import deque


class DQN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.fc3 = torch.nn.Linear(16, output_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent:
    def __init__(self, input_size=1, output_size=1, lr=1e-3, memory_size=250_000):
        self.model = DQN(input_size, output_size)
        self.target_model = DQN(input_size, output_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss = torch.nn.MSELoss()
        self.state_memory = torch.zeros((memory_size, input_size))
        self.new_state_memory = torch.zeros((memory_size, input_size))
        self.reward_memory = np.zeros(memory_size)
        self.action_memory = np.zeros(memory_size, dtype=np.uint8)
        self.terminal_memory = np.zeros(memory_size)
        self.memory_counter = 0
        self.memory_size = memory_size
    
    def train(self, x, y):
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.loss(output, y)
        loss.backward()
        self.optimizer.step()
    
    def replay(self, batch_size, gamma):
        if self.memory_counter <= batch_size:
            return
        batch_states = np.random.choice(self.memory_counter, batch_size)
        batch = (self.state_memory[batch_states], self.action_memory[batch_states], self.reward_memory[batch_states], self.new_state_memory[batch_states], self.terminal_memory[batch_states])
        for i in range(batch_size):
            state = batch[0][i]
            action = batch[1][i]
            reward = batch[2][i]
            next_state = batch[3][i]
            done = batch[4][i]
            if done:
                target = torch.tensor([reward])
            else:
                target = torch.tensor([reward + gamma * torch.max(self.target_model(next_state))])
            y = self.model(state)
            

            y[0] = target

            self.train(state, y)
    
    def predict(self, x):
        return self.model(x)
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def get_action(self, x, epsilon):
        with torch.no_grad():

            if np.random.rand() <= epsilon:
                return np.array([np.random.random()*2-1])
            else:
                return np.array([self.model(x).detach().numpy().item()])
    
    def remember(self, state, action, reward, next_state, done):
        self.state_memory[self.memory_counter%self.memory_size] = state
        self.new_state_memory[self.memory_counter%self.memory_size] = next_state
        self.reward_memory[self.memory_counter%self.memory_size] = reward
        self.action_memory[self.memory_counter%self.memory_size] = action
        self.terminal_memory[self.memory_counter%self.memory_size] = done
        self.memory_counter += 1

