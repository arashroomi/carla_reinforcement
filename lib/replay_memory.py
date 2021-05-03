import numpy as np
import random
import torch

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.mem_cntr = 0

    def push(self, state, action, next_state, reward, done):
        transition = torch.Tensor([state]), torch.Tensor([action]), torch.Tensor([next_state]), torch.Tensor(
            [reward]), torch.Tensor([done])
        self.memory.append(transition)
        self.mem_cntr += 1
        if len(self.memory) > self.capacity: del self.memory[0]

    def sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*sample)

        batch_state = torch.cat(batch_state)
        batch_action = torch.LongTensor(batch_action)
        batch_reward = torch.cat(batch_reward)
        batch_done = torch.cat(batch_done)
        batch_next_state = torch.cat(batch_next_state)

        return batch_state, batch_action, batch_reward.unsqueeze(1), batch_next_state, batch_done.unsqueeze(1)

    def __len__(self):
        return len(self.memory)