import gym
import torch
import pickle
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from replay_memory import ReplayMemory


class ConvNet(nn.Module):
    def __init__(self, lr, n_actions,n_quant, name, input_dims,chkpt_name,logging_dir):
        self.n_actions = n_actions

        self.n_quant = n_quant

        super(ConvNet, self).__init__()
        self.MODEL_NAME = chkpt_name#"dqn_with_uncertainty"
        self.checkpoint_dir = f"logs/{logging_dir}/network/save/{chkpt_name}-{name}"
        self.load_dir = f"logs/{logging_dir}/network/load/{chkpt_name}-{name}"
        #self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)

        # action value distribution
        self.fc_action = nn.Linear(512, n_actions * n_quant)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv_state = conv3.view(conv3.size()[0], -1)
        flat1 = F.relu(self.fc1(conv_state))
        flat2 = self.fc_action(flat1).view(-1,self.n_actions , self.n_quant)

        return flat2

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_dir)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.load_dir))


class QRAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions,n_quant, input_dims,
                 mem_size, batch_size,chkpt_name, eps_min, eps_dec,
                 replace, logging_dir):
        self.n_quant = n_quant
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace

        self.chkpt_dir = chkpt_name
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayMemory(mem_size)

        self.Z = ConvNet(self.lr, self.n_actions,self.n_quant,
                                    input_dims=self.input_dims,
                                    chkpt_name=self.chkpt_dir,
                                        name='Z',
                                        logging_dir=logging_dir)

        self.Z_tgt = ConvNet(self.lr, self.n_actions,self.n_quant,
                                    input_dims=self.input_dims,
                                    chkpt_name=self.chkpt_dir,
                                        name='Z_tgt',
                                        logging_dir=logging_dir)

        self.tau = torch.Tensor((2 * np.arange(self.Z.n_quant) + 1) / (2.0 * self.Z.n_quant)).view(1, -1)
        self.tau.to(self.Z.device)
        self.cumulative_density = torch.tensor((2 * np.arange(self.n_quant) + 1) / (2.0 * self.n_quant),device=self.Z.device, dtype=torch.float)


    def select_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation],dtype=torch.float).to(self.Z.device)
            #action = self.Z.forward(state).mean(2).max(1)[1]


            Cvar_indices = torch.tensor(range(int(self.n_quant * 0.25))).cuda()
            Cvar = torch.index_select(self.Z.forward(state), 2, Cvar_indices)
            action = Cvar.mean(2).max(1)[1]


        else:
            action = np.random.choice(self.action_space)

        return int(action)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.push(state, action, state_, reward, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample(self.batch_size)

        states = torch.tensor(state).to(self.Z.device)
        rewards = torch.tensor(reward).to(self.Z.device)
        dones = torch.tensor(done).to(self.Z.device)
        actions = torch.tensor(action).to(self.Z.device)
        states_ = torch.tensor(new_state).to(self.Z.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.Z_tgt.load_state_dict(self.Z.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.Z.save_checkpoint()
        self.Z_tgt.save_checkpoint()

    def load_models(self):
        self.Z.load_checkpoint()
        self.Z_tgt.load_checkpoint()

    def huber(self,x, k=1.0):
        return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))




    def learn(self):

        if self.memory.mem_cntr < self.batch_size:
            return
        self.replace_target_network()

        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)
        dones = dones.to(self.Z.device)
        actions = actions.to(self.Z.device)
        states = states.float().to(self.Z.device)
        states_ = states_.float().to(self.Z.device)

        #actions = actions.unsqueeze(dim=-1).expand(-1, -1, self.n_quant)



        indices = np.arange(self.batch_size)
        theta = self.Z.forward(states)[indices, actions,:]
        #print('theta:', theta.size())
        #print('states_:', states_)
        #print('rewards:', rewards.size())



        Znext = self.Z_tgt(states_).detach()

        Cvar_indices = torch.tensor(range(int(self.n_quant*0.25))).cuda()
        Cvar = torch.index_select(Znext, 2, Cvar_indices)
        Znext_max = Znext[np.arange(self.batch_size), Cvar.mean(2).max(1)[1]]
        #Znext_max = Znext[np.arange(self.batch_size), Znext.mean(2).max(1)[1]]
        #print('Zmax:', Znext.size())
        #print('Cvar:', Cvar.size())

        #print('dones:', dones.size())

        mask = torch.logical_not(dones).float().cuda()


        Ttheta = torch.add(rewards.cuda(),torch.mul(torch.mul(mask,self.gamma),Znext_max.cuda()))

        diff = Ttheta.t().unsqueeze(-1) - theta


        loss = self.huber(diff) * torch.abs(self.cumulative_density.view(1, -1) - (diff < 0).to(torch.float))
        loss = loss.transpose(0, 1)
        loss = loss.mean(1).sum(-1).mean()

        self.Z.optimizer.zero_grad()
        loss.backward()
        self.Z.optimizer.step()

        self.decrement_epsilon()
        self.learn_step_counter += 1

