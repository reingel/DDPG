import numpy as np
import torch
import torch.nn as nn


class dpg(object):
    def __init__(self, gamma=0.99, period=10):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.da = 0.01
        self.gamma = gamma
        self.target_update_period = period
        self.n_train = 0
        # behavioral policy
        self.actor = actor().to(self.device)
        self.critic = critic().to(self.device)
        self.actor_optim = torch.optim.SGD(self.actor.parameters(), lr=0.001)
        self.critic_optim = torch.optim.SGD(self.critic.parameters(), lr=0.001)
        # target policy
        self.actorTgt = actor().to(self.device)
        self.criticTgt = critic().to(self.device)
        self.actorTgt_optim = torch.optim.SGD(self.actorTgt.parameters(), lr=0.001)
        self.criticTgt_optim = torch.optim.SGD(self.criticTgt.parameters(), lr=0.001)
        # copy parameters
        self.copy_params()

    def copy_params(self):
        self.actorTgt.load_state_dict(self.actor.state_dict())
        self.criticTgt.load_state_dict(self.critic.state_dict())
        print(f'target network updated ----------------')

    def __call__(self, s):
        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        a = self.actor(s)
        return a

    def train(self, minibatch):
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()

        states = torch.tensor(list(map(lambda x: x.s, minibatch)), dtype=torch.float32).to(self.device)
        actions = torch.tensor(list(map(lambda x: x.a, minibatch)), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(list(map(lambda x: x.r, minibatch)), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(list(map(lambda x: x.s1, minibatch)), dtype=torch.float32).to(self.device)

        # loss for critic network
        next_actions = self.actorTgt(next_states)
        q1 = self.criticTgt(next_states, next_actions)
        y = rewards + self.gamma * q1.detach()
        q = self.critic(states, actions)
        Jc = (y - q)**2
        Jc_mean = torch.mean(Jc)

        # loss for actor network
        actions = self.actor(states)
        q = self.critic(states, actions.detach())
        qd = self.critic(states, actions.detach() + self.da)
        grad_a_Q = (qd - q) / self.da
        Ja = -actions * grad_a_Q.detach()
        Ja_mean = torch.mean(Ja)
            
        Jc_mean.backward()
        Ja_mean.backward()

        self.critic_optim.step()
        self.actor_optim.step()

        self.n_train += 1
        if(self.n_train % self.target_update_period == 0):
            self.copy_params()


class actor(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, s):
        a = self.net(s)
        return a


class critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, s, a):
        x = torch.cat((s, a), axis=1)
        q = self.net(x)
        return q
