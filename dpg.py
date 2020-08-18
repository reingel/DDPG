import numpy as np
import torch
import torch.nn as nn


class dpg(object):
    def __init__(self, state_dim, action_dim, max_action, max_episode, max_step):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # env param
        self.max_action = max_action
        self.max_episode = max_episode
        self.max_step = max_step
        # sim param
        self.da = 0.01
        self.gamma = 0.99
        self.n_train = 0
        self.alpha = 0.01
        # behavioral policy
        self.actor = actor(state_dim, action_dim).to(self.device)
        self.critic = critic(state_dim, action_dim).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.005)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.005)
        # target policy
        self.actorTgt = actor(state_dim, action_dim).to(self.device)
        self.criticTgt = critic(state_dim, action_dim).to(self.device)
        # copy parameters
        self.update_params()

    def update_params(self):
        # self.actorTgt.load_state_dict(self.actor.state_dict())
        # self.criticTgt.load_state_dict(self.critic.state_dict())
        for param, paramTgt in zip(self.actor.parameters(), self.actorTgt.parameters()):
            paramTgt.data.copy_(self.alpha * param.data + (1 - self.alpha) * paramTgt.data)
        for param, paramTgt in zip(self.critic.parameters(), self.criticTgt.parameters()):
            paramTgt.data.copy_(self.alpha * param.data + (1 - self.alpha) * paramTgt.data)
    
    def gen_noise(self, episode):
        self.noise = np.random.randn(self.max_step) * np.exp(-1.0 * episode / self.max_episode)
        self.noise = torch.tensor(self.noise, dtype=torch.float32).to(self.device)

    def __call__(self, s, step):
        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        a = self.actor(s) * self.max_action
        return a

    def train(self, minibatch):
        states = torch.tensor(list(map(lambda x: x.s, minibatch)), dtype=torch.float32).to(self.device)
        actions = torch.tensor(list(map(lambda x: x.a, minibatch)), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(list(map(lambda x: x.r, minibatch)), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(list(map(lambda x: x.s1, minibatch)), dtype=torch.float32).to(self.device)
        dones = torch.tensor(list(map(lambda x: not x.done, minibatch)), dtype=torch.float32).to(self.device)

        # loss for critic network
        next_actions = self.actorTgt(next_states)
        q1 = self.criticTgt(next_states, next_actions)
        y = rewards + self.gamma * q1.detach() * dones
        q = self.critic(states, actions)
        Jc = (y - q)**2
        Jc_mean = torch.mean(Jc)

        self.critic_optim.zero_grad()
        Jc_mean.backward()
        self.critic_optim.step()

        # loss for actor network
        actions = self.actor(states)
        q = self.critic(states, actions)
        # q = self.critic(states, actions.detach())
        # qd = self.critic(states, actions.detach() + self.da)
        # grad_a_Q = (qd - q) / self.da
        # Ja = -actions * grad_a_Q.detach()
        Ja = -q
        Ja_mean = torch.mean(Ja)

        self.actor_optim.zero_grad()
        Ja_mean.backward()
        self.actor_optim.step()

        self.n_train += 1
        # if(self.n_train % self.target_update_period == 0):
        self.update_params()


class actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )

    def forward(self, s):
        a = self.net(s)
        return a


class critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, s, a):
        x = torch.cat((s, a), axis=1)
        q = self.net(x)
        return q
