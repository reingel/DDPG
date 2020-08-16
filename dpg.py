import numpy as np
import torch
import torch.nn as nn
class dpg(object):
    def __init__(self, gamma=0.9, period=50):
        self.da = 0.01
        self.gamma = gamma
        self.target_update_period = period
        self.n_train = 0
        # behavioral policy
        self.actor = actor()
        self.critic = critic()
        self.actor_optim = torch.optim.SGD(self.actor.parameters(), lr=0.001)
        self.critic_optim = torch.optim.SGD(self.critic.parameters(), lr=0.001)
        # target policy
        self.actorTgt = actor()
        self.criticTgt = critic()
        self.actorTgt_optim = torch.optim.SGD(self.actorTgt.parameters(), lr=0.001)
        self.criticTgt_optim = torch.optim.SGD(self.criticTgt.parameters(), lr=0.001)
        # copy parameters
        self.copy_params()

    def copy_params(self):
        self.actorTgt.load_state_dict(self.actor.state_dict())
        self.criticTgt.load_state_dict(self.critic.state_dict())

    def __call__(self, s):
        a = self.actor(s)
        return a

    def train(self, minibatch):
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()

        N = len(minibatch)
        sum_Jc = 0.0
        sum_Ja = 0.0
        for transition in minibatch:
            s, a, r, s1 = transition

            # loss for critic network
            a1 = self.actorTgt(s1)
            q1 = self.criticTgt(s1, a1)
            y = r + self.gamma * q1.detach()
            q = self.critic(s, a)
            Jc = (y - q)**2
            sum_Jc += Jc

            # loss for actor network
            a = self.actor(s)
            q = self.critic(s, a.detach())
            qd = self.critic(s, a.detach() + self.da)
            grad_a_Q = (qd - q) / self.da
            Ja = a * grad_a_Q.detach()
            sum_Ja += Ja
        sum_Jc /= N
        sum_Ja /= N
            
        sum_Jc.backward()
        sum_Ja.backward()
        self.critic_optim.step()
        self.actor_optim.step()

        self.n_train += 1
        if(self.n_train % self.target_update_period == 0):
            self.copy_params()


n_hid_act = 128
n_hid_cri = 128
class actor(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, n_hid_act),
            nn.ReLU(),
            nn.Linear(n_hid_act, 1)
        )

    def forward(self, s):
        s = torch.tensor(s, dtype=torch.float32)
        a = self.net(s)
        return a


class critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2+1, n_hid_cri),
            nn.ReLU(),
            nn.Linear(n_hid_cri, 1)
        )

    def forward(self, s, a):
        x = np.concatenate((s, a.item()), axis=None)
        x = torch.tensor(x, dtype=torch.float32)
        q = self.net(x)
        return q
