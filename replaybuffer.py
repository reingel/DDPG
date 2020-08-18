import random
import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition', ('s', 'a', 'r', 's1', 'done'))

class replay_buffer(object):
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
    
    def push(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
    
    def sample(self, batch_size=100):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

rb = replay_buffer()
