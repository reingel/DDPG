import numpy as np
import gym
import dpg
from replaybuffer import Transition
from replaybuffer import rb

verbose = False

epsilon = 0.9
decay = 0.99
eps_min = 0.05

env = gym.make('MountainCarContinuous-v0').unwrapped
agt = dpg.dpg()

file = open('result.txt', 'w+')
file.write(f'episode, step, return, epsilon' + '\n')
file.close()

for episode in range(1000):
    R = 0.0
    s = env.reset()
    for step in range(10000):
        if np.random.random() < epsilon:
            a = np.array([np.random.random() * 2.0 - 1.0])
        else:
            a = agt(s)
            a = np.array([a.item()])
            a = np.clip(a, -1.0, 1.0)

        s1, r, done, _ = env.step(a)

        transition = Transition(s, a, r, s1)
        rb.push(transition)

        buffer_size = len(rb)
        if buffer_size >= 1000 and step % 1000 == 0:
            print(f'training: buffer_size={buffer_size}')
            agt.train(rb.sample(batch_size=100))

        if (episode+1) % 10 == 0 and step < 2000:
            env.render()

        s = s1.copy()
        R += r

        if(verbose): print(transition)
        if(step % 100 == 0):
            print(f'episode = {episode+1}, step = {step}, return = {R:.3f}, epsilon = {epsilon:.3f}')

        if done:
            break

    file = open('result.txt', 'a')
    file.write(f'{episode+1}, {step}, {R:.3f}, {epsilon:.3f}' + '\n')
    file.close()
    epsilon = epsilon*decay if epsilon > eps_min else eps_min

env.close()
